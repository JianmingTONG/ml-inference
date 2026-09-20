"""HE-friendly MNIST MLP for the CROSS submission.

The harness model (``harness/mnist/model.py``) is 784-128-64-10 with ReLU.
CKKS evaluates polynomials, not ReLU, and CROSS's activation registry accepts
only a bare ``x**(2**k)`` squaring chain (``jaxite_word.nn.register_activation``
rejects anything else, because reaching a non-power-of-two exponent would need
an operand mod-switched down to meet the other and the only level-changing
primitive is Rescale, which also divides the scale).

So the one documented change to the harness architecture is ReLU -> x**2.
The topology, the layer widths and the input/output contract are unchanged.

A bare square is hard to train directly: activations square every layer and
the scale explodes.  ``TrainHEMLP`` therefore carries a BatchNorm1d before
each square while the statistics are still moving, and ``to_inference_model``
folds each BatchNorm into the preceding Linear.  That is an exact algebraic
rewrite -- BN(Wx+b) = gamma*(Wx+b-mu)/sqrt(var+eps) + beta is itself affine --
so the inference model computes the same function with no BatchNorm and no
branch, which is what ``torch.fx`` (and therefore ``nn.vectorize``) accepts.
"""
from __future__ import annotations

import copy

import torch
from torch import nn


class TrainHEMLP(nn.Module):
  """Training-time model: Linear -> BN -> square, twice, then Linear."""

  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(28 * 28, 128)
    self.bn1 = nn.BatchNorm1d(128)
    self.fc2 = nn.Linear(128, 64)
    self.bn2 = nn.BatchNorm1d(64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    x = x.view(-1, 28 * 28)
    x = self.bn1(self.fc1(x))
    x = x * x
    x = self.bn2(self.fc2(x))
    x = x * x
    return self.fc3(x)


class HEMLP(nn.Module):
  """Inference-shaped model: three Linears and two squares, nothing else.

  This is the module ``nn.vectorize`` traces.  ``x * x`` is a ``call_function``
  node the frontend claims as a polynomial activation; writing ``nn.ReLU``
  here would vectorize to the same x**2 but emit a substitution warning, so
  the square is written out literally to keep the traced graph and the trained
  function identical.
  """

  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(28 * 28, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    x = self.fc1(x)
    x = x * x
    x = self.fc2(x)
    x = x * x
    return self.fc3(x)


def _fold_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> tuple[torch.Tensor, torch.Tensor]:
  """Return the (weight, bias) of the single Linear equal to bn(linear(x))."""
  scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
  weight = linear.weight * scale.unsqueeze(1)
  bias = (linear.bias - bn.running_mean) * scale + bn.bias
  return weight, bias


def to_inference_model(trained: TrainHEMLP) -> HEMLP:
  """Fold both BatchNorms into their preceding Linear."""
  trained = copy.deepcopy(trained).eval()
  model = HEMLP()
  with torch.no_grad():
    w1, b1 = _fold_bn(trained.fc1, trained.bn1)
    model.fc1.weight.copy_(w1)
    model.fc1.bias.copy_(b1)
    w2, b2 = _fold_bn(trained.fc2, trained.bn2)
    model.fc2.weight.copy_(w2)
    model.fc2.bias.copy_(b2)
    model.fc3.weight.copy_(trained.fc3.weight)
    model.fc3.bias.copy_(trained.fc3.bias)
  return model.eval()


# The harness exports raw ToTensor pixels in [0, 1]; both the harness model
# (`harness/mnist/test.py::predict`) and this submission apply the MNIST
# normalization on the client before anything is encrypted.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def normalize(pixels):
  """Apply the client-side MNIST normalization to raw [0,1] pixels."""
  return (pixels - MNIST_MEAN) / MNIST_STD
