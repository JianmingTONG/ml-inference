"""Shallow HE-friendly MNIST MLP: 784 -> 50 -> 10 with one square.

Why a second architecture. CROSS's ring is derived from the multiplicative
depth the program emits, and the depth ladder is coarse:

    depth 1  ->  N = 8192    (log2(QP) = 183)
    depth 2  ->  N = 16384   (243)
    depth 3  ->  N = 16384   (365)
    depth 5  ->  N = 32768   (485)   <- the 784-128-64-10 model in he_mlp.py

The harness topology with two squares emits depth 5 and lands on N = 32768.
Lattica-ai's submission uses 784 -> 50 -> 10 with a single square, which emits
depth 3 and fits N = 16384 -- half the ring, and 8 Q towers instead of 12.

This module is that architecture, so the two can be measured against each
other on the same accelerator.

N = 4096, which Lattica actually runs, is not reachable from CROSS at any
depth: its 32-bit register word keeps every native modulus below 2^31 and one
60-bit CKKS scale is stored as two ~30-bit primes (`composite_degree=2` is
enforced in `he_params`), so even a single matvec needs log2(QP) = 183 against
the 109-bit ceiling at N = 4096. Lattica reaches 106 bits with 61- and 45-bit
moduli -- native 64-bit arithmetic, which the TPU backend does not have.
"""
from __future__ import annotations

import copy

import torch
from torch import nn


class TrainShallowHEMLP(nn.Module):
  """Training-time model: Linear -> BN -> square -> Linear."""

  def __init__(self, hidden: int = 50):
    super().__init__()
    self.fc1 = nn.Linear(28 * 28, hidden)
    self.bn1 = nn.BatchNorm1d(hidden)
    self.fc2 = nn.Linear(hidden, 10)

  def forward(self, x):
    x = x.view(-1, 28 * 28)
    x = self.bn1(self.fc1(x))
    x = x * x
    return self.fc2(x)


class ShallowHEMLP(nn.Module):
  """Inference-shaped model: two Linears and one square, nothing else."""

  def __init__(self, hidden: int = 50):
    super().__init__()
    self.fc1 = nn.Linear(28 * 28, hidden)
    self.fc2 = nn.Linear(hidden, 10)

  def forward(self, x):
    x = self.fc1(x)
    x = x * x
    return self.fc2(x)


def _fold_bn(linear: nn.Linear, bn: nn.BatchNorm1d):
  """Return the (weight, bias) of the single Linear equal to bn(linear(x))."""
  scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
  return linear.weight * scale.unsqueeze(1), (linear.bias - bn.running_mean) * scale + bn.bias


def to_inference_model(trained: TrainShallowHEMLP) -> ShallowHEMLP:
  """Fold the BatchNorm into its preceding Linear."""
  trained = copy.deepcopy(trained).eval()
  hidden = trained.fc1.out_features
  model = ShallowHEMLP(hidden)
  with torch.no_grad():
    w1, b1 = _fold_bn(trained.fc1, trained.bn1)
    model.fc1.weight.copy_(w1)
    model.fc1.bias.copy_(b1)
    model.fc2.weight.copy_(trained.fc2.weight)
    model.fc2.bias.copy_(trained.fc2.bias)
  return model.eval()
