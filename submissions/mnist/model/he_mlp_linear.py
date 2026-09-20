"""Depth-1 MNIST classifier: a single Linear(784, 10), no activation.

This is the only architecture CROSS can place on **ring degree 4096**, the
ring Lattica-ai's submission uses.

Reaching it needs three things at once, all through `packing.PackingPolicy`:

    register_word_size = 19   # so aux_mod_size = 18..20 instead of 31
    scaling_mod_size   = 34   # the smallest the N=4096 prime search sustains
    depth              = 1    # one matvec; num_q = 2

which yields 4 Q towers and 2 P towers, log2(QP) = 100.1 against the 109-bit
ceiling the HE Standard gives at N = 4096 -- and `he_params.qp_is_secure`
confirms 128-bit classical.

Why only depth 1. CROSS stores one logical CKKS scale as two RNS primes
(`composite_degree = 2` is enforced), and an NTT at degree N needs primes
congruent to 1 mod 2N. Depth 1 therefore costs 4 Q primes, already ~66 bits;
depth 2 costs 6 and lands at ~100 bits of Q alone, which leaves nothing for
the key-switching modulus P under a 109-bit ceiling. Lattica fit a two-layer
network with a square into 106 bits because their 61- and 45-bit moduli are
native 64-bit arithmetic: two primes where CROSS needs four.

So at Lattica's ring, CROSS runs a linear classifier and they run an MLP.
That is the cost of 32-bit lanes, stated plainly.
"""
from __future__ import annotations

import copy

import torch
from torch import nn


class TrainLinearHE(nn.Module):
  """Training-time model: BatchNorm then a single Linear."""

  def __init__(self):
    super().__init__()
    self.bn = nn.BatchNorm1d(28 * 28)
    self.fc = nn.Linear(28 * 28, 10)

  def forward(self, x):
    return self.fc(self.bn(x.view(-1, 28 * 28)))


class LinearHE(nn.Module):
  """Inference-shaped model: one Linear, nothing else. Depth 1."""

  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(28 * 28, 10)

  def forward(self, x):
    return self.fc(x)


def to_inference_model(trained: TrainLinearHE) -> LinearHE:
  """Fold the input BatchNorm into the Linear -- exact, both are affine."""
  trained = copy.deepcopy(trained).eval()
  model = LinearHE()
  with torch.no_grad():
    scale = trained.bn.weight / torch.sqrt(trained.bn.running_var + trained.bn.eps)
    weight = trained.fc.weight * scale.unsqueeze(0)
    shift = trained.bn.bias - trained.bn.running_mean * scale
    model.fc.weight.copy_(weight)
    model.fc.bias.copy_(trained.fc.bias + trained.fc.weight @ shift)
  return model.eval()
