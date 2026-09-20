#!/usr/bin/env python3
"""Train the depth-1 linear MNIST classifier (the N=4096 variant)."""
from __future__ import annotations
import argparse, os, sys
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from he_mlp_linear import TrainLinearHE, to_inference_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'he_mlp_linear_weights.pth')
DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(HERE))), 'harness', 'mnist', 'data')


def loaders(data_dir, bs):
  tf = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
  full = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
  test = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
  n = int(0.8 * len(full))
  tr, va = random_split(full, [n, len(full) - n])
  return (DataLoader(tr, batch_size=bs, shuffle=True),
          DataLoader(va, batch_size=bs), DataLoader(test, batch_size=bs))


@torch.no_grad()
def accuracy(model, loader):
  model.eval(); c = t = 0
  for X, y in loader:
    c += (model(X.view(-1, 784)).argmax(1) == y).sum().item(); t += y.size(0)
  return c / t


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--epochs', type=int, default=30)
  ap.add_argument('--batch_size', type=int, default=256)
  ap.add_argument('--lr', type=float, default=2e-3)
  ap.add_argument('--seed', type=int, default=42)
  ap.add_argument('--data_dir', default=DEFAULT_DATA)
  ap.add_argument('--out', default=DEFAULT_OUT)
  a = ap.parse_args()

  torch.manual_seed(a.seed); torch.set_num_threads(min(16, os.cpu_count() or 8))
  tr, va, te = loaders(a.data_dir, a.batch_size)
  model = TrainLinearHE()
  opt = torch.optim.Adam(model.parameters(), lr=a.lr)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
  crit = nn.CrossEntropyLoss()
  best = 0.0
  for ep in range(a.epochs):
    model.train()
    for X, y in tr:
      opt.zero_grad(); crit(model(X.view(-1, 784)), y).backward(); opt.step()
    sched.step()
    v = accuracy(model, va)
    print(f'epoch {ep+1}/{a.epochs}  val {v*100:.2f}%', flush=True)
    if v > best:
      best = v; torch.save(model.state_dict(), a.out + '.train')
  model.load_state_dict(torch.load(a.out + '.train'))
  inf = to_inference_model(model); model.eval()
  s = torch.randn(8, 784)
  with torch.no_grad():
    ref = model(s)
    # Relative, not absolute: this model's logits are large enough that an
    # absolute 1e-4 bound would fail on float32 rounding alone.
    dev = ((ref - inf(s)).abs().max() / ref.abs().max()).item()
  print(f'batchnorm fold max relative deviation: {dev:.3e}')
  assert dev < 1e-5, 'fold changed the function'
  torch.save(inf.state_dict(), a.out)
  print(f'test accuracy (inference-shaped): {accuracy(inf, te)*100:.2f}%')
  print(f'wrote {a.out}')


if __name__ == '__main__':
  main()
