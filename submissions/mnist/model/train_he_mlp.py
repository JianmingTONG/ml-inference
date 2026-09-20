#!/usr/bin/env python3
"""Train the HE-friendly MNIST MLP and export the inference-shaped weights.

Uses the same data pipeline as ``harness/mnist/mnist.py``: torchvision MNIST,
ToTensor + Normalize((0.1307,), (0.3081,)), 80/20 train/val split, Adam.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from he_mlp import TrainHEMLP, to_inference_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'he_mlp_weights.pth')
DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.dirname(HERE)), 'harness', 'mnist', 'data')


def loaders(data_dir, batch_size):
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)),
  ])
  full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
  test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
  train_size = int(0.8 * len(full))
  train, val = random_split(full, [train_size, len(full) - train_size])
  return (DataLoader(train, batch_size=batch_size, shuffle=True),
          DataLoader(val, batch_size=batch_size, shuffle=False),
          DataLoader(test, batch_size=batch_size, shuffle=False))


@torch.no_grad()
def accuracy(model, loader):
  model.eval()
  correct = total = 0
  for data, target in loader:
    predicted = model(data.view(-1, 784)).argmax(dim=1)
    correct += (predicted == target).sum().item()
    total += target.size(0)
  return correct / total


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--epochs', type=int, default=30)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--data_dir', default=DEFAULT_DATA)
  parser.add_argument('--out', default=DEFAULT_OUT)
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  train_loader, val_loader, test_loader = loaders(args.data_dir, args.batch_size)

  model = TrainHEMLP()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

  best = 0.0
  for epoch in range(args.epochs):
    model.train()
    running = 0.0
    for data, target in train_loader:
      optimizer.zero_grad()
      loss = criterion(model(data.view(-1, 784)), target)
      loss.backward()
      # Squares make the loss surface steep; clipping keeps early epochs from
      # diverging outright.
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      running += loss.item()
    scheduler.step()
    val = accuracy(model, val_loader)
    print(f'epoch {epoch + 1}/{args.epochs}  loss {running / len(train_loader):.4f}  '
          f'val {val * 100:.2f}%', flush=True)
    if val > best:
      best = val
      torch.save(model.state_dict(), args.out + '.train')

  model.load_state_dict(torch.load(args.out + '.train'))
  inference = to_inference_model(model)

  # The fold must be exact, or the encrypted model computes a different
  # function from the one that was trained.
  model.eval()
  sample = torch.randn(8, 784)
  with torch.no_grad():
    deviation = (model(sample) - inference(sample)).abs().max().item()
  print(f'batchnorm fold max |deviation|: {deviation:.3e}')
  assert deviation < 1e-4, 'BatchNorm folding changed the function'

  torch.save(inference.state_dict(), args.out)
  print(f'test accuracy (train-shaped)     : {accuracy(model, test_loader) * 100:.2f}%')
  print(f'test accuracy (inference-shaped) : {accuracy(inference, test_loader) * 100:.2f}%')
  print(f'wrote {args.out}')


if __name__ == '__main__':
  main()
