#!/usr/bin/env python3
"""Fold BatchNorm out of a training checkpoint into the inference model.

``train_he_mlp.py`` writes the inference-shaped weights when it finishes; this
exports the same artifact from the best checkpoint so far, and re-checks that
the fold is exact.
"""
import argparse, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from he_mlp import TrainHEMLP, to_inference_model

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', default=os.path.join(HERE, 'he_mlp_weights.pth.train'))
  parser.add_argument('--out', default=os.path.join(HERE, 'he_mlp_weights.pth'))
  args = parser.parse_args()

  trained = TrainHEMLP()
  trained.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
  trained.eval()
  inference = to_inference_model(trained)

  sample = torch.randn(16, 784)
  with torch.no_grad():
    deviation = (trained(sample) - inference(sample)).abs().max().item()
  if deviation >= 1e-4:
    raise SystemExit(f'BatchNorm fold changed the function (max dev {deviation:.3e})')
  torch.save(inference.state_dict(), args.out)
  print(f'fold max |deviation|: {deviation:.3e}')
  print(f'wrote {args.out}')


if __name__ == '__main__':
  main()
