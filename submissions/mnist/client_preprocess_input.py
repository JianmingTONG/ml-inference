#!/usr/bin/env python3
"""Stage 5 -- client: cleartext preprocessing of the input.

The harness exports raw ToTensor pixels in [0, 1]; the model was trained on
Normalize((0.1307,), (0.3081,)) data, exactly as the harness's own predictor
applies in ``harness/mnist/test.py``. Normalizing here keeps that affine step
on the client, where it is free, instead of spending ciphertext depth on it.

This is a documented pre-encryption preprocessing step (see README).
"""
import os
import sys
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import numpy as np  # noqa: E402

import cross_task as ct  # noqa: E402


def main():
    size = ct.parse_size(sys.argv)
    paths = ct.Paths(size)
    started = time.perf_counter()
    if not paths.test_input_file.exists():
        raise SystemExit(
            f'[client] {paths.test_input_file} not found; the harness input '
            'generation (stage 4) must run first')
    pixels = ct.read_pixels(paths.test_input_file)
    if len(pixels) != paths.batch_size:
        raise SystemExit(
            f'[client] dataset has {len(pixels)} samples, the {paths.name} '
            f'instance expects {paths.batch_size}')
    normalized = (pixels - 0.1307) / 0.3081
    paths.iointermdir.mkdir(parents=True, exist_ok=True)
    ct.atomic_write_bytes(paths.iointermdir / 'normalized_input.npy',
                          lambda handle: np.save(handle, normalized))
    ct.append_timing(paths, 'client_preprocess_input',
                     {'elapsed_s': time.perf_counter() - started,
                      'samples': int(len(pixels))})


if __name__ == '__main__':
    main()
