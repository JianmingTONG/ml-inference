#!/usr/bin/env python3
"""Stage 9 -- client: turn decrypted logits into labels.

argmax over the ten decrypted logits. This is a documented post-decryption
step (see README); no part of the model is evaluated here.
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
    decoded = paths.iointermdir / 'decrypted_logits.npy'
    if not decoded.exists():
        raise SystemExit(f'[client] {decoded} not found; run stage 8 first')
    logits = np.real(np.load(decoded))
    if not np.isfinite(logits).all():
        raise SystemExit('[client] decrypted logits contain non-finite values; '
                         'the ciphertext most likely exhausted its modulus chain')
    ct.write_labels(paths.predictions_file, np.argmax(logits, axis=1))
    ct.append_timing(paths, 'client_postprocess',
                     {'elapsed_s': time.perf_counter() - started})


if __name__ == '__main__':
    main()
