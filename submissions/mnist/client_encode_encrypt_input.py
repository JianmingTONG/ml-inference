#!/usr/bin/env python3
"""Stage 6 -- client: slot-encode and encrypt each input.

Runs on the CPU with a bare ``CKKSContext``: the client holds the key pair and
the slot layout and nothing else. It never builds a Mapping, never sees the
model's schedule, and never touches the accelerator.
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
    manifest = ct.read_manifest(paths.manifest_file)

    started = time.perf_counter()
    packed = ct.build_client_packed()
    ct.verify_against_manifest(packed, manifest)

    import mapping as mapping_mod
    from ckks_ctx import CKKSContext

    keys = ct.load_keys(paths.pubkeydir, paths.seckeydir)
    parameters = mapping_mod.ring_runtime_parameters(packed.ring_config, keys=keys)
    ctx = CKKSContext(parameters)
    setup_s = time.perf_counter() - started

    prepared = paths.iointermdir / 'normalized_input.npy'
    if not prepared.exists():
        raise SystemExit(f'[client] {prepared} not found; run stage 5 first')
    normalized = np.load(prepared)
    if normalized.ndim != 2 or normalized.shape[1] != 784:
        raise SystemExit(f'[client] expected (N, 784) input, got {normalized.shape}')
    if len(normalized) != paths.batch_size:
        raise SystemExit(
            f'[client] {len(normalized)} samples but the {paths.name} instance '
            f'expects {paths.batch_size}')
    scale = float(manifest['input_scale'])

    encrypt_started = time.perf_counter()
    for index, sample in enumerate(normalized):
        ciphertext = ctx.encrypt_slots(packed.pack(sample), scale=scale)
        ct.save_ciphertext(paths.ctxtupdir / f'cipher_input_{index}.npz', ciphertext)
    encrypt_s = time.perf_counter() - encrypt_started

    ct.append_timing(paths, 'client_encode_encrypt_input', {
        'context_setup_s': setup_s,
        'encrypt_s': encrypt_s,
        'per_sample_s': encrypt_s / max(1, len(normalized)),
        'samples': int(len(normalized)),
    })


if __name__ == '__main__':
    main()
