#!/usr/bin/env python3
"""Stage 8 -- client: decrypt and slot-decode each result."""
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

    scale = float(manifest['output_scale'])
    count = paths.batch_size
    absent = [index for index in range(count)
              if not (paths.ctxtdowndir / f'cipher_result_{index}.npz').exists()]
    if absent:
        raise SystemExit(
            f'[client] {len(absent)} of {count} result ciphertexts missing from '
            f'{paths.ctxtdowndir} (first: {absent[0]}); did stage 7 complete?')

    decrypt_started = time.perf_counter()
    logits = []
    for index in range(count):
        payload, meta = ct.load_ciphertext_payload(
            paths.ctxtdowndir / f'cipher_result_{index}.npz')
        ciphertext = ct.build_polynomial(
            payload, meta['moduli'], meta['degree_layout'], meta['precision'],
            scale=meta.get('scale') if meta.get('scale') is not None else scale,
            nsd=meta.get('nsd') if meta.get('nsd') is not None
                else manifest.get('output_nsd'))
        logits.append(packed.unpack(np.asarray(ctx.decrypt_slots(ciphertext, scale=scale))))
    decrypt_s = time.perf_counter() - decrypt_started

    logits = np.asarray(logits)
    if logits.shape != (count, 10):
        raise SystemExit(f'[client] decoded logits have shape {logits.shape}, '
                         f'expected ({count}, 10)')
    paths.iointermdir.mkdir(parents=True, exist_ok=True)
    ct.atomic_write_bytes(paths.iointermdir / 'decrypted_logits.npy',
                          lambda handle: np.save(handle, logits))
    ct.append_timing(paths, 'client_decrypt_decode', {
        'context_setup_s': setup_s,
        'decrypt_s': decrypt_s,
        'per_sample_s': decrypt_s / max(1, count),
        'samples': count,
    })


if __name__ == '__main__':
    main()
