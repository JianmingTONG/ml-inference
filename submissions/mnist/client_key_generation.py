#!/usr/bin/env python3
"""Stage 2.2 -- client: generate the CKKS key pair.

The ring is not chosen here and is not a tunable: ``packing.pack`` derives it
from the program's slot demand and the multiplicative depth it actually emits,
and the pair is sampled at that ring's own error width. The client packs the
architecture with untrained weights, so no model secret is needed to know which
ring to generate for.
"""
import os
import sys
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import cross_task as ct  # noqa: E402


def main():
    size = ct.parse_size(sys.argv)
    paths = ct.Paths(size)
    # The server stage that follows receives no arguments; tell it which
    # instance this run is for.
    ct.record_active_size(size)

    started = time.perf_counter()
    packed = ct.build_client_packed()
    packed_at = time.perf_counter()
    keys = ct.generate_keys(packed.ring_config)
    keygen_at = time.perf_counter()
    ct.save_keys(keys, paths.pubkeydir, paths.seckeydir)

    ring = packed.ring_config
    print(f"         [client] ring: degree={ring.degree} slots={ring.num_slots} "
          f"num_q={len(ring.q_towers)} num_p={len(ring.p_towers)} dnum={ring.dnum} "
          f"depth={packed.depth} security={ring.target.bits}-bit "
          f"{ring.target.cost_model.value}")
    ct.append_timing(paths, 'client_key_generation', {
        'pack_s': packed_at - started,
        'keygen_s': keygen_at - packed_at,
        'save_s': time.perf_counter() - keygen_at,
    })


if __name__ == '__main__':
    main()
