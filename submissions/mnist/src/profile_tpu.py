#!/usr/bin/env python3
"""Profile the CROSS encrypted-MNIST benchmark task on local TPU v6e chips.

Sweeps device counts and per-device (local) batch, and for each configuration
reports steady-state ``Mapping.execute`` latency, per-inference latency and
throughput.  Every configuration is checked against the cleartext model on the
same inputs first, so what is timed is demonstrably the benchmark computation
and not a degenerate one.

Each configuration is a separate ``Mapping``: device topology, global batch and
degree layout are static compilation inputs in CROSS, so changing any of them
requires recompiling.  ``Mapping.release`` is called between configurations --
a materialized Mapping pins its executable, evaluation keys and encoded
constants in HBM, and dropping the Python reference alone frees none of it.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

SUBMISSION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUBMISSION_ROOT / 'src'))
import cross_task as ct  # noqa: E402

import numpy as np  # noqa: E402


def cleartext_reference(model, inputs):
  import torch

  with torch.no_grad():
    return model(torch.tensor(np.asarray(inputs), dtype=torch.float64)).numpy()


def load_inputs(count):
  """Real normalized MNIST samples when the harness has exported them."""
  for size in (3, 2, 1, 0):
    path = ct.Paths(size).test_input_file
    if path.exists():
      pixels = ct.read_pixels(path)
      if len(pixels) >= count:
        return (pixels[:count] - 0.1307) / 0.3081
  rng = np.random.default_rng(0)
  return rng.normal(size=(count, 784))


def memory_stats(devices):
  stats = {}
  for device in devices:
    try:
      raw = device.memory_stats() or {}
    except Exception:
      continue
    stats[str(device)] = {
        'bytes_in_use': int(raw.get('bytes_in_use', 0)),
        'peak_bytes_in_use': int(raw.get('peak_bytes_in_use', 0)),
        'bytes_limit': int(raw.get('bytes_limit', 0)),
    }
  return stats


def run_configuration(packed, keys, model, device_count, local_batch,
                      iterations, warmup, trace_dir=None, bsgs_jobs=1):
  import jax

  from jaxite_word import Mapping

  devices = tuple(jax.devices('tpu')[:device_count])
  if len(devices) != device_count:
    raise RuntimeError(f'need {device_count} TPU chips, found {len(devices)}')
  global_batch = device_count * local_batch

  build_started = time.perf_counter()
  mapping = Mapping(packed, keys=keys, global_batch=global_batch,
                    devices=devices, bsgs_n_jobs=bsgs_jobs)
  build_s = time.perf_counter() - build_started

  try:
    inputs = load_inputs(global_batch)
    encrypt_started = time.perf_counter()
    ciphertext = mapping.encrypt_input(
        list(inputs) if global_batch > 1 else inputs[0])
    encrypt_s = time.perf_counter() - encrypt_started

    # Correctness first: time only a computation that is demonstrably right.
    output = mapping.execute(ciphertext)
    jax.block_until_ready(output.polynomial)
    decrypted = mapping.decrypt_output(output)
    decrypted = np.asarray(
        decrypted if global_batch > 1 else [decrypted])
    expected = cleartext_reference(model, inputs)
    logit_error = float(np.abs(np.real(decrypted) - expected).max())
    label_match = int(
        (np.argmax(np.real(decrypted), axis=1) == np.argmax(expected, axis=1)).sum())

    for _ in range(warmup):
      jax.block_until_ready(mapping.execute(ciphertext).polynomial)

    samples = []
    for _ in range(iterations):
      started = time.perf_counter()
      result = mapping.execute(ciphertext)
      jax.block_until_ready(result.polynomial)
      samples.append(time.perf_counter() - started)

    trace_path = None
    if trace_dir:
      trace_path = str(trace_dir)
      Path(trace_path).mkdir(parents=True, exist_ok=True)
      with jax.profiler.trace(trace_path):
        for _ in range(3):
          jax.block_until_ready(mapping.execute(ciphertext).polynomial)

    median = statistics.median(samples)
    record = {
        'device_count': device_count,
        'local_batch': local_batch,
        'global_batch': global_batch,
        'mapping_build_s': build_s,
        'encrypt_s': encrypt_s,
        'execute_median_s': median,
        'execute_mean_s': statistics.fmean(samples),
        'execute_stdev_s': statistics.stdev(samples) if len(samples) > 1 else 0.0,
        'execute_min_s': min(samples),
        'execute_max_s': max(samples),
        'per_inference_ms': median / global_batch * 1000,
        'throughput_infer_per_s': global_batch / median,
        'max_logit_abs_error': logit_error,
        'labels_matching_cleartext': label_match,
        'labels_total': int(global_batch),
        'live_memory': mapping.estimate_live_memory(),
        'device_memory': memory_stats(devices),
        'rotation_keys': len(mapping.required_rotation_indices),
        'iterations': iterations,
        'trace_dir': trace_path,
    }
    return record
  finally:
    mapping.release()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--devices', default='1,2,4,8',
                      help='comma-separated TPU chip counts to sweep')
  parser.add_argument('--local-batch', default='1',
                      help='comma-separated per-device batch sizes')
  parser.add_argument('--iterations', type=int, default=20)
  parser.add_argument('--warmup', type=int, default=3)
  parser.add_argument('--trace-config', default='8:1',
                      help='"devices:local_batch" to capture a trace for, or "none"')
  parser.add_argument('--bsgs-jobs', type=int,
                      default=int(os.environ.get('CROSS_BSGS_JOBS', '1')),
                      help='parallel workers encoding BSGS diagonals (offline only)')
  parser.add_argument('--out', default=str(SUBMISSION_ROOT / 'profile' / 'results'))
  args = parser.parse_args()

  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  device_counts = [int(v) for v in args.devices.split(',') if v]
  local_batches = [int(v) for v in args.local_batch.split(',') if v]
  trace_config = None
  if args.trace_config and args.trace_config != 'none':
    a, b = args.trace_config.split(':')
    trace_config = (int(a), int(b))

  print('[profile] building the packed program')
  model, _, packed = ct.build_packed()
  ring = packed.ring_config
  print(f'[profile] ring degree={ring.degree} slots={ring.num_slots} '
        f'num_q={len(ring.q_towers)} num_p={len(ring.p_towers)} dnum={ring.dnum} '
        f'depth={packed.depth} security={ring.target.bits}-bit')

  keys_dir = out_dir / 'keys'
  if (keys_dir / 'pub' / 'pk.npy').exists():
    keys = ct.load_keys(keys_dir / 'pub', keys_dir / 'sec')
  else:
    print('[profile] generating keys for this ring')
    keys = ct.generate_keys(ring)
    ct.save_keys(keys, keys_dir / 'pub', keys_dir / 'sec')

  records = []
  for local_batch in local_batches:
    for device_count in device_counts:
      label = f'{device_count}chip_lb{local_batch}'
      print(f'\n[profile] === {device_count} chip(s), local batch {local_batch}, '
            f'global batch {device_count * local_batch} ===', flush=True)
      trace_dir = None
      if trace_config == (device_count, local_batch):
        trace_dir = out_dir / 'traces' / label
      try:
        record = run_configuration(
            packed, keys, model, device_count, local_batch,
            args.iterations, args.warmup, trace_dir, args.bsgs_jobs)
      except Exception as error:
        print(f'[profile] {label} failed: {type(error).__name__}: {error}')
        records.append({'device_count': device_count,
                        'local_batch': local_batch,
                        'error': f'{type(error).__name__}: {error}'})
        continue
      records.append(record)
      print(f"[profile] build {record['mapping_build_s']:.1f}s | "
            f"execute {record['execute_median_s'] * 1000:.1f} ms median | "
            f"{record['per_inference_ms']:.1f} ms/inference | "
            f"{record['throughput_infer_per_s']:.2f} inferences/s | "
            f"max logit error {record['max_logit_abs_error']:.2e} | "
            f"labels {record['labels_matching_cleartext']}/{record['labels_total']}",
            flush=True)
      (out_dir / 'profile_results.json').write_text(json.dumps(records, indent=2))

  (out_dir / 'profile_results.json').write_text(json.dumps(records, indent=2))

  print('\n[profile] summary')
  header = (f"{'chips':>5} {'lbatch':>7} {'gbatch':>7} {'execute ms':>11} "
            f"{'ms/infer':>9} {'infer/s':>9} {'speedup':>8}")
  print(header)
  print('-' * len(header))
  baseline = next((r['per_inference_ms'] for r in records
                   if r.get('device_count') == device_counts[0]
                   and r.get('local_batch') == local_batches[0]
                   and 'error' not in r), None)
  for record in records:
    if 'error' in record:
      print(f"{record['device_count']:>5} {record['local_batch']:>7} "
            f"{'':>7} {record['error']}")
      continue
    speedup = (baseline / record['per_inference_ms']) if baseline else float('nan')
    print(f"{record['device_count']:>5} {record['local_batch']:>7} "
          f"{record['global_batch']:>7} {record['execute_median_s'] * 1000:>11.1f} "
          f"{record['per_inference_ms']:>9.2f} "
          f"{record['throughput_infer_per_s']:>9.2f} {speedup:>7.2f}x")
  print(f"\n[profile] wrote {out_dir / 'profile_results.json'}")


if __name__ == '__main__':
  main()
