#!/usr/bin/env python3
"""Consolidate the CROSS/TPU benchmark results into one report.

Pulls together three sources that are measured independently:

  * ``profile/results/profile_results.json``  -- the TPU device/batch sweep
  * ``io/<size>/intermediate/cross_timings.jsonl`` -- per-stage timings from a
    real harness run of this submission
  * ``measurements/<size>/results-*.json`` -- the harness's own stage timings,
    for whichever submission produced them (CROSS or the OpenFHE reference)
"""
from __future__ import annotations

import argparse
import json

from pathlib import Path

SUBMISSION_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SUBMISSION_ROOT.parent
INSTANCE_NAMES = ('single', 'small', 'medium', 'large')


def load_json(path: Path, default=None):
  try:
    return json.loads(path.read_text())
  except Exception:
    return default


def load_jsonl(path: Path):
  records = []
  try:
    for line in path.read_text().splitlines():
      line = line.strip()
      if line:
        records.append(json.loads(line))
  except Exception:
    pass
  return records


def section(title):
  print()
  print(title)
  print('=' * len(title))


def tpu_sweep(results_path: Path):
  records = load_json(results_path, [])
  if not records:
    print(f'(no TPU sweep results at {results_path})')
    return
  section('TPU v6e sweep — steady-state Mapping.execute')
  header = (f"{'chips':>5} {'local':>6} {'global':>7} {'execute ms':>11} "
            f"{'ms/infer':>9} {'infer/s':>9} {'scaling':>8} {'max err':>10} {'labels':>8}")
  print(header)
  print('-' * len(header))
  ok = [r for r in records if 'error' not in r]
  base = next((r for r in ok if r['device_count'] == 1 and r['local_batch'] == 1), None)
  base_tp = base['throughput_infer_per_s'] if base else None
  for record in records:
    if 'error' in record:
      print(f"{record['device_count']:>5} {record['local_batch']:>6} "
            f"{'--':>7} {record['error']}")
      continue
    scaling = (record['throughput_infer_per_s'] / base_tp) if base_tp else float('nan')
    print(f"{record['device_count']:>5} {record['local_batch']:>6} "
          f"{record['global_batch']:>7} {record['execute_median_s'] * 1000:>11.1f} "
          f"{record['per_inference_ms']:>9.2f} {record['throughput_infer_per_s']:>9.2f} "
          f"{scaling:>7.2f}x {record['max_logit_abs_error']:>10.1e} "
          f"{record['labels_matching_cleartext']}/{record['labels_total']:<6}")
  if ok:
    memory = ok[0]['live_memory']
    print()
    print(f"rotation keys       : {ok[0]['rotation_keys']}")
    print(f"evaluation key      : {memory['evaluation_key_bytes'] / 2**20:.1f} MiB")
    print(f"rotation keys total : {memory['rotation_keys_bytes'] / 2**30:.2f} GiB")
    print(f"matvec constants    : {memory['matvec_constants_bytes'] / 2**30:.2f} GiB")
    print(f"live memory (logical): {memory['total_bytes'] / 2**30:.2f} GiB")
    builds = [r['mapping_build_s'] for r in ok]
    print(f"Mapping build       : {min(builds):.0f}-{max(builds):.0f} s per configuration")


def stage_timings(size: int):
  name = INSTANCE_NAMES[size]
  records = load_jsonl(REPO_ROOT / 'io' / name / 'intermediate' / 'cross_timings.jsonl')
  if not records:
    return
  section(f'CROSS submission stage timings — {name} instance')
  for record in records:
    stage = record['stage']
    if stage == 'client_key_generation':
      print(f"  2 client_key_generation   : pack {record['pack_s']:.2f}s  "
            f"keygen {record['keygen_s']:.2f}s  save {record['save_s']:.2f}s")
    elif stage == 'server_preprocess_model':
      print(f"  3 server_preprocess_model : {record['startup_wall_s']:.1f}s total  "
            f"(mapping {record['mapping_build_s']:.1f}s, warmup {record['warmup_s']:.1f}s)")
    elif stage == 'client_preprocess_input':
      print(f"  5 client_preprocess_input : {record['elapsed_s']:.3f}s")
    elif stage == 'client_encode_encrypt_input':
      print(f"  6 client_encode_encrypt   : {record['encrypt_s']:.2f}s for "
            f"{record['samples']} sample(s) = {record['per_sample_s'] * 1000:.0f} ms each "
            f"(+{record['context_setup_s']:.1f}s context setup)")
    elif stage == 'server_encrypted_compute':
      print(f"  7 server_encrypted_compute: {record['execute_total_s']:.3f}s TPU execute for "
            f"{record['count']} inference(s) on {record['device_count']} chip(s) = "
            f"{record['per_inference_s'] * 1000:.0f} ms each, "
            f"{record['throughput_infer_per_s']:.2f}/s")
    elif stage == 'client_decrypt_decode':
      print(f"  8 client_decrypt_decode   : {record['decrypt_s']:.2f}s for "
            f"{record['samples']} sample(s) = {record['per_sample_s'] * 1000:.0f} ms each")
    elif stage == 'client_postprocess':
      print(f"  9 client_postprocess      : {record['elapsed_s']:.3f}s")


def harness_measurements(size: int, label: str):
  name = INSTANCE_NAMES[size]
  directory = REPO_ROOT / 'measurements' / name
  files = sorted(directory.glob('results-*.json')) if directory.exists() else []
  if not files:
    return
  section(f'Harness measurements — {name} ({label})')
  for path in files:
    data = load_json(path, {})
    print(f'  {path.name}: ' + json.dumps(
        {k: v for k, v in data.items() if k not in ('sizes',)},
        default=str)[:400])


def markdown_table(results_path: Path) -> str:
  """The sweep as a markdown table, for pasting into a write-up."""
  records = [r for r in load_json(results_path, []) if 'error' not in r]
  if not records:
    return '(no TPU sweep results)\n'
  base = next((r for r in records
               if r['device_count'] == 1 and r['local_batch'] == 1), None)
  base_tp = base['throughput_infer_per_s'] if base else None
  lines = [
      '| chips | ct/chip | batch | execute (ms) | ms/inference | inferences/s | scaling |',
      '|---:|---:|---:|---:|---:|---:|---:|',
  ]
  for record in records:
    scaling = (f"{record['throughput_infer_per_s'] / base_tp:.2f}x"
               if base_tp else '--')
    lines.append(
        f"| {record['device_count']} | {record['local_batch']} | "
        f"{record['global_batch']} | {record['execute_median_s'] * 1000:.1f} | "
        f"{record['per_inference_ms']:.2f} | "
        f"{record['throughput_infer_per_s']:.2f} | {scaling} |")
  return '\n'.join(lines) + '\n'


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--results', default=str(SUBMISSION_ROOT / 'profile' / 'results' / 'profile_results.json'))
  parser.add_argument('--sizes', default='0,1')
  parser.add_argument('--label', default='CROSS/TPU')
  parser.add_argument('--markdown', action='store_true',
                      help='emit the sweep as a markdown table and exit')
  args = parser.parse_args()

  if args.markdown:
    print(markdown_table(Path(args.results)), end='')
    return

  print('CROSS on TPU v6e — FHE-Benchmarks-ML-Inference (MNIST)')
  tpu_sweep(Path(args.results))
  for size in [int(v) for v in args.sizes.split(',') if v]:
    stage_timings(size)
    harness_measurements(size, args.label)


if __name__ == '__main__':
  main()
