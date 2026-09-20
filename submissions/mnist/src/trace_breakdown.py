#!/usr/bin/env python3
"""Turn a ``jax.profiler`` trace of a CROSS Mapping into an op breakdown.

CROSS's own ``Profiler`` (``jaxite_word/profiler.py``) filters a TPU trace down
to one device lane, merges events by name, and writes ``filtered_events.json``
for ``profile_analysis/analyze_trace_json.py`` to categorize into INTT / NTT /
BConv / VecModOps / Memory Reorder / Permutation / Type Conversion.

That path is built for a ``KernelWrapper`` around a raw-array kernel, and it
hard-codes ``pid == 3`` ("TPU:0").  A Mapping compiled for eight chips writes
eight device lanes, so this module resolves the lanes from the trace's own
process-name metadata instead, and can either aggregate them or report one
chip.  The output file is the same shape CROSS's analyzer already consumes.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

CROSS_ROOT = os.environ.get('CROSS_ROOT', '/home/jianming_gatech/CROSS')
ANALYZER = Path(CROSS_ROOT) / 'profile_analysis' / 'analyze_trace_json.py'

DEVICE_LANE = re.compile(r'/device:TPU:(\d+)')


def find_trace(trace_dir: Path) -> Path:
  candidates = [Path(root) / name
                for root, _, files in os.walk(trace_dir)
                for name in files if name.endswith('.trace.json.gz')]
  if not candidates:
    raise FileNotFoundError(f'no .trace.json.gz under {trace_dir}')
  return max(candidates, key=lambda path: path.stat().st_mtime)


def read_trace(path: Path) -> dict:
  with gzip.open(path, 'rt', encoding='utf-8') as handle:
    return json.load(handle)


def device_pids(events) -> dict[int, int]:
  """Map pid -> TPU device index, from the trace's process_name metadata."""
  found = {}
  for event in events:
    if event.get('ph') != 'M' or event.get('name') != 'process_name':
      continue
    label = str(event.get('args', {}).get('name', ''))
    match = DEVICE_LANE.search(label)
    if match:
      found[int(event['pid'])] = int(match.group(1))
  return found


def xla_op_tids(events, pids) -> set[tuple[int, int]]:
  """The (pid, tid) lanes carrying XLA ops, not the step/framework lanes."""
  lanes = set()
  for event in events:
    if event.get('ph') != 'M' or event.get('name') != 'thread_name':
      continue
    pid = int(event.get('pid', -1))
    if pid not in pids:
      continue
    label = str(event.get('args', {}).get('name', ''))
    if 'XLA Ops' in label:
      lanes.add((pid, int(event['tid'])))
  return lanes


def build_filtered(trace: dict, device: int | None):
  events = trace.get('traceEvents', [])
  pids = device_pids(events)
  if not pids:
    raise RuntimeError('no TPU device lanes found in trace')
  lanes = xla_op_tids(events, pids)
  if not lanes:
    raise RuntimeError('no "XLA Ops" lanes found in trace')

  selected_pids = {pid for pid, index in pids.items()
                   if device is None or index == device}
  merged = defaultdict(lambda: {'dur': [0.0], 'count': 0, 'args': {}})
  for event in events:
    if event.get('ph') == 'M' or 'dur' not in event:
      continue
    pid, tid = int(event.get('pid', -1)), int(event.get('tid', -1))
    if pid not in selected_pids or (pid, tid) not in lanes:
      continue
    name = event.get('name', '')
    entry = merged[name]
    entry['dur'][0] += float(event['dur'])
    entry['count'] += 1
    if not entry['args'] and 'args' in event:
      entry['args'] = event['args']
  return {name: dict(entry) for name, entry in merged.items()}, pids


def category_report(filtered, iterations, lanes):
  """Group device time by XLA's own ``hlo_category``.

  Preferred over name-keyword matching for a fused TPU program: CROSS's
  analyzer classifies by op name, which works when each HLO is one recognizable
  kernel, but XLA fuses this Mapping into ``loop fusion`` / ``custom fusion``
  nodes whose names carry no arithmetic meaning -- everything lands in "Other".
  ``hlo_category`` is the compiler's own label for the same node.

  ``while`` nodes are the BSGS giant-step loops. Their duration *contains*
  their children's, so counting them with the leaves would double-count; they
  are reported separately instead.
  """
  leaves, counts, containers = defaultdict(float), defaultdict(int), defaultdict(float)
  for name, entry in filtered.items():
    category = (entry.get('args') or {}).get('hlo_category') or 'unknown'
    duration = entry['dur'][0]
    if name.startswith('while.'):
      containers[category] += duration
      continue
    leaves[category] += duration
    counts[category] += entry['count']

  total = sum(leaves.values())
  print()
  print('device time by XLA hlo_category (leaf nodes, while-loops excluded)')
  print(f'{"ms":>10} {"share":>7} {"count":>8}  category')
  print('-' * 46)
  for category, value in sorted(leaves.items(), key=lambda kv: -kv[1]):
    print(f'{value / 1000:10.2f} {100 * value / total:6.1f}% {counts[category]:8d}  {category}')
  print('-' * 46)
  print(f'{total / 1000:10.2f} {"100.0%":>7}           total')
  if containers:
    print()
    for category, value in sorted(containers.items(), key=lambda kv: -kv[1]):
      print(f'{value / 1000:10.2f} ms  {category} (container: includes the leaves above)')
  if iterations and lanes:
    per = total / (iterations * lanes)
    print()
    print(f'normalized: {total / 1000:.1f} ms over {lanes} lane(s) x {iterations} '
          f'iteration(s) = {per / 1000:.1f} ms per chip per execute')
  return total


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('trace_dir')
  parser.add_argument('--device', type=int, default=None,
                      help='report one TPU chip; default aggregates all lanes')
  parser.add_argument('--out', default=None)
  parser.add_argument('--iterations', type=int, default=3,
                      help='executes captured in the trace, for normalization')
  parser.add_argument('--keyword-report', action='store_true',
                      help="also run CROSS's name-keyword analyzer")
  args = parser.parse_args()

  trace_dir = Path(args.trace_dir)
  trace_file = find_trace(trace_dir)
  trace = read_trace(trace_file)
  filtered, pids = build_filtered(trace, args.device)

  out = Path(args.out) if args.out else trace_dir / 'filtered_events.json'
  out.write_text(json.dumps(filtered, indent=1))
  total = sum(entry['dur'][0] for entry in filtered.values())
  scope = f'TPU:{args.device}' if args.device is not None else f'{len(pids)} chip(s)'
  print(f'trace      : {trace_file}')
  print(f'device lanes: {sorted(pids.values())}  (reporting {scope})')
  print(f'events     : {len(filtered)} distinct op names, {total / 1000:.2f} ms total')
  print(f'wrote      : {out}')

  lanes = 1 if args.device is not None else len(pids)
  category_report(filtered, args.iterations, lanes)

  if args.keyword_report:
    if ANALYZER.exists():
      print()
      print("CROSS name-keyword analyzer (profile_analysis/analyze_trace_json.py)")
      subprocess.run([sys.executable, str(ANALYZER), str(out)], check=False)
    else:
      print(f'(CROSS analyzer not found at {ANALYZER})', file=sys.stderr)


if __name__ == '__main__':
  main()
