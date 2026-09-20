#!/usr/bin/env python3
"""CPU-only self-test for the CROSS submission (no TPU required).

Run via ``scripts/selftest.sh``. Verifies the accelerator-independent half of
the submission, so a reviewer without a TPU can still tell whether the
submission is intact.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
SUBMISSION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUBMISSION_ROOT / 'src'))
STAGES = ('client_key_generation', 'client_preprocess_input',
          'client_encode_encrypt_input', 'server_preprocess_model',
          'server_encrypted_compute', 'client_decrypt_decode',
          'client_postprocess')

import numpy as np  # noqa: E402

import cross_task as ct  # noqa: E402

CHECKS = []


def check(name):
  def register(function):
    CHECKS.append((name, function))
    return function
  return register


@check('CROSS library is importable')
def _cross_import():
  import importlib
  for module in ('jaxite_word.nn', 'jaxite_word.packing'):
    importlib.import_module(module)
  return f'CROSS_ROOT={ct.CROSS_ROOT or "(on PYTHONPATH)"}'


@check('model weights load and the BatchNorm fold is exact')
def _model():
  import torch

  from he_mlp import TrainHEMLP, to_inference_model

  model = ct.load_model()
  sample = torch.randn(4, 784, dtype=torch.float64)
  with torch.no_grad():
    out = model(sample)
  assert out.shape == (4, 10), f'model output shape {out.shape}'

  trained = TrainHEMLP().eval()
  folded = to_inference_model(trained)
  probe = torch.randn(8, 784)
  with torch.no_grad():
    deviation = (trained(probe) - folded(probe)).abs().max().item()
  assert deviation < 1e-4, f'BatchNorm fold deviation {deviation:.3e}'
  return f'fold deviation {deviation:.2e}'


@check('client and server derive the same ring')
def _ring_agreement():
  _, _, server_packed = ct.build_packed()
  client_packed = ct.build_client_packed()
  server, client = server_packed.ring_config, client_packed.ring_config
  assert server.degree == client.degree, 'degree differs'
  assert server.num_slots == client.num_slots, 'num_slots differs'
  assert server.q_towers == client.q_towers, 'q_towers differ'
  assert server_packed.depth == client_packed.depth, 'depth differs'
  assert server.target.bits >= 128, f'security {server.target.bits} < 128'
  return (f'degree={server.degree} slots={server.num_slots} '
          f'depth={server_packed.depth} security={server.target.bits}-bit')


@check('manifest mismatch is refused')
def _manifest_guard():
  packed = ct.build_client_packed()
  good = {'degree': packed.ring_config.degree,
          'num_slots': packed.ring_config.num_slots,
          'num_q': len(packed.ring_config.q_towers),
          'num_p': len(packed.ring_config.p_towers),
          'dnum': packed.ring_config.dnum, 'depth': packed.depth}
  ct.verify_against_manifest(packed, good)
  try:
    ct.verify_against_manifest(packed, {**good, 'degree': good['degree'] * 2})
  except RuntimeError:
    return 'mismatched ring rejected'
  raise AssertionError('a mismatched ring was accepted')


@check('encrypt -> file -> decrypt round-trips with scale intact')
def _codec_roundtrip():
  import mapping as mapping_mod
  from ckks_ctx import CKKSContext

  packed = ct.build_client_packed()
  keys = ct.generate_keys(packed.ring_config)
  ctx = CKKSContext(
      mapping_mod.ring_runtime_parameters(packed.ring_config, keys=keys))

  values = np.random.default_rng(0).normal(size=784)
  scale = float(packed.ring_config.scaling_factor)
  ciphertext = ctx.encrypt_slots(packed.pack(values), scale=scale)

  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'ct.npz'
    ct.save_ciphertext(path, ciphertext)
    payload, meta = ct.load_ciphertext_payload(
        path, expect_shape=np.asarray(ciphertext.to_array()).shape)
    assert meta['scale'] == scale, 'scale did not survive the file boundary'
    rebuilt = ct.build_polynomial(
        payload, meta['moduli'], meta['degree_layout'], meta['precision'],
        scale=meta['scale'], nsd=meta['nsd'])
    recovered = np.real(np.asarray(ctx.decrypt_slots(rebuilt, scale=scale)))
  error = float(np.abs(recovered[:784] - values).max())
  assert error < 1e-6, f'round-trip error {error:.3e}'
  return f'max error {error:.1e}'


@check('a corrupt ciphertext is rejected by name')
def _corrupt_ciphertext():
  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'broken.npz'
    path.write_bytes(b'not an npz file')
    try:
      ct.load_ciphertext_payload(path)
    except ValueError as error:
      assert 'broken.npz' in str(error), 'error does not name the file'
      return 'truncated file rejected'
  raise AssertionError('a corrupt ciphertext was accepted')


@check('wrong-shape ciphertext is rejected')
def _shape_guard():
  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'ct.npz'
    np.savez(path, payload=np.zeros((1, 2, 4, 4, 3), dtype=np.uint32),
             moduli=np.arange(3, dtype=np.int64),
             degree_layout=np.asarray([4, 4], dtype=np.int64),
             precision=np.int64(32))
    try:
      ct.load_ciphertext_payload(path, expect_shape=(1, 2, 128, 256, 12))
    except ValueError:
      return 'shape mismatch rejected'
  raise AssertionError('a mis-shaped ciphertext was accepted')


@check('atomic write leaves no partial file on failure')
def _atomic_write():
  with tempfile.TemporaryDirectory() as directory:
    target = Path(directory) / 'out.bin'

    def explode(handle):
      handle.write(b'partial')
      raise RuntimeError('simulated failure')

    try:
      ct.atomic_write_bytes(target, explode)
    except RuntimeError:
      pass
    assert not target.exists(), 'a failed write left the target behind'
    leftovers = [p.name for p in Path(directory).iterdir()]
    assert not leftovers, f'temporary files left behind: {leftovers}'
    ct.atomic_write_bytes(target, lambda handle: handle.write(b'ok'))
    assert target.read_bytes() == b'ok'
  return 'no partial file, no temp left behind'


@check('paths do not create directories when merely read')
def _no_side_effects():
  paths = ct.Paths(0)
  before = paths.rundir.exists()
  _ = (paths.socket_path, paths.daemon_pidfile, paths.ready_marker,
       paths.iodir, paths.ctxtupdir)
  assert paths.rundir.exists() == before, 'reading a path created it'
  return 'read-only path access confirmed'


@check('every harness stage exists and is runnable')
def _stages_present():
  """The harness runs ``<stage>.py`` if present, else ``build/<stage>``.

  A stage that is merely missing is silently skipped by
  ``utils.run_exe_or_python``, so the run would proceed and fail much later
  with confusing symptoms. Check all seven up front.
  """
  import ast

  missing, broken = [], []
  for stage in STAGES:
    script = SUBMISSION_ROOT / f'{stage}.py'
    if not script.exists():
      missing.append(stage)
      continue
    try:
      ast.parse(script.read_text(), filename=str(script))
    except SyntaxError as error:
      broken.append(f'{stage}: {error}')
  assert not missing, f'missing stage scripts: {missing}'
  assert not broken, f'stage scripts do not compile: {broken}'
  return f'{len(STAGES)} stages present and compiling'


@check('instance batch sizes match the harness')
def _batch_sizes():
  """The harness owns these; the submission only mirrors them."""
  harness_params = SUBMISSION_ROOT.parent.parent / 'harness' / 'params.py'
  text = harness_params.read_text()
  import re
  match = re.search(r'batch_size\s*=\s*\[([^\]]*)\]', text)
  assert match, 'could not find batch_size in harness/params.py'
  harness_sizes = tuple(int(v.strip()) for v in match.group(1).split(','))
  assert harness_sizes == ct.BATCH_SIZES, (
      f'submission mirrors {ct.BATCH_SIZES} but the harness declares '
      f'{harness_sizes}')
  return f'{harness_sizes}'


@check('processes that write under run/ create it first')
def _stage_dirs():
  """Guard the regression that removing the rundir mkdir side effect exposed.

  ``Paths.rundir`` is a pure property, so a process that *writes* under it must
  call ``ensure_rundir()`` first. Readers -- the stages that only connect to
  the socket or quote the log path in an error -- need nothing. This caught
  ``server_preprocess_model`` opening its daemon log in a directory that did
  not exist yet.
  """
  writers = {
      'server_preprocess_model.py': 'opens the daemon log',
      'src/server_daemon.py': 'writes the pidfile, socket and markers',
  }
  for relative, what in writers.items():
    body = (SUBMISSION_ROOT / relative).read_text()
    assert 'ensure_rundir' in body, (
        f'{relative} {what} under run/ but never calls ensure_rundir()')

  # And the directory really is absent until something asks for it.
  paths = ct.Paths(0)
  assert 'mkdir' not in type(paths).rundir.fget.__code__.co_names, \
      'rundir property regained a mkdir side effect'
  return f'{len(writers)} writers create run/ before use'


def main():
  print('CROSS submission self-test (CPU only)\n')
  failures = 0
  for name, function in CHECKS:
    try:
      detail = function()
    except Exception as error:
      failures += 1
      print(f'  FAIL  {name}')
      print(f'        {type(error).__name__}: {error}')
      if os.environ.get('SELFTEST_TRACEBACK'):
        traceback.print_exc()
    else:
      print(f'  ok    {name}' + (f'  ({detail})' if detail else ''))
  print()
  if failures:
    print(f'{failures} of {len(CHECKS)} checks FAILED')
    return 1
  print(f'all {len(CHECKS)} checks passed')
  return 0


if __name__ == '__main__':
  sys.exit(main())
