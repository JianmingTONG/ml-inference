"""Shared plumbing for the CROSS/TPU submission to the ML-inference benchmark.

The benchmark runs each stage as a separate process that communicates through
files.  CROSS is a JAX library and a TPU can be held by exactly one process at
a time, so the split is:

  * client stages  -- ``JAX_PLATFORMS=cpu``.  They build a bare ``CKKSContext``
    from the ring the packed program chose and the key pair on disk, and do
    nothing but encode/encrypt and decrypt/decode.  That is what a client
    actually is: it never sees the model's schedule and never touches the
    accelerator.
  * server stages  -- the TPU.  ``server_preprocess_model`` materializes one
    ``Mapping`` (BSGS diagonals, evaluation and rotation keys, constant
    encoding, XLA compilation) inside a daemon that keeps it resident;
    ``server_encrypted_compute`` hands that daemon a ciphertext file.

The daemon exists because a compiled JAX executable cannot cross a process
boundary.  Rebuilding the Mapping per request would report compilation time as
inference time, which is exactly the number the benchmark is trying to measure.
A real FHE server loads its keys once too -- the reference OpenFHE submission
prints "[server] Loading keys" on every invocation for the same reason.

Both sides derive the ring from the *same* packed program.  Nothing here picks
a security parameter: ``packing.pack`` does, from the program's slot demand and
emitted depth.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# submissions/mnist/src/cross_task.py -> submissions/mnist -> repository root
SUBMISSION_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SUBMISSION_ROOT.parent.parent


def _locate_cross() -> str:
  """Find the CROSS checkout that provides ``jaxite_word``.

  CROSS is not packaged (the repository is used in place), so it has to be put
  on ``sys.path`` by hand. Search, in order: an explicit ``CROSS_ROOT``; an
  already-importable ``jaxite_word``; then the usual sibling locations. Raising
  a named error here is far kinder than the ``ModuleNotFoundError:
  jaxite_word`` a caller would otherwise meet three imports deep.
  """
  explicit = os.environ.get('CROSS_ROOT')
  candidates = []
  if explicit:
    candidates.append(Path(explicit))
  candidates += [
      REPO_ROOT.parent / 'CROSS',
      REPO_ROOT / 'third_party' / 'cross',
      REPO_ROOT / 'third_party' / 'CROSS',
      Path.home() / 'CROSS',
  ]
  for candidate in candidates:
    if (candidate / 'jaxite_word' / '__init__.py').is_file():
      return str(candidate)
  import importlib.util
  if importlib.util.find_spec('jaxite_word') is not None:
    return ''  # already installed, or already on PYTHONPATH
  searched = ', '.join(str(c) for c in candidates)
  raise RuntimeError(
      'cannot find the CROSS library (jaxite_word). Set CROSS_ROOT to your '
      f'CROSS checkout. Searched: {searched}')


CROSS_ROOT = _locate_cross()

for _path in (CROSS_ROOT, str(SUBMISSION_ROOT / 'model'), str(REPO_ROOT / 'harness')):
  if _path and _path not in sys.path:
    sys.path.insert(0, _path)

INSTANCE_NAMES = ('single', 'small', 'medium', 'large')
# Must match harness/params.py; the harness owns this and we only mirror it.
BATCH_SIZES = (1, 100, 1000, 10000)

# How many TPU chips the server Mapping is compiled for, and therefore how many
# independent ciphertexts one `execute` consumes.  The public global batch must
# be divisible by the device count; Mapping derives local batch = B // D.
DEFAULT_DEVICE_COUNT = int(os.environ.get('CROSS_DEVICE_COUNT', '8'))

SOCKET_NAME = 'cross_server.sock'


# --------------------------------------------------------------------------
# Benchmark directory layout (mirrors harness/params.py and submission/params.h)
# --------------------------------------------------------------------------
class Paths:
  """The benchmark's directory contract, for one instance size."""

  def __init__(self, size: int, rootdir=None):
    if not 0 <= size < len(INSTANCE_NAMES):
      raise ValueError(f'invalid instance size {size}')
    self.size = size
    self.name = INSTANCE_NAMES[size]
    self.batch_size = BATCH_SIZES[size]
    self.rootdir = Path(rootdir) if rootdir else REPO_ROOT

  @property
  def iodir(self): return self.rootdir / 'io' / self.name

  @property
  def datadir(self): return self.rootdir / 'datasets' / self.name

  @property
  def pubkeydir(self): return self.iodir / 'public_keys'

  @property
  def seckeydir(self): return self.iodir / 'secret_key'

  @property
  def ctxtupdir(self): return self.iodir / 'ciphertexts_upload'

  @property
  def ctxtdowndir(self): return self.iodir / 'ciphertexts_download'

  @property
  def iointermdir(self): return self.iodir / 'intermediate'

  @property
  def dataintermdir(self): return self.datadir / 'intermediate'

  @property
  def test_input_file(self): return self.dataintermdir / 'test_pixels.txt'

  @property
  def ground_truth_file(self): return self.dataintermdir / 'test_labels.txt'

  @property
  def predictions_file(self): return self.iodir / 'encrypted_model_predictions.txt'

  @property
  def manifest_file(self): return self.pubkeydir / 'ring_manifest.json'

  @property
  def rundir(self):
    """Daemon control files.

    NOT under ``iodir``: ``run_submission.py`` deletes ``io/<size>`` at the
    start of every invocation, which would take the previous run's pidfile and
    socket with it -- leaving that daemon resident, still holding the TPU, and
    unfindable by the next one. A TPU admits exactly one process, so the second
    run would then fail to initialize the backend.

    Reading a path does not create it; call ``ensure_rundir()`` to do that.
    A property that mkdirs surprises every caller that only wanted to look.
    """
    return SUBMISSION_ROOT / 'run' / self.name

  def ensure_rundir(self) -> Path:
    """Create and return the daemon control directory."""
    directory = self.rundir
    directory.mkdir(parents=True, exist_ok=True)
    return directory

  @property
  def socket_path(self): return self.rundir / SOCKET_NAME

  @property
  def daemon_pidfile(self): return self.rundir / 'cross_server.pid'

  @property
  def daemon_log(self): return self.rundir / 'cross_server.log'

  @property
  def ready_marker(self): return self.rundir / 'cross_server.ready'

  @property
  def failed_marker(self): return self.rundir / 'cross_server.failed'


def atomic_write_bytes(path: Path, write) -> None:
  """Write via a temporary file in the same directory, then rename.

  Every file here is produced by one benchmark stage and consumed by the next
  one, as a separate process. A stage killed midway through a plain write
  leaves a truncated file that the next stage happily opens, and the failure
  then surfaces somewhere far away as a corrupt ciphertext. ``os.replace`` is
  atomic within a filesystem, so a reader sees either the old file or the
  complete new one.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  handle = tempfile.NamedTemporaryFile(
      dir=str(path.parent), prefix=f'.{path.name}.', suffix='.tmp',
      delete=False)
  temporary = Path(handle.name)
  try:
    with handle:
      write(handle)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, path)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise


def atomic_write_text(path: Path, text: str) -> None:
  atomic_write_bytes(path, lambda handle: handle.write(text.encode()))


ACTIVE_SIZE_FILE = SUBMISSION_ROOT / 'run' / 'active_size'


def record_active_size(size: int) -> None:
  """Remember which instance size this run is for.

  The harness invokes ``server_preprocess_model`` with no arguments
  (``utils.run_exe_or_python(model_exec_dir, "server_preprocess_model")``),
  but the server has to know the instance size to find its key material and to
  size its batch. ``client_key_generation`` runs immediately before it and does
  get the size, so it records it here. Kept outside ``io/`` because the harness
  deletes ``io/<size>`` at the start of every invocation.
  """
  ACTIVE_SIZE_FILE.parent.mkdir(parents=True, exist_ok=True)
  atomic_write_text(ACTIVE_SIZE_FILE, f'{int(size)}\n')


def read_active_size() -> int:
  if not ACTIVE_SIZE_FILE.exists():
    raise SystemExit(
        f'{ACTIVE_SIZE_FILE} not found: client_key_generation records the '
        'instance size there and must run first (benchmark stage 2.2).')
  return int(ACTIVE_SIZE_FILE.read_text().strip())


def report_server_steps(paths: 'Paths', steps: dict) -> None:
  """Publish fine-grained server metrics the harness folds into results-N.json.

  The harness reads ``io/<size>/server_reported_steps.json`` if a submission
  writes one, which is how the pure accelerator time is reported separately
  from the stage wall clock that also covers ciphertext file I/O.
  """
  path = paths.iodir / 'server_reported_steps.json'
  existing = {}
  if path.exists():
    try:
      existing = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
      existing = {}
  existing.update({key: round(float(value), 4) for key, value in steps.items()})
  atomic_write_text(path, json.dumps(existing, indent=2))


def parse_size(argv) -> int:
  if len(argv) < 2 or not argv[1].isdigit():
    print(f'Usage: {argv[0]} instance-size  (0-single 1-small 2-medium 3-large)')
    raise SystemExit(0)
  return int(argv[1])


# --------------------------------------------------------------------------
# The model, and the packed program both parties derive from it
# --------------------------------------------------------------------------
# Which HE-friendly architecture to deploy. CROSS derives the ring from the
# multiplicative depth the program emits, so the architecture *is* the
# parameter choice:
#
#   shallow  784-50-10,     one square,  depth 3 -> N = 16384,  8 Q towers
#   deep     784-128-64-10, two squares, depth 5 -> N = 32768, 12 Q towers
#
# `shallow` is the default: 4.4x faster per inference, and still ahead of both
# other leaderboard entries on accuracy.
#
# `shallow` mirrors the topology Lattica-ai's submission uses and is the
# smallest ring CROSS can reach for a useful MNIST model. See
# model/he_mlp_shallow.py for why N = 4096 is not reachable at any depth.
MODEL_VARIANTS = {
    'deep': ('he_mlp', 'HEMLP', 'he_mlp_weights.pth'),
    'shallow': ('he_mlp_shallow', 'ShallowHEMLP', 'he_mlp_shallow_weights.pth'),
}
MODEL_VARIANT = os.environ.get('CROSS_MODEL', 'shallow').lower()
if MODEL_VARIANT not in MODEL_VARIANTS:
  raise SystemExit(
      f'CROSS_MODEL={MODEL_VARIANT!r} is not one of {sorted(MODEL_VARIANTS)}')

_MODULE, _CLASS, _WEIGHT_FILE = MODEL_VARIANTS[MODEL_VARIANT]
WEIGHTS = SUBMISSION_ROOT / 'model' / _WEIGHT_FILE


def _model_class():
  import importlib
  return getattr(importlib.import_module(_MODULE), _CLASS)


def load_model():
  """The trained inference-shaped HE MLP, in float64 for exact packing."""
  import torch

  model = _model_class()()
  if not WEIGHTS.exists():
    raise FileNotFoundError(
        f'{WEIGHTS} not found; train it with '
        f'model/train_{_MODULE}.py (CROSS_MODEL={MODEL_VARIANT}).')
  model.load_state_dict(torch.load(WEIGHTS, map_location='cpu'))
  return model.double().eval()


def build_packed(model=None):
  """Vectorize and pack the model.  Deterministic: same bytes, same ring.

  ``lazy_constants`` keeps the matrix-free source instead of materializing
  every MatVec diagonal here.  At 16384 slots the eager form is gigabytes, and
  Mapping encodes the diagonals it actually needs anyway.
  """
  from jaxite_word import nn, packing

  if model is None:
    model = load_model()
  program = nn.vectorize(model, (784,), warn=False)
  packed = packing.pack(
      program, policy=packing.PackingPolicy(lazy_constants=True))
  return model, program, packed


def build_client_packed():
  """The packed program a *client* can derive, without the server's weights.

  The ring, the slot layout and the input/output coordinate maps follow from
  the program's shapes and its emitted depth -- not from any weight value.  So
  a client packs the same architecture with untrained parameters and gets the
  identical ``RingConfig`` and identical pack/unpack maps, while the MatVec
  constants (the only weight-dependent artifact) stay on the server where they
  belong.  ``verify_against_manifest`` then checks the two sides really did
  land on the same ring before anything is encrypted.
  """
  from jaxite_word import nn, packing

  model = _model_class()().double().eval()
  program = nn.vectorize(model, (784,), warn=False)
  return packing.pack(
      program, policy=packing.PackingPolicy(lazy_constants=True))


def verify_against_manifest(packed, manifest):
  """Refuse to encrypt unless the client's ring matches the server's."""
  ring = packed.ring_config
  mismatches = []
  for name, mine, theirs in (
      ('degree', int(ring.degree), int(manifest['degree'])),
      ('num_slots', int(ring.num_slots), int(manifest['num_slots'])),
      ('num_q', len(ring.q_towers), int(manifest['num_q'])),
      ('num_p', len(ring.p_towers), int(manifest['num_p'])),
      ('dnum', int(ring.dnum), int(manifest['dnum'])),
      ('depth', int(packed.depth), int(manifest['depth'])),
  ):
    if mine != theirs:
      mismatches.append(f'{name}: client {mine} != server {theirs}')
  if mismatches:
    raise RuntimeError(
        'client and server disagree on the deployment ring: '
        + '; '.join(mismatches))


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------
def generate_keys(ring_config):
  """Client-side CKKS key pair, at the ring's own error width."""
  import key_gen as kg

  runtime = ring_config.runtime_kwargs
  kwargs = dict(runtime() if callable(runtime) else runtime)
  return kg.gen_pke_pair(
      [int(v) for v in kwargs['q_towers']],
      [int(v) for v in kwargs['p_towers']],
      int(kwargs['degree']),
      noise_std=float(kwargs['sigma']),
  )


def _to_arrays(key):
  """Normalize a key polynomial (nested lists of ints) to int64 arrays."""
  return np.asarray(key, dtype=np.int64)


def save_keys(keys, pubkeydir: Path, seckeydir: Path):
  pubkeydir.mkdir(parents=True, exist_ok=True)
  seckeydir.mkdir(parents=True, exist_ok=True)
  np.save(pubkeydir / 'pk.npy', _to_arrays(keys['public_key']))
  np.save(seckeydir / 'sk.npy', _to_arrays(keys['secret_key']))


def load_keys(pubkeydir: Path, seckeydir: Path) -> dict:
  """Reload the key pair in the nested-list form ``key_gen`` produced."""
  return {
      'public_key': _restore(np.load(pubkeydir / 'pk.npy')),
      'secret_key': _restore(np.load(seckeydir / 'sk.npy')),
  }


def _restore(array: np.ndarray):
  """Rebuild the nested-list shape key_gen produced from a saved array."""
  if array.ndim == 2:
    return [[int(v) for v in row] for row in array]
  if array.ndim == 3:
    return [[[int(v) for v in row] for row in block] for block in array]
  raise ValueError(f'unexpected key array rank {array.ndim}')


# --------------------------------------------------------------------------
# Ring manifest: what the client must know to encrypt for this deployment
# --------------------------------------------------------------------------
def write_manifest(path: Path, packed, input_scale, output_scale,
                   global_batch, device_count, extra=None):
  ring = packed.ring_config
  manifest = {
      'model_fingerprint': packed.fingerprint,
      'degree': int(ring.degree),
      'num_slots': int(ring.num_slots),
      'num_q': len(ring.q_towers),
      'num_p': len(ring.p_towers),
      'dnum': int(ring.dnum),
      'composite_degree': int(ring.composite_degree),
      'security_bits': int(ring.target.bits),
      'security_model': str(ring.target.cost_model.value),
      'depth': int(packed.depth),
      'operations': [op[1] for op in packed.operations],
      'input_scale': float(input_scale),
      'output_scale': float(output_scale),
      'global_batch': int(global_batch),
      'device_count': int(device_count),
  }
  if extra:
    manifest.update(extra)
  atomic_write_text(path, json.dumps(manifest, indent=2))
  return manifest


def read_manifest(path: Path) -> dict:
  """Load the deployment manifest stage 3 published."""
  if not path.exists():
    raise FileNotFoundError(
        f'{path} not found: the server has not published a deployment '
        f'manifest. Run server_preprocess_model (benchmark stage 3) first.')
  try:
    return json.loads(path.read_text())
  except json.JSONDecodeError as error:
    raise ValueError(f'{path} is not valid JSON: {error}') from error


# --------------------------------------------------------------------------
# Ciphertext serialization
# --------------------------------------------------------------------------
# CKKS tracked scale and noise-scale degree ride on the Polynomial as private
# attributes. CROSS has no public ciphertext serialization, and the benchmark's
# stages communicate through files, so this one helper pair owns reading and
# writing them. A ciphertext that crosses the file boundary without its scale
# is rejected by Mapping with "input Polynomial scale None does not match
# planned scale".
_SCALE_ATTR = '_ckks_scale'
_NSD_ATTR = '_ckks_nsd'


def save_ciphertext(path: Path, poly):
  """Persist a canonical rank-5 Polynomial payload plus its metadata."""
  scale = getattr(poly, _SCALE_ATTR, None)
  nsd = getattr(poly, _NSD_ATTR, None)
  fields = {
      'payload': np.asarray(poly.to_array()),
      'moduli': np.asarray(poly.get_moduli(), dtype=np.int64),
      'degree_layout': np.asarray(poly.degree_layout, dtype=np.int64),
      'precision': np.int64(poly.precision),
      'scale': np.float64(scale if scale is not None else np.nan),
      'nsd': np.int64(nsd if nsd is not None else -1),
  }
  atomic_write_bytes(path, lambda handle: np.savez(handle, **fields))


def load_ciphertext_payload(path: Path, expect_shape=None):
  """Return (payload, metadata dict) for a persisted ciphertext.

  ``expect_shape`` is the payload shape this deployment was compiled for, with
  the batch axis ignored. Checking it here turns a mismatched or truncated file
  into one clear error naming the file, instead of a shape error raised deep
  inside the evaluator with no idea which of 10000 inputs was at fault.
  """
  if not path.exists():
    raise FileNotFoundError(f'ciphertext not found: {path}')
  # Any failure to read the container at all -- truncated, not an npz, wrong
  # magic -- is reported against the file by name. numpy raises a bare
  # ValueError here, which on its own tells the caller nothing about which of
  # 10000 ciphertexts was at fault.
  try:
    with np.load(path) as data:
      fields = {key: data[key] for key in data.files}
  except Exception as error:
    raise ValueError(
        f'{path} is not a readable ciphertext ({type(error).__name__}: '
        f'{error}); it may be truncated from an interrupted run') from error

  missing = [key for key in ('payload', 'moduli', 'degree_layout', 'precision')
             if key not in fields]
  if missing:
    raise ValueError(f'{path} is missing {missing}')
  payload = np.asarray(fields['payload'])
  scale = float(fields['scale']) if 'scale' in fields else float('nan')
  nsd = int(fields['nsd']) if 'nsd' in fields else -1
  metadata = {
      'moduli': [int(v) for v in fields['moduli']],
      'degree_layout': tuple(int(v) for v in fields['degree_layout']),
      'precision': int(fields['precision']),
      'scale': None if np.isnan(scale) else scale,
      'nsd': None if nsd < 0 else nsd,
  }

  if payload.ndim != 5:
    raise ValueError(
        f'{path}: ciphertext payload must be rank 5 '
        f'(batch, elements, r, c, moduli); got shape {payload.shape}')
  if expect_shape is not None:
    expected, actual = tuple(expect_shape)[1:], tuple(payload.shape)[1:]
    if expected != actual:
      raise ValueError(
          f'{path}: payload shape {actual} does not match the shape this '
          f'deployment was compiled for, {expected}. The ciphertext was '
          f'produced for a different ring or model.')
  return payload, metadata


def build_polynomial(payload, moduli, degree_layout, precision, ctx=None,
                     scale=None, nsd=None):
  """Wrap a persisted payload back into a canonical Polynomial.

  ``Polynomial.from_array`` is the public constructor for an already-canonical
  payload.  Reusing ``ctx``'s NTT context when one is available avoids
  rebuilding the transform tables for every request.
  """
  import jax.numpy as jnp
  from polynomial import Polynomial

  batch, num_elements, rows, columns, num_moduli = payload.shape
  shapes = {
      'batch': batch,
      'num_elements': num_elements,
      'degree': rows * columns,
      'num_moduli': num_moduli,
      'precision': precision,
      'degree_layout': (rows, columns),
  }
  parameters = {'moduli': list(moduli)}
  dtype = jnp.uint32 if precision <= 32 else jnp.uint64
  value = Polynomial.from_array(
      jnp.asarray(payload, dtype=dtype), shapes, parameters)
  # Restore the tracked CKKS metadata the payload alone cannot carry.
  if scale is not None:
    setattr(value, _SCALE_ATTR, float(scale))
  if nsd is not None:
    setattr(value, _NSD_ATTR, int(nsd))
  return value


# --------------------------------------------------------------------------
# Dataset I/O (the harness's plain-text contract)
# --------------------------------------------------------------------------
def read_pixels(path: Path) -> np.ndarray:
  rows = []
  with open(path) as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      values = [float(v) for v in line.split()]
      if len(values) != 784:
        raise ValueError(f'expected 784 pixels per line, got {len(values)}')
      rows.append(values)
  return np.asarray(rows, dtype=np.float64)


def write_labels(path: Path, labels):
  """Write one predicted label per line -- the harness's scoring contract."""
  atomic_write_text(path, ''.join(f'{int(label)}\n' for label in labels))


# --------------------------------------------------------------------------
# Timing log shared across stages
# --------------------------------------------------------------------------
def append_timing(paths: Paths, stage: str, payload: dict):
  paths.iointermdir.mkdir(parents=True, exist_ok=True)
  log = paths.iointermdir / 'cross_timings.jsonl'
  record = dict(payload)
  record['stage'] = stage
  record['wall_clock'] = time.time()
  with open(log, 'a') as handle:
    handle.write(json.dumps(record) + '\n')
