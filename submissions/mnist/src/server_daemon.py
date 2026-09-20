#!/usr/bin/env python3
"""The CROSS server: one resident ``Mapping`` on the TPU, behind a socket.

Started by ``server_preprocess_model`` (benchmark stage 3), which blocks until
this process reports that the Mapping is materialized -- BSGS diagonals
encoded, evaluation and rotation keys installed, constants bound, XLA
compilation done.  ``server_encrypted_compute`` (stage 7) then sends one
request per run and this process does nothing but ``Mapping.execute``.

Protocol: newline-delimited JSON over a Unix socket.

  {"cmd": "ping"}                              -> {"ok", "ready"}
  {"cmd": "manifest"}                          -> the ring/deployment manifest
  {"cmd": "infer", "indir", "outdir", "count"} -> per-chunk execute timings
  {"cmd": "stop"}                              -> shut down

Only this process touches the accelerator.  A TPU admits exactly one client
process at a time, which is also why the benchmark's client stages run with
``JAX_PLATFORMS=cpu``: the client is a CPU-side entity in this deployment.
"""
from __future__ import annotations

import argparse
import json
import os

import signal
import socket
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_task as ct  # noqa: E402

import numpy as np  # noqa: E402

IDLE_TIMEOUT = float(os.environ.get('CROSS_DAEMON_IDLE_TIMEOUT', '3600'))
# How long accept() blocks before re-checking the idle deadline and stop flag.
ACCEPT_TIMEOUT = 5.0
# A request is one JSON object terminated by a newline. Cap it so a confused
# or hostile peer cannot make the server buffer without bound.
MAX_MESSAGE_BYTES = 4 << 20


def read_message(connection):
  """Read one newline-terminated JSON request, or None if the peer hung up.

  A plain ``recv`` loop that stops only on a newline hangs forever if the peer
  dies mid-message, and grows without bound if the newline never arrives. This
  honours EOF and a size cap, and the caller sets a socket timeout.
  """
  chunks, size = [], 0
  while True:
    chunk = connection.recv(65536)
    if not chunk:
      if not chunks:
        return None
      raise ConnectionError('peer closed the connection mid-request')
    chunks.append(chunk)
    size += len(chunk)
    if size > MAX_MESSAGE_BYTES:
      raise ValueError(f'request exceeds {MAX_MESSAGE_BYTES} bytes')
    if chunk.endswith(b'\n'):
      break
  return json.loads(b''.join(chunks).decode())


def write_message(connection, payload: dict) -> None:
  connection.sendall((json.dumps(payload) + '\n').encode())


class Server:

  def __init__(self, size: int, device_count: int):
    self.paths = ct.Paths(size)
    self.device_count = device_count
    self.ready = False
    self.mapping = None
    self.manifest = None
    self.setup_timings = {}
    self._stopping = False

  # ---------------------------------------------------------------- setup
  def materialize(self):
    import jax

    from jaxite_word import Mapping

    started = time.perf_counter()
    _, _, packed = ct.build_packed()
    packed_at = time.perf_counter()

    keys = ct.load_keys(self.paths.pubkeydir, self.paths.seckeydir)
    keys_at = time.perf_counter()

    available = jax.devices('tpu')
    if len(available) < self.device_count:
      raise RuntimeError(
          f'requested {self.device_count} TPU chips, found {len(available)}. '
          f'Set CROSS_DEVICE_COUNT to at most {len(available)}.')
    devices = tuple(available[:self.device_count])
    # One ciphertext per chip. A larger per-chip batch aborts the XLA TPU
    # fusion emitter on this stack (see RESULTS.md section 2), so the local
    # batch is fixed at 1 rather than exposed as a knob that crashes.
    global_batch = self.device_count

    # Everything expensive happens here: level/scale/noise propagation, a BSGS
    # plan per matvec, rotation-key materialization, constant encoding and the
    # fused XLA lowering for this exact device tuple.
    self.mapping = Mapping(
        packed,
        keys=keys,
        global_batch=global_batch,
        devices=devices,
        # Offline diagonal encoding only; it has no bearing on what stage 7
        # measures, and this host has cores to spare.
        bsgs_n_jobs=int(os.environ.get('CROSS_BSGS_JOBS', '1')),
    )
    mapping_at = time.perf_counter()

    # Compile is lazy until the first execute in some paths; force it now so
    # stage 7 measures steady-state inference, not compilation.
    warm = self._warmup()
    warm_at = time.perf_counter()

    self.setup_timings = {
        'pack_s': packed_at - started,
        'key_load_s': keys_at - packed_at,
        'mapping_build_s': mapping_at - keys_at,
        'warmup_s': warm_at - mapping_at,
        'total_s': warm_at - started,
        'warmup_execute_s': warm,
    }
    self.manifest = ct.write_manifest(
        self.paths.manifest_file,
        packed,
        input_scale=self.mapping.input_spec.scale,
        output_scale=self.mapping.output_spec.scale,
        global_batch=global_batch,
        device_count=self.device_count,
        extra={
            'input_payload_shape': list(self.mapping.input_spec.payload_shape),
            'output_payload_shape': list(self.mapping.output_spec.payload_shape),
            'input_moduli': [int(m) for m in self.mapping.input_spec.moduli],
            'output_moduli': [int(m) for m in self.mapping.output_spec.moduli],
            'input_precision': 32,
            'input_nsd': self.mapping.input_spec.nsd,
            'output_nsd': self.mapping.output_spec.nsd,
            'rotation_key_count':
                len(self.mapping.required_rotation_indices),
            'live_memory': self.mapping.estimate_live_memory(),
            'setup_timings': self.setup_timings,
            'devices': [str(d) for d in devices],
        },
    )
    self.ready = True

  def _warmup(self) -> float:
    """One execute on a zero input, so XLA compilation is not measured later."""
    import jax

    zeros = [np.zeros(784) for _ in range(self.mapping.global_batch)]
    encrypted = self.mapping.encrypt_input(
        zeros if self.mapping.global_batch > 1 else zeros[0])
    started = time.perf_counter()
    output = self.mapping.execute(encrypted)
    jax.block_until_ready(output.polynomial)
    return time.perf_counter() - started

  # ------------------------------------------------------------- inference
  def infer(self, indir: str, outdir: str, count: int) -> dict:
    """Execute `count` encrypted inferences, streaming one chunk at a time.

    Chunks are loaded, executed and written back one at a time rather than
    reading every ciphertext up front. At the large instance the eager form
    would hold 10000 x 3.0 MB of input plus 10000 x 517 KB of results -- about
    35 GB -- live at once, for no gain: the Mapping consumes exactly
    `global_batch` ciphertexts per call and nothing needs the rest yet.
    Peak memory is now one chunk regardless of instance size.
    """
    import jax

    indir, outdir = Path(indir), Path(outdir)
    if count <= 0:
      raise ValueError(f'count must be positive, got {count}')
    missing = [index for index in range(count)
               if not (indir / f'cipher_input_{index}.npz').exists()]
    if missing:
      preview = ', '.join(str(i) for i in missing[:5])
      raise FileNotFoundError(
          f'{len(missing)} of {count} input ciphertexts are missing from '
          f'{indir} (first: {preview}). Did stage 6 complete?')
    outdir.mkdir(parents=True, exist_ok=True)

    batch = self.mapping.global_batch
    expect = self.mapping.input_spec.payload_shape
    output_fields = {
        'moduli': np.asarray(
            [int(m) for m in self.mapping.output_spec.moduli], dtype=np.int64),
        'degree_layout': np.asarray(
            self.mapping.output_spec.degree_layout, dtype=np.int64),
        'precision': np.int64(32),
        'scale': np.float64(self.mapping.output_spec.scale),
        'nsd': np.int64(self.mapping.output_spec.nsd
                        if self.mapping.output_spec.nsd is not None else -1),
    }

    chunk_times, transfer_times = [], []
    load_s = store_s = 0.0

    for start in range(0, count, batch):
      lanes = min(batch, count - start)

      load_started = time.perf_counter()
      window, metadata = [], None
      for index in range(start, start + lanes):
        payload, metadata = ct.load_ciphertext_payload(
            indir / f'cipher_input_{index}.npz', expect_shape=expect)
        window.append(payload)
      # The Mapping is compiled for exactly `batch` ciphertexts. Repeat the
      # last one to fill a short final chunk; those lanes are computed and
      # discarded.
      window += [window[-1]] * (batch - lanes)
      stacked = np.concatenate(window, axis=0)
      del window
      load_s += time.perf_counter() - load_started

      transfer_started = time.perf_counter()
      ciphertext = ct.build_polynomial(
          stacked, metadata['moduli'], metadata['degree_layout'],
          metadata['precision'],
          # A present-but-None key must still fall back to the planned value.
          scale=(metadata.get('scale') if metadata.get('scale') is not None
                 else self.mapping.input_spec.scale),
          nsd=(metadata.get('nsd') if metadata.get('nsd') is not None
               else self.mapping.input_spec.nsd))
      jax.block_until_ready(ciphertext.polynomial)
      transfer_times.append(time.perf_counter() - transfer_started)

      execute_started = time.perf_counter()
      output = self.mapping.execute(ciphertext)
      jax.block_until_ready(output.polynomial)
      chunk_times.append(time.perf_counter() - execute_started)

      store_started = time.perf_counter()
      payload = np.asarray(output.polynomial)
      for lane in range(lanes):
        result = payload[lane:lane + 1]
        ct.atomic_write_bytes(
            outdir / f'cipher_result_{start + lane}.npz',
            lambda handle, value=result: np.savez(
                handle, payload=value, **output_fields))
      store_s += time.perf_counter() - store_started
      del stacked, ciphertext, output, payload

    execute_total = float(sum(chunk_times))
    return {
        'count': count,
        'global_batch': batch,
        'device_count': self.device_count,
        'chunks': len(chunk_times),
        'chunk_execute_s': chunk_times,
        'chunk_transfer_s': transfer_times,
        'execute_total_s': execute_total,
        'load_ciphertexts_s': load_s,
        'store_ciphertexts_s': store_s,
        'per_inference_s': execute_total / count,
        'throughput_infer_per_s': count / execute_total,
    }

  # ---------------------------------------------------------------- serving
  def shutdown(self):
    """Release the accelerator and remove this server's control files."""
    if self.mapping is not None:
      try:
        # A materialized Mapping pins its executable, evaluation keys and
        # encoded constants in HBM; dropping the reference frees none of it.
        self.mapping.release()
      except Exception:
        traceback.print_exc()
      self.mapping = None
    self.ready = False
    for path in (self.paths.socket_path, self.paths.daemon_pidfile,
                 self.paths.ready_marker):
      try:
        path.unlink(missing_ok=True)
      except OSError:
        pass

  def handle(self, request: dict) -> dict:
    command = request.get('cmd')
    if command == 'ping':
      return {'ok': True, 'ready': self.ready}
    if command == 'manifest':
      return {'ok': True, 'manifest': self.manifest}
    if command == 'infer':
      if not self.ready:
        return {'ok': False, 'error': 'server is not ready'}
      for field in ('indir', 'outdir', 'count'):
        if field not in request:
          return {'ok': False, 'error': f'infer requires {field!r}'}
      return {'ok': True, 'result': self.infer(
          request['indir'], request['outdir'], int(request['count']))}
    if command == 'stop':
      return {'ok': True, 'stopping': True}
    return {'ok': False, 'error': f'unknown command {command!r}'}

  def serve(self):
    socket_path = self.paths.socket_path
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
      listener.bind(str(socket_path))
      listener.listen(8)
      listener.settimeout(ACCEPT_TIMEOUT)

      print('[cross-server] listening', flush=True)
      last_activity = time.time()
      while not self._stopping:
        try:
          connection, _ = listener.accept()
        except socket.timeout:
          if time.time() - last_activity > IDLE_TIMEOUT:
            print('[cross-server] idle timeout, exiting', flush=True)
            break
          continue
        except OSError as error:
          if self._stopping:
            break
          print(f'[cross-server] accept failed: {error}', flush=True)
          continue

        last_activity = time.time()
        with connection:
          # A request-time failure is reported to the caller; it must not take
          # the server down, because rebuilding the Mapping costs ~8 minutes.
          try:
            request = read_message(connection)
          except Exception as error:
            print(f'[cross-server] malformed request: {error}', flush=True)
            continue
          if request is None:
            continue
          try:
            response = self.handle(request)
          except Exception as error:
            traceback.print_exc()
            response = {'ok': False,
                        'error': f'{type(error).__name__}: {error}'}
          if response.pop('stopping', False):
            self._stopping = True
          try:
            write_message(connection, response)
          except OSError as error:
            print(f'[cross-server] could not reply: {error}', flush=True)
        last_activity = time.time()
    finally:
      listener.close()
      self.shutdown()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('size', type=int)
  parser.add_argument('--devices', type=int, default=ct.DEFAULT_DEVICE_COUNT)
  args = parser.parse_args()

  if args.devices < 1:
    raise SystemExit(f'--devices must be >= 1, got {args.devices}')

  server = Server(args.size, args.devices)
  paths = server.paths
  paths.ensure_rundir()

  def on_signal(signum, _frame):
    # Stop accepting, release HBM and remove the socket/pidfile, so the next
    # run can take the accelerator. Without this a SIGTERM leaves the control
    # files behind and the TPU pinned until the kernel reaps the process.
    print(f'[cross-server] signal {signum}, shutting down', flush=True)
    server._stopping = True

  for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    try:
      signal.signal(signum, on_signal)
    except (ValueError, OSError):
      pass

  paths.daemon_pidfile.write_text(f'{os.getpid()}\n')
  try:
    server.materialize()
    print(f'[cross-server] ready: {json.dumps(server.setup_timings)}',
          flush=True)
    ct.atomic_write_text(paths.ready_marker, 'ready\n')
    server.serve()
  except BaseException:
    # Record why, so server_preprocess_model can fail fast with the real
    # reason instead of waiting out its whole startup timeout.
    detail = traceback.format_exc()
    traceback.print_exc()
    try:
      ct.atomic_write_text(paths.failed_marker, detail)
    except OSError:
      pass
    server.shutdown()
    raise
  finally:
    paths.daemon_pidfile.unlink(missing_ok=True)


if __name__ == '__main__':
  main()
