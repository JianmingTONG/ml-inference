#!/usr/bin/env python3
"""Stage 3 -- server: materialize the encrypted model on the TPU.

Every offline cost of the CROSS deployment is paid and measured here: BSGS plan
selection and diagonal encoding per MatVec, evaluation- and rotation-key
materialization, plaintext constant encoding, device placement across the TPU
chips, and the fused XLA compilation. The result stays resident in a daemon,
because a compiled JAX executable cannot cross a process boundary and
rebuilding it per request would charge compilation time to inference.

The harness invokes this stage with no arguments, so the instance size comes
from the handshake ``client_key_generation`` wrote.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import cross_task as ct  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
STARTUP_TIMEOUT = float(os.environ.get('CROSS_STARTUP_TIMEOUT', '5400'))


def stop_existing(paths):
    """Stop every resident CROSS server, not just this instance size's.

    A TPU admits exactly one process. A daemon left over from a different
    instance size still owns the chips, and the new one would fail to
    initialize the backend, so sweep the whole run directory.
    """
    run_root = paths.rundir.parent
    if not run_root.exists():
        return
    for pidfile in sorted(run_root.glob('*/cross_server.pid')):
        try:
            pid = int(pidfile.read_text().strip())
        except (ValueError, OSError):
            pidfile.unlink(missing_ok=True)
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pidfile.unlink(missing_ok=True)
            continue
        print(f"         [server] stopping resident CROSS server pid {pid}")
        for _ in range(150):
            time.sleep(0.2)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        pidfile.unlink(missing_ok=True)
    time.sleep(3)  # the driver releases the accelerator a moment after exit


def clear_stale_tpu_lock():
    """Remove a libtpu lockfile orphaned by a killed process."""
    lockfile = Path('/tmp/libtpu_lockfile')
    try:
        if lockfile.exists():
            lockfile.unlink()
            print('         [server] cleared stale libtpu lockfile')
    except OSError:
        pass


def main():
    size = ct.read_active_size()
    paths = ct.Paths(size)
    paths.iodir.mkdir(parents=True, exist_ok=True)
    paths.ensure_rundir()
    stop_existing(paths)
    clear_stale_tpu_lock()
    for marker in (paths.ready_marker, paths.failed_marker):
        marker.unlink(missing_ok=True)

    devices = os.environ.get('CROSS_DEVICE_COUNT', str(ct.DEFAULT_DEVICE_COUNT))
    log = open(paths.daemon_log, 'w')
    started = time.perf_counter()
    daemon = subprocess.Popen(
        [sys.executable, os.path.join(SRC, 'server_daemon.py'), str(size),
         '--devices', devices],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    while True:
        if paths.ready_marker.exists():
            break
        if paths.failed_marker.exists():
            print(paths.failed_marker.read_text(), file=sys.stderr)
            sys.exit('[server] CROSS Mapping materialization failed')
        # Watch the process itself, not only the markers: a daemon killed by
        # the OOM killer or aborting inside XLA never writes either one.
        exit_code = daemon.poll()
        if exit_code is not None:
            try:
                tail = ''.join(paths.daemon_log.read_text(
                    errors='replace').splitlines(keepends=True)[-25:])
                print(tail, file=sys.stderr)
            except OSError:
                pass
            sys.exit(f'[server] CROSS server exited with code {exit_code} before '
                     f'becoming ready (see {paths.daemon_log})')
        if time.perf_counter() - started > STARTUP_TIMEOUT:
            daemon.terminate()
            sys.exit(f'[server] timed out after {STARTUP_TIMEOUT:.0f}s '
                     f'materializing the CROSS Mapping (see {paths.daemon_log})')
        time.sleep(0.5)

    manifest = ct.read_manifest(paths.manifest_file)
    setup, memory = manifest['setup_timings'], manifest['live_memory']
    print(f"         [server] CROSS Mapping resident on {manifest['device_count']} "
          f"TPU v6e chip(s), global_batch={manifest['global_batch']}")
    print(f"         [server] ring degree={manifest['degree']} "
          f"slots={manifest['num_slots']} depth={manifest['depth']} "
          f"security={manifest['security_bits']}-bit {manifest['security_model']}")
    print(f"         [server] setup: pack {setup['pack_s']:.2f}s, "
          f"keys {setup['key_load_s']:.2f}s, mapping {setup['mapping_build_s']:.2f}s, "
          f"warmup {setup['warmup_s']:.2f}s")
    print(f"         [server] rotation keys: {manifest['rotation_key_count']}, "
          f"live memory estimate: {memory['total_bytes'] / 2**30:.2f} GiB")

    elapsed = time.perf_counter() - started
    ct.append_timing(paths, 'server_preprocess_model',
                     {'startup_wall_s': elapsed, **setup})
    ct.report_server_steps(paths, {
        'server_model_preprocessing_total': elapsed,
        'server_mapping_build': setup['mapping_build_s'],
        'server_mapping_warmup': setup['warmup_s'],
    })


if __name__ == '__main__':
    main()
