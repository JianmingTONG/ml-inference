#!/usr/bin/env python3
"""Stage 7 -- server: run the encrypted inference on the TPU.

Hands the resident daemon the uploaded ciphertexts. Everything expensive was
already paid in stage 3, so what this stage measures is ``Mapping.execute``
plus ciphertext file I/O -- the encrypted MNIST inference itself. The pure
accelerator time is published separately through the harness's
``server_reported_steps.json`` hook.
"""
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import cross_task as ct  # noqa: E402

TIMEOUT = float(os.environ.get('CROSS_REQUEST_TIMEOUT', '14400'))
CONNECT_ATTEMPTS = 10


def connect(paths):
    """Connect to the resident server, retrying a listener not yet up."""
    last = None
    for attempt in range(CONNECT_ATTEMPTS):
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(TIMEOUT)
            connection.connect(str(paths.socket_path))
            return connection
        except OSError as error:
            last = error
            connection.close()
            time.sleep(0.5 * (attempt + 1))
    sys.exit(f'[server] cannot reach the CROSS server at {paths.socket_path}: '
             f'{last}. Is it still running? (see {paths.daemon_log})')


def request(paths, payload):
    connection = connect(paths)
    try:
        connection.sendall((json.dumps(payload) + '\n').encode())
        chunks = []
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                if not chunks:
                    sys.exit('[server] CROSS server closed the connection without '
                             f'replying (see {paths.daemon_log})')
                break
            chunks.append(chunk)
            if chunk.endswith(b'\n'):
                break
    except socket.timeout:
        sys.exit(f'[server] no reply within {TIMEOUT:.0f}s '
                 '(raise CROSS_REQUEST_TIMEOUT)')
    finally:
        connection.close()
    try:
        return json.loads(b''.join(chunks).decode())
    except json.JSONDecodeError as error:
        sys.exit(f'[server] malformed reply from the CROSS server: {error}')


def main():
    size = ct.parse_size(sys.argv)
    paths = ct.Paths(size)
    if not paths.socket_path.exists():
        sys.exit(f'[server] no resident CROSS server at {paths.socket_path}; '
                 'stage 3 (server_preprocess_model) must run first')

    started = time.perf_counter()
    response = request(paths, {
        'cmd': 'infer',
        'indir': str(paths.ctxtupdir),
        'outdir': str(paths.ctxtdowndir),
        'count': paths.batch_size,
    })
    if not response.get('ok'):
        sys.exit(f"[server] encrypted compute failed: {response.get('error')}")
    result = response['result']
    elapsed = time.perf_counter() - started

    print(f"         [server] encrypted MNIST inference on "
          f"{result['device_count']} TPU v6e chip(s): "
          f"{result['execute_total_s']:.3f}s of accelerator time for "
          f"{result['count']} inference(s) -> "
          f"{result['per_inference_s'] * 1000:.2f} ms each, "
          f"{result['throughput_infer_per_s']:.2f} inferences/s")
    ct.append_timing(paths, 'server_encrypted_compute',
                     {**result, 'stage_wall_s': elapsed})
    ct.report_server_steps(paths, {
        'server_encrypted_compute_total': elapsed,
        'server_tpu_execute': result['execute_total_s'],
        'server_tpu_execute_per_inference': result['per_inference_s'],
        'server_ciphertext_load': result['load_ciphertexts_s'],
        'server_ciphertext_store': result['store_ciphertexts_s'],
    })


if __name__ == '__main__':
    main()
