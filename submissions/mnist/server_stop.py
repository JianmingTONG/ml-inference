#!/usr/bin/env python3
"""Release the TPU held by the resident CROSS server (not a benchmark stage)."""
import os
import socket
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import cross_task as ct  # noqa: E402


def main():
    size = ct.parse_size(sys.argv) if len(sys.argv) > 1 else ct.read_active_size()
    paths = ct.Paths(size)
    if not paths.socket_path.exists():
        print('[server] no resident CROSS server')
        return
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(120)
    connection.connect(str(paths.socket_path))
    connection.sendall(b'{"cmd": "stop"}\n')
    connection.recv(4096)
    connection.close()
    print('[server] CROSS server stopped')


if __name__ == '__main__':
    main()
