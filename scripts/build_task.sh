#!/usr/bin/env bash
# Copyright (c) 2025 HomomorphicEncryption.org
# Licensed under the Apache v2 License. See LICENSE.md.
#
# ------------------------------------------------------------
# Usage: ./scripts/build_task.sh <TASK_DIR>
#
# Build step for the CROSS/TPU submission. CROSS (jaxite_word) is a JAX
# library used in place, so "building" means checking the runtime is present
# and the trained weights exist -- there is nothing to compile.
#
# The expensive artifact, the compiled CKKS Mapping with its evaluation keys,
# BSGS diagonals and encoded constants, is built by benchmark stage 3
# (server_preprocess_model) on the accelerator, where the harness measures it.
# ------------------------------------------------------------
set -euo pipefail

ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
TASK_DIR="${1:-$ROOT/submissions/mnist}"
[[ "$TASK_DIR" = /* ]] || TASK_DIR="$ROOT/$TASK_DIR"

python3 - "$ROOT" "$TASK_DIR" <<'PY'
import importlib.util
import os
import sys

root, task_dir = sys.argv[1], sys.argv[2]

missing = [name for name in ("jax", "torch", "numpy")
           if importlib.util.find_spec(name) is None]
if missing:
    sys.exit(f"[build] missing Python packages: {', '.join(missing)}. "
             f"See {task_dir}/README.md for the environment setup.")

# CROSS is used in place and is not packaged, so locate it the same way the
# submission does and fail with a path list rather than a bare ImportError.
sys.path.insert(0, os.path.join(task_dir, "src"))
import cross_task  # noqa: E402

print(f"[build] CROSS found at {cross_task.CROSS_ROOT or '(on PYTHONPATH)'}")

weights = os.path.join(task_dir, "model", "he_mlp_weights.pth")
if not os.path.isfile(weights):
    sys.exit(f"[build] {weights} not found. Train it with:\n"
             f"    python3 {task_dir}/model/train_he_mlp.py")
print(f"[build] model weights present: {os.path.basename(weights)}")
PY

echo "[build] CROSS submission ready (no compilation required)."
