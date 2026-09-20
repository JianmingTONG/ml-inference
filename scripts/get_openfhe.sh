#!/usr/bin/env bash
# Copyright (c) 2025 HomomorphicEncryption.org
# Licensed under the Apache v2 License. See LICENSE.md.
#
# ------------------------------------------------------------
# Replaced by the CROSS/TPU submission.
#
# The harness calls this unconditionally for a local submission. This
# submission implements CKKS in JAX (CROSS / jaxite_word) and links no
# OpenFHE, so there is nothing to fetch or build here.
#
# To restore the upstream OpenFHE reference submission, restore this script
# and scripts/build_task.sh from the fhe-benchmarking/ml-inference main branch.
# ------------------------------------------------------------
set -euo pipefail
echo "[get-openfhe] CROSS submission: OpenFHE is not used, nothing to build."
