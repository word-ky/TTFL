#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -u scripts/phase_a_t023.py --extraction /home/wenchang/asdasdsad/wjq/TTFL/runs/20260914-212025-ttfl-t023-extract-gpu1/artifacts/t023_rotation_ssl_alignment --output "$AUTODL_ARTIFACTS_DIR/t023_rotation_ssl_alignment" --commit "$TTFL_COMMIT"
