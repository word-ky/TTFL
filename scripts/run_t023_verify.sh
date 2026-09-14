#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -u scripts/verify_t023.py --project /home/wenchang/asdasdsad/wjq/TTFL --scoring /home/wenchang/asdasdsad/wjq/TTFL/runs/20260914-213328-ttfl-t023-score-final/artifacts/t023_rotation_ssl_alignment --output "$AUTODL_ARTIFACTS_DIR/t023_rotation_ssl_alignment" --commit "$TTFL_COMMIT"
