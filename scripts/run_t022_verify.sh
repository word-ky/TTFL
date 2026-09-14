#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -u scripts/verify_t022.py --project /home/wenchang/asdasdsad/wjq/TTFL --scoring /home/wenchang/asdasdsad/wjq/TTFL/runs/20260914-194845-ttfl-t022-score-receipt/artifacts/t022_state_response_semantics --output "$AUTODL_ARTIFACTS_DIR/t022_state_response_semantics" --commit "$TTFL_COMMIT"
