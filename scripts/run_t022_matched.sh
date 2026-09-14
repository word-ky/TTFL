#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -u scripts/matched_t021.py --response --project /home/wenchang/asdasdsad/wjq/TTFL --phase-a /home/wenchang/asdasdsad/wjq/TTFL/runs/20260914-194148-ttfl-t022-phase-a/artifacts/t022_state_response_semantics --output "$AUTODL_ARTIFACTS_DIR/t022_state_response_semantics" --commit "$TTFL_COMMIT"
