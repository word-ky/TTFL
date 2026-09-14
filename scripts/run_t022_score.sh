#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -m unittest discover -s tests -v
"$PY" -u scripts/score_t022.py --project /home/wenchang/asdasdsad/wjq/TTFL --matched /home/wenchang/asdasdsad/wjq/TTFL/runs/20260914-194335-ttfl-t022-matched/artifacts/t022_state_response_semantics --output "$AUTODL_ARTIFACTS_DIR/t022_state_response_semantics" --commit "$TTFL_COMMIT"
