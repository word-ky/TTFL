#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
BASE=/media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10
DATA=/media/wenchang/F/wjq/TTFL/dataset
"$PY" -m unittest discover -s tests -v
"$PY" -u scripts/eval_pfllib_covariate_context.py --baseline "$BASE" --data-root "$DATA" --output "$AUTODL_ARTIFACTS_DIR/t002_smoke" --clients 2 --commit "$TTFL_COMMIT"
"$PY" -u scripts/eval_pfllib_covariate_context.py --baseline "$BASE" --data-root "$DATA" --output "$AUTODL_ARTIFACTS_DIR/t002_covariate" --commit "$TTFL_COMMIT"
