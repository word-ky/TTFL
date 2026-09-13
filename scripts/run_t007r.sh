#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
BASE=/media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10
DATA=/media/wenchang/F/wjq/TTFL/dataset
POOLS=/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-193525-ttfl-t007-preflight-gpu1/artifacts/t007_identity_preflight/calibration_pools.json
ARGS=(--baseline "$BASE" --data-root "$DATA" --pools "$POOLS" --commit "$TTFL_COMMIT")
"$PY" -m unittest discover -s tests -v
"$PY" -u scripts/eval_t007r.py "${ARGS[@]}" --output "$AUTODL_ARTIFACTS_DIR/t007r_smoke" --clients 6
"$PY" -u scripts/eval_t007r.py "${ARGS[@]}" --output "$AUTODL_ARTIFACTS_DIR/t007r_transfer"
