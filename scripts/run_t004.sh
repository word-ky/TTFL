#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
BASE=/media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10
DATA=/media/wenchang/F/wjq/TTFL/dataset
OLD2=/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-151900-ttfl-t002-gpu0/artifacts/t002_covariate
OLD3=/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-153752-ttfl-t003-gpu0/artifacts/t003_moment_context
OUT=$AUTODL_ARTIFACTS_DIR/t004_paired_oracle
ARGS=(--baseline "$BASE" --data-root "$DATA" --identity "$OUT/clean_identity.json" --t002 "$OLD2" --t003 "$OLD3" --commit "$TTFL_COMMIT")
"$PY" -m unittest discover -s tests -v
"$PY" -u scripts/eval_t004_paired_oracle.py "${ARGS[@]}" --output "$OUT" --identity-only
"$PY" -u scripts/eval_t004_paired_oracle.py "${ARGS[@]}" --output "$AUTODL_ARTIFACTS_DIR/t004_smoke" --clients 2
"$PY" -u scripts/eval_t004_paired_oracle.py "${ARGS[@]}" --output "$OUT"
