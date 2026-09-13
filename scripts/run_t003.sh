#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
BASE=/media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10
DATA=/media/wenchang/F/wjq/TTFL/dataset
OLD=/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-151900-ttfl-t002-gpu0/artifacts/t002_covariate
OUT=$AUTODL_ARTIFACTS_DIR/t003_moment_context
REF=$OUT/reference_moments.pt
ARGS=(--baseline "$BASE" --data-root "$DATA" --reference "$REF" --t002 "$OLD" --commit "$TTFL_COMMIT")
"$PY" -m unittest discover -s tests -v
"$PY" -u scripts/eval_pfllib_moment_context.py "${ARGS[@]}" --output "$OUT" --reference-only
"$PY" -u scripts/eval_pfllib_moment_context.py "${ARGS[@]}" --output "$AUTODL_ARTIFACTS_DIR/t003_smoke" --clients 2
"$PY" -u scripts/eval_pfllib_moment_context.py "${ARGS[@]}" --output "$OUT"
