#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export MPLBACKEND=Agg
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
DATA=/media/wenchang/F/wjq/TTFL/dataset
OUT=/media/wenchang/F/wjq/TTFL/runs/$AUTODL_RUN_ID
mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$AUTODL_ARTIFACTS_DIR/output_path.txt"
"$PY" -m unittest discover -s tests -v
for dataset in "$@"; do
  "$PY" -u scripts/run_pfllib.py --dataset "$dataset" --data-root "$DATA" --output "$OUT/${dataset}_smoke" --commit "$TTFL_COMMIT" --smoke
  "$PY" -u scripts/eval_pfllib_context.py --baseline "$OUT/${dataset}_smoke"
  "$PY" -u scripts/run_pfllib.py --dataset "$dataset" --data-root "$DATA" --output "$OUT/$dataset" --commit "$TTFL_COMMIT"
  "$PY" -u scripts/eval_pfllib_context.py --baseline "$OUT/$dataset"
done
