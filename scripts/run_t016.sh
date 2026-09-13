#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -m unittest discover -s tests -v
"$PY" -u scripts/eval_t016_confusion_prevalence.py --project /home/wenchang/asdasdsad/wjq/TTFL --data /media/wenchang/F/wjq/TTFL/dataset --output "$AUTODL_ARTIFACTS_DIR/t016_confusion_debiased_semantics" --commit "$TTFL_COMMIT"
