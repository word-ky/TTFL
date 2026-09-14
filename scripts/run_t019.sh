#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -m unittest discover -s tests -v
"$PY" -u scripts/eval_t019.py --project /home/wenchang/asdasdsad/wjq/TTFL --output "$AUTODL_ARTIFACTS_DIR/t019_real_channel_heterogeneity" --commit "$TTFL_COMMIT"
