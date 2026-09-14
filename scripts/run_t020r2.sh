#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -m unittest discover -s tests -v
"$PY" -u scripts/resume_t020r2.py --project /home/wenchang/asdasdsad/wjq/TTFL --output "$AUTODL_ARTIFACTS_DIR/t020r2_mc_convergence" --commit "$TTFL_COMMIT"
