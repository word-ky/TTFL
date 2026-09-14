#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -u scripts/eval_t018r.py --project /home/wenchang/asdasdsad/wjq/TTFL --output "$AUTODL_ARTIFACTS_DIR/t018r_constrained_prevalence" --commit "$TTFL_COMMIT" --preflight /home/wenchang/asdasdsad/wjq/TTFL/runs/20260914-124138-ttfl-t018r-preflight/artifacts/t018r_constrained_prevalence
