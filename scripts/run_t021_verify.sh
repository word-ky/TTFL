#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -u scripts/verify_t021.py --project /home/wenchang/asdasdsad/wjq/TTFL --scoring /home/wenchang/asdasdsad/wjq/TTFL/runs/20260914-184639-ttfl-t021-score/artifacts/t021_frozen_representation_observability --output "$AUTODL_ARTIFACTS_DIR/t021_frozen_representation_observability" --commit "$TTFL_COMMIT"
