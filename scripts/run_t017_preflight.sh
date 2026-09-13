#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -m unittest discover -s tests -v
"$PY" -u scripts/preflight_t017.py --project /home/wenchang/asdasdsad/wjq/TTFL --data /media/wenchang/F/wjq/TTFL/dataset --output "$AUTODL_ARTIFACTS_DIR/t017_channel_noise_decomposition" --commit "$TTFL_COMMIT"
