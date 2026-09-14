#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -m unittest discover -s tests -v
"$PY" -u scripts/extract_t021.py --project /home/wenchang/asdasdsad/wjq/TTFL --baseline /media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10 --data-root /media/wenchang/F/wjq/TTFL/dataset --output "$AUTODL_ARTIFACTS_DIR/t021_frozen_representation_observability" --commit "$TTFL_COMMIT"
