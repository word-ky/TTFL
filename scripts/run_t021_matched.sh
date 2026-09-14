#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -m unittest discover -s tests -v
"$PY" -u scripts/matched_t021.py --project /home/wenchang/asdasdsad/wjq/TTFL --phase-a /home/wenchang/asdasdsad/wjq/TTFL/runs/20260914-183528-ttfl-t021-phase-a/artifacts/t021_frozen_representation_observability --output "$AUTODL_ARTIFACTS_DIR/t021_frozen_representation_observability" --commit "$TTFL_COMMIT"
