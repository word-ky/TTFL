#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -m unittest discover -s tests -v
"$PY" -u scripts/eval_t008_source_identifiability.py \
 --baseline /media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10 \
 --data-root /media/wenchang/F/wjq/TTFL/dataset \
 --old /home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-203200-ttfl-t007r-gpu1/artifacts/t007r_transfer \
 --output "$AUTODL_ARTIFACTS_DIR/t008_source_identifiability" --commit "$TTFL_COMMIT"
