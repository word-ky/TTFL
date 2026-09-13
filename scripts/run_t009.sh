#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -m unittest discover -s tests -v
"$PY" -u scripts/eval_t009_natural_context.py \
 --baseline /media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10 \
 --data-root /media/wenchang/F/wjq/TTFL/dataset \
 --t007r /home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-203200-ttfl-t007r-gpu1/artifacts/t007r_transfer \
 --t008 /home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-213117-ttfl-t008-gpu1/artifacts/t008_source_identifiability \
 --output "$AUTODL_ARTIFACTS_DIR/t009_natural_context" --commit "$TTFL_COMMIT"
