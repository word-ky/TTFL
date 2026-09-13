#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
BASE=/media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10
DATA=/media/wenchang/F/wjq/TTFL/dataset
OLD=/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-173459-ttfl-t005-gpu1/artifacts/t005_pairing_audit
ARGS=(--baseline "$BASE" --data-root "$DATA" --t005 "$OLD" --commit "$TTFL_COMMIT")
"$PY" -m unittest discover -s tests -v
"$PY" -u scripts/eval_t006_semantic_pairing.py "${ARGS[@]}" --output "$AUTODL_ARTIFACTS_DIR/t006_smoke" --clients 2
"$PY" -u scripts/eval_t006_semantic_pairing.py "${ARGS[@]}" --output "$AUTODL_ARTIFACTS_DIR/t006_semantic_pairing"
