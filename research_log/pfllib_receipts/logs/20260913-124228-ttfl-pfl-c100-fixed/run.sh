#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTFL/current'
export AUTODL_RUN_ID='20260913-124228-ttfl-pfl-c100-fixed'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-124228-ttfl-pfl-c100-fixed'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-124228-ttfl-pfl-c100-fixed/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
set -e; export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MPLBACKEND=Agg; PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python; OUT=/media/wenchang/F/wjq/TTFL/runs/$AUTODL_RUN_ID; mkdir -p "$OUT"; printf "%s\n" "$OUT" > "$AUTODL_ARTIFACTS_DIR/output_path.txt"; "$PY" -m unittest discover -s tests -v; "$PY" -u scripts/eval_pfllib_context.py --baseline /media/wenchang/F/wjq/TTFL/runs/20260913-123913-ttfl-pfl-c100-gpu1/Cifar100_smoke; "$PY" -u scripts/run_pfllib.py --dataset Cifar100 --data-root /media/wenchang/F/wjq/TTFL/dataset --output "$OUT/Cifar100" --commit 5db9d154450d4a0fdbe7608b132318801c3df894; "$PY" -u scripts/eval_pfllib_context.py --baseline "$OUT/Cifar100"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
