#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTFL/current'
export AUTODL_RUN_ID='20260913-193525-ttfl-t007-preflight-gpu1'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-193525-ttfl-t007-preflight-gpu1'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-193525-ttfl-t007-preflight-gpu1/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2; /home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python -m unittest discover -s tests -v && /home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python -u scripts/preflight_t007.py --baseline /media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10 --data-root /media/wenchang/F/wjq/TTFL/dataset --output "$AUTODL_ARTIFACTS_DIR/t007_identity_preflight" --commit bc07059
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
