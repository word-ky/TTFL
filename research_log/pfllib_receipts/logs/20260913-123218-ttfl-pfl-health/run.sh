#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTFL/current'
export AUTODL_RUN_ID='20260913-123218-ttfl-pfl-health'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-123218-ttfl-pfl-health'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-123218-ttfl-pfl-health/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
CUDA_VISIBLE_DEVICES=0 /home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python -u scripts/run_pfllib.py --dataset PFLsmoke --data-root /media/wenchang/F/wjq/TTFL/health_data --output "$AUTODL_ARTIFACTS_DIR/health" --commit pending-baseline-import --synthetic
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
