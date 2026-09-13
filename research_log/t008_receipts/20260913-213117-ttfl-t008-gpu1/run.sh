#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTFL/current'
export AUTODL_RUN_ID='20260913-213117-ttfl-t008-gpu1'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-213117-ttfl-t008-gpu1'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-213117-ttfl-t008-gpu1/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1 TTFL_COMMIT=4a71d7d; bash scripts/run_t008.sh
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
