#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTFL/current'
export AUTODL_RUN_ID='20260913-153752-ttfl-t003-gpu0'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-153752-ttfl-t003-gpu0'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-153752-ttfl-t003-gpu0/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
CUDA_VISIBLE_DEVICES=0 TTFL_COMMIT=7bc31ded5e2a0dc6ad7e20b7688240c34c24c366 bash scripts/run_t003.sh
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
