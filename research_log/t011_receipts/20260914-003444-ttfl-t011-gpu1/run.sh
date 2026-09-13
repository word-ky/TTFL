#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTFL/current'
export AUTODL_RUN_ID='20260914-003444-ttfl-t011-gpu1'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260914-003444-ttfl-t011-gpu1'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260914-003444-ttfl-t011-gpu1/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
CUDA_VISIBLE_DEVICES=1 TTFL_COMMIT=f32367413cae9f6629b5648d2dd665a73c8c078c bash scripts/run_t011.sh
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
