#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTFL/current'
export AUTODL_RUN_ID='20260913-124728-ttfl-pfl-tiny-gpu0'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-124728-ttfl-pfl-tiny-gpu0'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-124728-ttfl-pfl-tiny-gpu0/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
CUDA_VISIBLE_DEVICES=0 TTFL_COMMIT=5db9d154450d4a0fdbe7608b132318801c3df894 bash scripts/run_pfllib_gpu.sh TinyImagenet
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
