#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTFL/current'
export AUTODL_RUN_ID='20260913-123506-ttfl-pfl-data'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-123506-ttfl-pfl-data'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-123506-ttfl-pfl-data/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt; export OMP_NUM_THREADS=2; for dataset in MNIST Cifar10 Cifar100 TinyImagenet; do /home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python -u scripts/prepare_pfllib.py --dataset "$dataset" --root /media/wenchang/F/wjq/TTFL --mnist-root /home/wenchang/asdasdsad/wjq/TTFL/shared/digits/MNIST || exit $?; done
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
