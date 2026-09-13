#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
BASE=/home/wenchang/asdasdsad/wjq/TTFL
PY="$BASE/.venv/bin/python"
"$PY" -m unittest discover -s tests -v
"$PY" -u -m scripts.run_t001 --data-root "$BASE/shared/digits" --mnistm-train "$BASE/shared/digits/MNISTM/MNIST-M/training" --output "$AUTODL_ARTIFACTS_DIR/t001" --commit "$TTFL_COMMIT" "$@"
