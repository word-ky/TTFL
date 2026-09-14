#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python -m unittest discover -s tests -v
