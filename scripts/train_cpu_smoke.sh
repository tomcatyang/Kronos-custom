#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_cpu_smoke.sh
#   bash scripts/train_cpu_smoke.sh --skip-basemodel
#   bash scripts/train_cpu_smoke.sh --skip-tokenizer
#   bash scripts/train_cpu_smoke.sh --skip-existing
#
# This script runs the CPU smoke-training config. Extra arguments are passed
# directly to finetune_csv/train_sequential.py.

cd "$(dirname "$0")/.."

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv. Run: bash init.sh" >&2
  exit 1
fi

cd finetune_csv

../.venv/bin/python train_sequential.py \
  --config configs/config_cpu_smoke.yaml \
  "$@"
