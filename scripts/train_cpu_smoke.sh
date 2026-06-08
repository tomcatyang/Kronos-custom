#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_cpu_smoke.sh
#     Train both tokenizer and basemodel with the CPU smoke config.
#
#   bash scripts/train_cpu_smoke.sh --skip-basemodel
#     Train tokenizer only.
#
#   bash scripts/train_cpu_smoke.sh --skip-tokenizer
#     Train basemodel only. Requires an existing fine-tuned tokenizer path.
#
#   bash scripts/train_cpu_smoke.sh --skip-existing
#     Train missing phases only; skip tokenizer/basemodel if best_model exists.
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
