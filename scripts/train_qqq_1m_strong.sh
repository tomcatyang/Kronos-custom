#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_qqq_1m_strong.sh
#     Train both tokenizer and basemodel with the stronger QQQ 1-minute config.
#
#   bash scripts/train_qqq_1m_strong.sh --skip-tokenizer
#     Train basemodel only. Requires an existing fine-tuned tokenizer path.
#
#   bash scripts/train_qqq_1m_strong.sh --skip-existing
#     Train missing phases only; skip tokenizer/basemodel if best_model exists.
#
# Extra arguments are passed directly to finetune_csv/train_sequential.py.

cd "$(dirname "$0")/.."

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv. Run: bash init.sh" >&2
  exit 1
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

.venv/bin/python finetune_csv/train_sequential.py \
  --config finetune_csv/configs/config_qqq_1m_strong.yaml \
  "$@"
