#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/evaluate_qqq_1m_strong.sh
#     Evaluate the stronger QQQ 1-minute model on the test split.
#
#   bash scripts/evaluate_qqq_1m_strong.sh --windows 500 --device cuda:0
#     Override evaluation window count and device.
#
# Extra arguments are passed directly to finetune_csv/evaluate_finetuned_model.py.

cd "$(dirname "$0")/.."

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv. Run: bash init.sh" >&2
  exit 1
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

.venv/bin/python finetune_csv/evaluate_finetuned_model.py \
  --config finetune_csv/configs/config_qqq_1m_strong.yaml \
  --windows 200 \
  --device cuda:0 \
  "$@"
