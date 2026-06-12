#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash custom/backtest_qqq_lora.sh
#     Backtest the QQQ LoRA adapter on the held-out test split.
#
#   bash custom/backtest_qqq_lora.sh --max-steps 100
#     Run a shorter smoke backtest on the first 100 test bars.
#
# Extra arguments are passed directly to custom/backtest_qqq_lora.py.

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-custom/configs/qqq_lora.yaml}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  echo "Run: bash init.sh" >&2
  exit 1
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" custom/backtest_qqq_lora.py \
  --config "$CONFIG_PATH" \
  --split test \
  "$@"

