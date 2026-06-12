#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash custom/train_qqq_lora.sh
#     Train the QQQ LoRA adapter with custom/configs/qqq_lora.yaml.
#
#   bash custom/train_qqq_lora.sh --dry-run
#     Run one training step and one validation step without saving an adapter.
#
#   CONFIG_PATH=custom/configs/qqq_lora.yaml bash custom/train_qqq_lora.sh
#     Use a custom config path.
#
# Extra arguments are passed directly to custom/train_qqq_lora.py.

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-custom/configs/qqq_lora.yaml}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  echo "Run: bash init.sh" >&2
  exit 1
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" custom/train_qqq_lora.py \
  --config "$CONFIG_PATH" \
  "$@"

