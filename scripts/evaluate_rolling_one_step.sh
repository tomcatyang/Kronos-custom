#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/evaluate_rolling_one_step.sh
#     Run rolling one-step evaluation for the QQQ 5-minute model.
#
#   bash scripts/evaluate_rolling_one_step.sh --config finetune_csv/configs/config_qqq_1m_cpu.yaml
#     Run the same evaluation for another config.
#
# Extra arguments are passed directly to finetune_csv/evaluate_rolling_one_step.py.

cd "$(dirname "$0")/.."

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv. Run: bash init.sh" >&2
  exit 1
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

args=(
  --config finetune_csv/configs/config_qqq_5m_cpu.yaml
  --windows 50
  --calibration-windows 10
  --calibration-metric signed_return
  --device cpu
)

.venv/bin/python finetune_csv/evaluate_rolling_one_step.py "${args[@]}" "$@"
