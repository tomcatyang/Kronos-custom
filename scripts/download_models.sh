#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x ".venv/bin/huggingface-cli" ]]; then
  echo "Missing .venv/bin/huggingface-cli. Run: uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -r Kronos/requirements.txt -r requirements.txt" >&2
  exit 1
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p models

.venv/bin/huggingface-cli download NeoQuasar/Kronos-Tokenizer-base \
  --local-dir models/Kronos-Tokenizer-base \
  --local-dir-use-symlinks False

.venv/bin/huggingface-cli download NeoQuasar/Kronos-base \
  --local-dir models/Kronos-base \
  --local-dir-use-symlinks False
