#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x ".venv/bin/huggingface-cli" ]]; then
  echo "Missing .venv/bin/huggingface-cli. Run: uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -r Kronos/requirements.txt -r requirements.txt" >&2
  exit 1
fi

# 默认直连官方站。实测本机直连 huggingface.co 正常，而 hf-mirror.com 当前会把
# 请求 308 重定向回官方站，huggingface-cli 不跟随跨域重定向，反而导致 FileMetadataError。
# 如确实需要走镜像，运行前设置 USE_HF_MIRROR=1。
if [[ "${USE_HF_MIRROR:-0}" == "1" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
else
  export HF_ENDPOINT="https://huggingface.co"
fi

# 拉长单文件请求超时，避免网络抖动时元数据 HEAD 请求过早失败。
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
# 关闭 hf_transfer，使用更稳定（可重试/断点续传）的默认下载器。
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "Using HF_ENDPOINT=$HF_ENDPOINT"

mkdir -p models

.venv/bin/huggingface-cli download NeoQuasar/Kronos-Tokenizer-base \
  --local-dir models/Kronos-Tokenizer-base

.venv/bin/huggingface-cli download NeoQuasar/Kronos-base \
  --local-dir models/Kronos-base
