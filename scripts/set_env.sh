#!/usr/bin/env bash

# 在运行推理/评估脚本前执行: source scripts/set_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLM_PROVIDER="${LLM_PROVIDER:-vllm}"
VLLM_PORT="${VLLM_PORT:-8000}"
OLLAMA_PORT="${OLLAMA_PORT:-21434}"
OLLAMA_HOST_BIND="${OLLAMA_HOST_BIND:-127.0.0.1}"

if [[ "$LLM_PROVIDER" == "ollama" ]]; then
  PORT="$OLLAMA_PORT"
  DEFAULT_BASE_URL="http://${OLLAMA_HOST_BIND}:${PORT}/v1"
  SERVED="${OLLAMA_MODEL_TAG:-${LLM_MODEL:-gemma-4-31B-it-GGUF}}"
  DEFAULT_API_KEY="ollama"
else
  PORT="$VLLM_PORT"
  DEFAULT_BASE_URL="http://127.0.0.1:${PORT}/v1"
  SERVED="${LLM_MODEL:-DeepSeek-Coder-V2-Lite-Instruct}"
  DEFAULT_API_KEY="EMPTY"
fi

export LLM_BASE_URL="${LLM_BASE_URL:-$DEFAULT_BASE_URL}"
export LLM_MODEL="$SERVED"
export LLM_API_KEY="${LLM_API_KEY:-$DEFAULT_API_KEY}"

echo "已设置 LLM 环境变量（当前 shell 有效）:"
echo "  LLM_PROVIDER = $LLM_PROVIDER"
echo "  LLM_BASE_URL = $LLM_BASE_URL"
echo "  LLM_MODEL    = $LLM_MODEL"
echo "  LLM_API_KEY  = $LLM_API_KEY"
echo "  项目根目录   = $PROJECT_ROOT"
