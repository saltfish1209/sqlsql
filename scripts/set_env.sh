#!/usr/bin/env bash

# 在运行推理/评估脚本前执行: source scripts/set_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PORT="${VLLM_PORT:-8000}"
SERVED="${LLM_MODEL:-DeepSeek-Coder-V2-Lite-Instruct}"

export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
export LLM_MODEL="$SERVED"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"

echo "已设置 LLM 环境变量（当前 shell 有效）:"
echo "  LLM_BASE_URL = $LLM_BASE_URL"
echo "  LLM_MODEL    = $LLM_MODEL"
echo "  LLM_API_KEY  = $LLM_API_KEY"
echo "  项目根目录   = $PROJECT_ROOT"
