#!/usr/bin/env bash
set -euo pipefail

# 启动 vLLM OpenAI 兼容服务
# 用法: bash scripts/start_vllm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="${VLLM_MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}"
SERVED_NAME="${LLM_MODEL:-Qwen3.5-9B}"
PORT="${VLLM_PORT:-8000}"
HOST_BIND="${VLLM_HOST:-0.0.0.0}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.72}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-6}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[WARN] 模型目录不存在: $MODEL_PATH" >&2
  echo "请设置 VLLM_MODEL_PATH 指向本地模型目录。" >&2
fi

export LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export LLM_MODEL="$SERVED_NAME"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"

echo "启动 vLLM:"
echo "  --model             = $MODEL_PATH"
echo "  --served-model-name = $SERVED_NAME"
echo "  --host              = $HOST_BIND"
echo "  --port              = $PORT"
echo "  --gpu-memory-util   = $GPU_MEM_UTIL"
echo "  --max-num-seqs      = $MAX_NUM_SEQS"
echo "  --max-model-len     = $MAX_MODEL_LEN"
echo ""
echo "在另一个终端执行: source scripts/set_env.sh"
echo ""

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST_BIND" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS"\
  --max-model-len "$MAX_MODEL_LEN"