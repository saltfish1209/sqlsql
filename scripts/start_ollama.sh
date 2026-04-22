#!/usr/bin/env bash
set -euo pipefail

# 启动/配置 Ollama，并将本地 GGUF 注册为可调用模型
# 用法: bash scripts/start_ollama.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OLLAMA_HOST_BIND="${OLLAMA_HOST_BIND:-127.0.0.1}"
OLLAMA_PORT="${OLLAMA_PORT:-21434}"
OLLAMA_MODEL_TAG="${OLLAMA_MODEL_TAG:-gemma-4-31B-it-GGUF}"
OLLAMA_MODEL_PATH="${OLLAMA_MODEL_PATH:-$PROJECT_ROOT/models/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf}"
OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
OLLAMA_GPU_LAYERS="${OLLAMA_GPU_LAYERS:--1}"
OLLAMA_NUM_PREDICT="${OLLAMA_NUM_PREDICT:-2048}"
OLLAMA_TEMPERATURE="${OLLAMA_TEMPERATURE:-0}"
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/ollama/bin/ollama}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/ollama/models}"

GENERATED_DIR="$PROJECT_ROOT/.ollama"
MODEFILE_PATH="$GENERATED_DIR/Modelfile"

export OLLAMA_HOST="${OLLAMA_HOST_BIND}:${OLLAMA_PORT}"
export OLLAMA_MODELS

if [[ -x "$OLLAMA_BIN" ]]; then
  OLLAMA_CMD="$OLLAMA_BIN"
elif command -v ollama >/dev/null 2>&1; then
  OLLAMA_CMD="$(command -v ollama)"
else
  echo "[ERROR] 未找到 ollama。请先上传并安装到 ~/ollama/bin/ollama，或将其加入 PATH。" >&2
  exit 1
fi

mkdir -p "$GENERATED_DIR"

if [[ -n "$OLLAMA_MODEL_PATH" ]]; then
  if [[ ! -f "$OLLAMA_MODEL_PATH" ]]; then
    echo "[ERROR] GGUF 文件不存在: $OLLAMA_MODEL_PATH" >&2
    exit 1
  fi

  cat > "$MODEFILE_PATH" <<EOF
FROM $OLLAMA_MODEL_PATH

PARAMETER num_ctx $OLLAMA_CONTEXT_LENGTH
PARAMETER num_gpu $OLLAMA_GPU_LAYERS
PARAMETER num_predict $OLLAMA_NUM_PREDICT
PARAMETER temperature $OLLAMA_TEMPERATURE
EOF

  echo "创建/更新 Ollama 模型:"
  echo "  model tag  = $OLLAMA_MODEL_TAG"
  echo "  gguf file  = $OLLAMA_MODEL_PATH"
  echo "  modelfile  = $MODEFILE_PATH"
  "$OLLAMA_CMD" create "$OLLAMA_MODEL_TAG" -f "$MODEFILE_PATH"
else
  echo "[WARN] 未设置 OLLAMA_MODEL_PATH，跳过 GGUF 注册，直接使用已有模型: $OLLAMA_MODEL_TAG"
fi

export LLM_PROVIDER="ollama"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:${OLLAMA_PORT}/v1}"
export LLM_MODEL="${LLM_MODEL:-$OLLAMA_MODEL_TAG}"
export LLM_API_KEY="${LLM_API_KEY:-ollama}"

echo ""
echo "Ollama 配置完成:"
echo "  OLLAMA_HOST  = $OLLAMA_HOST"
echo "  OLLAMA_MODELS= $OLLAMA_MODELS"
echo "  OLLAMA_BIN   = $OLLAMA_CMD"
echo "  LLM_BASE_URL = $LLM_BASE_URL"
echo "  LLM_MODEL    = $LLM_MODEL"
echo "  LLM_API_KEY  = $LLM_API_KEY"
echo ""
echo "如需当前终端启动服务，执行:"
echo "  \"$OLLAMA_CMD\" serve"
echo ""
echo "如需在另一个终端加载环境变量，执行:"
echo "  export LLM_PROVIDER=ollama"
echo "  source scripts/set_env.sh"
