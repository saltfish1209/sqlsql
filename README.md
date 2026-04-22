# question_to_sql

基于 XiYan-SQL 架构的电力物资采购 Text-to-SQL 系统，融合 BIRD 榜首论文（Automatic Metadata Extraction）的核心思想进行增强。

## 项目亮点

| 特性 | 来源 | 说明 |
|---|---|---|
| **三路混合 Schema Linking** | 原创 | CrossEncoder(A) + ExactMatch(B) + LSH 模糊(C) 互补召回 |
| **扩展窗口梯队回退** | 原创 | Tier1→Tier2→Tier3 递进扩展，不丢失高相关列 |
| **三路并发 SQL 生成** | 原创 | Thinking / ICL / Direct 三条路径增加候选多样性 |
| **自洽性投票 + 优先级平票** | 原创 | Self-Consistency 选择 + thinking > ICL > direct 优先级 |
| **自动数据库 Profiling** | 论文新增 | 列级统计(NULL率/唯一值/格式/Top值)注入 Prompt |
| **Schema 字段随机化** | 论文新增 | 字段顺序随机化增加生成多样性 |
| **Literal-Column 校验** | 论文新增 | WHERE 字面量验证是否存在于对应列 |

## 目录结构

```
question_to_sql/
├── config/              # 统一配置中心
│   └── settings.py
│ 
├── pipeline/            # 核心推理管道
│   ├── system.py        # 主编排器
│   ├── schema_linker.py # 三路混合 Schema Linking
│   ├── entity_extractor.py # LLM+规则实体提取
│   ├── generator.py     # 三路 SQL 生成
│   ├── refiner.py       # 执行反馈修正 + Literal校验
│   ├── selector.py      # 自洽性投票选择
│   ├── profiler.py      # 自动数据库 Profiling
│   ├── db_engine.py     # SQLite 引擎
│   ├── llm_client.py    # LLM 客户端
│   └── utils.py         # 工具函数
│ 
├── training/            # 训练与评估
│   ├── prepare_data.py  # CrossEncoder 数据准备
│   ├── train_cross_encoder.py # CrossEncoder 模型训练
│   ├──evaluate_cross_encoder.py # CrossEncoder 训练效果评估
│   ├── evaluate.py      # 端到端评估
│   ├── evaluate_topk.py # Top-K 召回率评估
│   └── lora_train.py    # LoRA 微调
│ 
├── generation/          # 训练数据生成
│ 
├── scripts/             # 启动脚本
│   ├── start_vllm.sh
│   ├── set_env.sh
│   ├── set_env.ps1
│   └── profile_db.py
│ 
├── data/                # 数据目录
│ 
└── models/              # 模型目录
    ├── harrier-oss-v1-0.6b/    # Embedding 模型
    ├── jina-reranker-v3/       # Reranker 基座模型
    └── my_schema_pruner_model/ # 微调后的精排模型
```

## 快速开始

### 1. 启动 vLLM 服务
```bash
bash scripts/start_vllm.sh
```

### 2. 设置环境变量（另一个终端）
```bash
source scripts/set_env.sh
```

### 3. 运行推理
```bash
python -m pipeline.system
```

### 4. 运行评估
```bash
# 详细模式
DEBUG_MODE=True python training/evaluate.py

# 简洁模式
DEBUG_MODE=False python training/evaluate.py
```

### 5. 运行 Profiler 预览
```bash
python scripts/profile_db.py
```

## 使用 Ollama 运行 GGUF

当前项目的 Python 主流程无需区分 `vLLM` 或 `Ollama`，因为统一走 OpenAI 兼容接口。使用 GGUF 时，推荐通过 `Ollama` 承载模型服务。

### 1. 服务器安装 Ollama

先确认服务器已安装 `ollama`：

```bash
ollama --version
```

若未安装，请先完成 `ollama` 安装，再继续下面步骤。

### 2. 注册本地 GGUF 模型

假设你已经把 GGUF 下载到服务器，例如：

```bash
/home/your_user/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
```

执行：

```bash
export OLLAMA_MODEL_PATH=/home/your_user/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
export OLLAMA_MODEL_TAG=gemma-4-26b-a4b-it-q4
bash scripts/start_ollama.sh
```

脚本 [scripts/start_ollama.sh](C:/Users/yinjun/Desktop/nl2sql/scripts/start_ollama.sh:1) 会自动完成：

1. 基于本地 `.gguf` 生成 `Modelfile`
2. 执行 `ollama create`
3. 将该模型注册为 `OLLAMA_MODEL_TAG` 指定的名字

### 3. 启动 Ollama 服务

新开一个终端执行：

```bash
ollama serve
```

默认监听 `11434` 端口，项目会通过 OpenAI 兼容接口访问它。

### 4. 在项目中切到 Ollama

再开一个终端执行：

```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL_TAG=gemma-4-26b-a4b-it-q4
source scripts/set_env.sh
python -m pipeline.system
```

此时 [scripts/set_env.sh](C:/Users/yinjun/Desktop/nl2sql/scripts/set_env.sh:8) 会自动设置：

```bash
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=gemma-4-26b-a4b-it-q4
LLM_API_KEY=ollama
```

### 5. 首次使用与后续使用的区别

首次使用某个 GGUF 模型时：

```bash
export OLLAMA_MODEL_PATH=/home/your_user/models/your-model.gguf
export OLLAMA_MODEL_TAG=your-model-tag
bash scripts/start_ollama.sh
```

后续如果模型已经注册过，就不需要重复 `ollama create`，只需：

```bash
ollama serve
export LLM_PROVIDER=ollama
export OLLAMA_MODEL_TAG=your-model-tag
source scripts/set_env.sh
python -m pipeline.system
```

### 6. 常用 Ollama 参数

[scripts/start_ollama.sh](C:/Users/yinjun/Desktop/nl2sql/scripts/start_ollama.sh:10) 支持这些可调参数：

```bash
export OLLAMA_HOST_BIND=0.0.0.0
export OLLAMA_PORT=11434
export OLLAMA_MODEL_TAG=gemma-4-26b-a4b-it-q4
export OLLAMA_MODEL_PATH=/home/your_user/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_GPU_LAYERS=-1
export OLLAMA_NUM_PREDICT=512
export OLLAMA_TEMPERATURE=0
```

其中：

- `OLLAMA_MODEL_TAG`：注册后的模型名，也是项目调用时使用的模型名
- `OLLAMA_MODEL_PATH`：本地 GGUF 文件路径
- `OLLAMA_CONTEXT_LENGTH`：上下文长度
- `OLLAMA_GPU_LAYERS=-1`：尽量把可用层放到 GPU
- `OLLAMA_NUM_PREDICT`：默认生成长度
- `OLLAMA_TEMPERATURE`：生成温度

## 如何切换回 vLLM

如果你仍然想使用原有的 `vLLM` 路线，只需恢复到 `vLLM` 模式即可。

### 1. 启动 vLLM

```bash
bash scripts/start_vllm.sh
```

### 2. 切回 vLLM 环境

```bash
export LLM_PROVIDER=vllm
source scripts/set_env.sh
python -m pipeline.system
```

此时 [scripts/set_env.sh](C:/Users/yinjun/Desktop/nl2sql/scripts/set_env.sh:18) 会把环境变量切回 `vLLM` 对应配置。

## 更换其他模型时需要改什么

原则很简单：项目侧只认 `LLM_MODEL` 和 `LLM_BASE_URL`，所以更换模型时通常只改服务层配置，不改 Python 主流程。

### 1. 换另一个 GGUF 模型

例如你想切到另一个 GGUF：

```bash
export OLLAMA_MODEL_PATH=/home/your_user/models/another-model.gguf
export OLLAMA_MODEL_TAG=another-model-q4
bash scripts/start_ollama.sh
```

然后运行项目：

```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL_TAG=another-model-q4
source scripts/set_env.sh
python -m pipeline.system
```

你真正需要改的通常只有两项：

- `OLLAMA_MODEL_PATH`
- `OLLAMA_MODEL_TAG`

### 2. 换回其他 vLLM 模型

如果是非 GGUF、继续走 `vLLM`，通常改：

```bash
export VLLM_MODEL_PATH=/path/to/your/hf-model
export LLM_MODEL=your-served-model-name
bash scripts/start_vllm.sh
```

然后：

```bash
export LLM_PROVIDER=vllm
source scripts/set_env.sh
python -m pipeline.system
```


## 模型架构

| 组件 | 模型                       | 用途 |
|---|--------------------------|---|
| **Embedding (Bi-Encoder)** | `harrier-oss-v1-0.6b`    | Schema 列描述语义嵌入、ICL 模板检索 |
| **Reranker (Cross-Encoder)** | `jina-reranker-v3`       | Schema Linking A路精排基座 |
| **Schema Pruner** | `my_schema_pruner_model` | 基于 jina-reranker-v3 微调的领域精排模型 |
| **SQL 生成 LLM** | 开源LLM (vLLM / Ollama)   | 三路 SQL 生成、实体提取、修正 |

### Reranker 微调流程

```
jina-reranker-v3 (通用基座) → train_cross_encoder.py 领域适配 → my_schema_pruner_model (推理使用)
```

1. `prepare_data.py` 从标注数据生成训练集
2. `train_cross_encoder.py` 基于 `jina-reranker-v3` 做领域适配微调，输出至 `models/my_schema_pruner_model`
3. 推理时 `schema_linker.py` 自动加载微调后的 `my_schema_pruner_model`
4. 若暂无训练数据，可将 `SCHEMA_PRUNER_MODEL_PATH` 直接指向 `models/jina-reranker-v3` 使用开箱即用能力

## 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_BASE_URL` | LLM 服务地址 | (必填) |
| `LLM_MODEL` | 当前调用的模型名称 | (必填) |
| `LLM_API_KEY` | API Key | `EMPTY` / `ollama` |
| `LLM_PROVIDER` | 服务提供方 | `vllm` |
| `DEBUG_MODE` | 调试输出 | `True` |
| `EMBED_MODEL_PATH` | Embedding 模型路径 | `models/harrier-oss-v1-0.6b` |
| `RERANKER_BASE_MODEL_PATH` | Reranker 基座模型路径 | `models/jina-reranker-v3` |
| `SCHEMA_PRUNER_MODEL_PATH` | 微调后精排模型路径 | `models/my_schema_pruner_model` |

### Ollama 相关环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `OLLAMA_HOST_BIND` | Ollama 监听地址 | `127.0.0.1` |
| `OLLAMA_PORT` | Ollama 端口 | `11434` |
| `OLLAMA_MODEL_TAG` | Ollama 注册模型名 | `gemma-4-26b-a4b-it-q4` |
| `OLLAMA_MODEL_PATH` | 本地 GGUF 文件路径 | 空 |
| `OLLAMA_CONTEXT_LENGTH` | 上下文长度 | `4096` |
| `OLLAMA_GPU_LAYERS` | GPU 层数 | `-1` |
| `OLLAMA_NUM_PREDICT` | 默认生成长度 | `512` |
| `OLLAMA_TEMPERATURE` | 生成温度 | `0` |

### vLLM 相关环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `VLLM_MODEL_PATH` | vLLM 模型目录 | `models/DeepSeek-Coder-V2-Lite-Instruct` |
| `VLLM_PORT` | vLLM 端口 | `8000` |

