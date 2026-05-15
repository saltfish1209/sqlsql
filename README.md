# nl2sql

这是一个面向电力物资采购场景的 Text-to-SQL 系统。当前版本已经重构为 **retrieval-first** 的分层框架，不再依赖“先理解完整 schema 再直接生成 SQL”的旧思路，而是采用：

1. 轻量预处理
2. 候选字段检索
3. 证据实体抽取
4. 结构化 JSON 规划
5. SQL 生成
6. 低成本 Refiner / Selector 修复

## 项目亮点

| 特性 | 来源 | 说明 |
|---|---|---|
| **多路混合 Schema Linking** | 原创 + DeepEye-SQL 启发 | CrossEncoder(A) + ExactMatch(B) + LSH 模糊(C) + Semantic Value(D) 互补召回 |
| **扩展窗口梯队回退** | 原创 | Tier1→Tier2→Tier3 递进扩展，不丢失高相关列 |
| **多路径 SQL 生成** | 原创 + DeepEye-SQL 启发 | ICL / Direct 快路并发，低置信时启用 Plan Path 慢路兜底 |
| **置信感知 SQL 选择** | DeepEye-SQL 启发 | 按执行结果簇计算 confidence，低置信时用稳定优先级裁决 |
| **自动数据库 Profiling** | 论文新增 | 列级统计(NULL率/唯一值/格式/Top值)注入 Prompt |
| **Schema 字段随机化** | 论文新增 | 字段顺序随机化增加生成多样性 |
| **Checker Tool-chain** | DeepEye-SQL 启发 | 执行错误、Literal-Column、空结果、SELECT *、NULL 排序风险检查 |

## 目录结构

```text
nl2sql/
├── config/              # 统一配置中心
│   └── settings.py
│ 
├── pipeline/            # 核心推理管道
│   ├── system.py        # 主编排器
│   ├── schema_linker.py # 三路混合 Schema Linking
│   ├── entity_extractor.py # LLM 实体提取 + 后置过滤
│   ├── generator.py     # ICL / Direct / Plan SQL 生成
│   ├── checkers.py      # DeepEye 风格确定性 SQL 检查器
│   ├── refiner.py       # 执行反馈修正 + Checker Tool-chain
│   ├── selector.py      # 置信感知投票选择
│   ├── profiler.py      # 自动数据库 Profiling
│   ├── db_engine.py     # SQLite 引擎
│   ├── llm_client.py    # LLM 客户端
│   └── utils.py         # 工具函数
│ 
├── training/            # 训练与评估
│   ├── prepare_data.py  # CrossEncoder 数据准备
│   ├── train_cross_encoder.py # CrossEncoder 模型训练
│   ├── evaluate_cross_encoder.py # CrossEncoder 训练效果评估
│   ├── evaluate.py      # 端到端评估
│   ├── evaluate_topk.py # Top-K 召回率评估
│   └── lora_train.py    # LoRA 微调
│ 
├── generation/          # 训练数据生成
│ 
├── scripts/             # 启动脚本
│   ├── start_vllm.sh
│   ├── start_ollama.sh
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

## 配置原则

项目的统一配置入口是 `config/settings.py`。

- 代码中的大多数参数都集中在 `Settings` dataclass 中。
- 与模型服务相关的关键项优先从环境变量读取，例如 `LLM_BASE_URL`、`LLM_MODEL`、`DEBUG_MODE`。
- 如果你先执行 `scripts/set_env.sh` 或 `scripts/set_env.ps1`，再运行 Python 脚本，`settings.py` 会直接读取这些环境变量。
- 如果没有设置对应环境变量，则回退到 `settings.py` 中定义的默认值。

可以把配置理解成两层：

1. 服务层配置：决定连哪个 LLM 服务，例如 `LLM_BASE_URL`、`LLM_MODEL`
2. 管道层配置：决定项目内部推理策略，例如 `top_k_embed`、`entity_max_tokens`、`max_repair_retries`

## settings.py 参数说明

下面按功能分组说明 `config/settings.py` 中主要参数的含义与使用建议。

### 1. 路径与数据文件

- `project_root`：项目根目录。
- `data_dir`：数据目录，默认是 `data/`。
- `models_dir`：本地模型目录，默认是 `models/`。
- `cache_dir`：相似度缓存目录，默认是 `pipeline/similarity_cache/`。
- `csv_path`：主业务 CSV 数据路径，SQLite 会把它加载为查询表。
- `schema_path`：Schema 文本描述文件路径。
- `schema_json_path`：Schema 的 JSON 结构文件路径。
- `qa_template_csv`：问答模板数据路径。
- `train_csv`：CrossEncoder 训练数据 CSV 路径。
- `table_name`：导入 SQLite 后使用的表名，默认 `procurement_table`。

### 2. 本地模型路径

- `embed_model`：Embedding 模型目录，对应环境变量 `EMBED_MODEL_PATH`。
- `reranker_base_model`：CrossEncoder 基座模型目录，对应环境变量 `RERANKER_BASE_MODEL_PATH`。
- `cross_encoder_model`：领域微调后的精排模型目录，对应环境变量 `SCHEMA_PRUNER_MODEL_PATH`。

如果尚未训练自己的 Schema Pruner，可以把 `SCHEMA_PRUNER_MODEL_PATH` 指向 `jina-reranker-v3` 基座模型先跑通流程。

### 3. Schema Linking 参数

- `top_k_embed`：Embedding 初筛保留的列数，值越大召回越强，但后续重排成本更高。
- `lsh_threshold`：LSH 模糊匹配阈值，越高越严格。
- `lsh_num_perm`：LSH 的哈希排列数，越大越稳定，但构建更慢。
- `c_secondary_seq_ratio`：C 路二级校验的序列相似度阈值。
- `c_secondary_jaccard`：C 路二级校验的 Jaccard 阈值。
- `c_secondary_seq_with_jac`：结合 Jaccard 时的较宽松序列阈值。
- `c_query_cover`：问题文本覆盖率阈值，用于控制 C 路候选质量。

如果你的字段很多但召回不足，可以先尝试略微提高 `top_k_embed` 或降低 `lsh_threshold`。

### 4. Embedding 参数

- `embed_query_prompt`：给 Embedding 模型的检索前缀。用于把自然语言问题映射到更适合检索历史模板的表示空间。

通常不建议随意删除这段前缀，除非你已经重新验证过召回效果。

### 5. Schema Linking 参数

- `enable_semantic_value_retrieval`：是否启用 D 路语义值检索，对应环境变量 `ENABLE_SEMANTIC_VALUE_RETRIEVAL`。启用后会构建独立的 `semantic_value_index.pkl` 缓存。
- `semantic_value_top_k`：每个实体最多返回多少个语义相似数据库值，对应环境变量 `SEMANTIC_VALUE_TOP_K`。
- `semantic_value_threshold`：语义值命中的最低余弦相似度，对应环境变量 `SEMANTIC_VALUE_THRESHOLD`。
- `semantic_value_max_values_per_column`：每列最多编码多少个唯一值，对应环境变量 `SEMANTIC_VALUE_MAX_VALUES_PER_COLUMN`，用于控制缓存体积和首次构建耗时。

D 路只对 B 路未精确命中的实体启用，因此不会替代 B 路；B 路仍是最高精度的 exact value linking，D 路主要补足简称、别名和语义相似值召回。

### 6. SQL Generator 参数

- `num_sql_per_path`：每条生成路径产出多少条 SQL，当前默认每路 1 条。
- `icl_temperature`：ICL 路径温度。
- `icl_few_shot_k`：ICL 按「问题模版」语义检索的示例对数 k（默认 3），对应环境变量 `ICL_FEW_SHOT_K`；Prompt 中以「类似问题 1…k」「目标回答字段 1…k」与 `回答模版` 成对展示。
- `direct_temperature`：Direct 路径温度，与 ICL 拉开差距以保留候选多样性。
- `max_gen_tokens`：SQL 生成最大长度，对应环境变量 `LLM_MAX_GEN_TOKENS`。
- `llm_request_timeout_sec`：LLM 请求超时秒数，对应环境变量 `LLM_REQUEST_TIMEOUT_SEC`。主要用于启用了超时控制的调用场景。
- `generator_prefix_code_fence`：是否预填 ` ```sql ` 作为 assistant 前缀，强制模型从 SQL 代码块开始输出。默认关闭。

当前项目的约定是：

- 主流水线不再启用 `thinking_path`，默认由 `ICL` / `Direct` 快路生成候选。
- 当快路低置信或结果为空时，`enable_plan_path=True` 会启用多步 `plan_path` 兜底。
- Baseline 的 `--enable-thinking` / `--no-thinking` 仍保留，用于对照实验。
- ICL few-shot 从 `data/train_dataset_template_only.csv` 读取 `问题模版` / `回答模版`，只用非测试集母版建检索池；相似度只基于 `问题模版` 字段计算。
- 默认切分按 `问题模版` 母版边界执行，比例为 `train_split=0.8`、`val_split=0.0`、`test_split=0.2`，避免 5 条扩展问题跨 train/test 泄漏。

### 7. Entity Extraction 参数

- `entity_max_tokens`：实体抽取阶段最大生成长度，对应环境变量 `ENTITY_MAX_TOKENS`。
- `entity_use_guided_json`：是否通过 `extra_body.guided_json` 强制模型输出 JSON 数组，对应环境变量 `ENTITY_USE_GUIDED_JSON`。
- `entity_prefix_bracket`：是否手动给 assistant 预填 `[` 作为数组前缀，对应环境变量 `ENTITY_PREFIX_BRACKET`。默认关闭，因为在 `vLLM + Qwen + guided_json` 组合下可能出现 `]]` 等尾部污染。
- `enable_thinking_for_entity`：实体提取是否开启思考模式，对应环境变量 `ENTITY_ENABLE_THINKING`。默认关闭。

建议：

- 优先开启 `entity_use_guided_json=True`
- 一般不要同时开启 `entity_prefix_bracket=True`
- 若实体抽取速度慢或输出不稳定，优先保持 `enable_thinking_for_entity=False`

### 8. Refiner 参数

- `max_repair_retries`：每条 SQL 的最大修复轮数。
- `refiner_temperature`：修复阶段温度，默认极低以减少发散。
- `refiner_max_tokens`：Refiner 最大生成长度，对应环境变量 `REFINER_MAX_TOKENS`。
- `refiner_enforce_timeout`：是否对 Refiner 强制超时，对应环境变量 `REFINER_ENFORCE_TIMEOUT`。默认关闭。
- `enable_thinking_for_refiner`：是否在 Refiner 中启用思考模式，对应环境变量 `REFINER_ENABLE_THINKING`。默认关闭。

说明：

- 当前实现中，Refiner 默认不强制超时，而是主要通过 `refiner_max_tokens` 控制输出长度。
- 如果模型出现长时间卡住不返回，可以临时打开 `REFINER_ENFORCE_TIMEOUT=True`。

### 9. Profiler 参数

- `profile_sample_rows`：数据库 Profiling 时每列采样的最大行数。
- `profile_distinct_threshold`：用于区分低基数 / 高基数字段的阈值。

如果数据表很大，而你只想快速预览 Profiling 结果，可以适当降低 `profile_sample_rows`。

### 10. 调试与训练参数

- `debug_mode`：是否开启调试日志，对应环境变量 `DEBUG_MODE`。
- `train_split` / `val_split` / `test_split`：训练、验证、测试集切分比例。
- `random_state`：数据切分随机种子。

## Debug 模式说明

项目中的中间日志统一通过 `pipeline/utils.py` 中的 `debug_print()` 控制：

```python
def debug_print(*args, **kwargs):
    """仅在 DEBUG_MODE=True 时输出，用于中间步骤日志。"""
    from config.settings import settings
    if settings.debug_mode:
        print(*args, **kwargs)
```

这意味着：

- `DEBUG_MODE=True` 时，会输出 Schema Linking、Entity、Generator、Refiner 等中间日志
- `DEBUG_MODE=False` 时，正常情况下只保留必要输出

### Bash 开启 / 关闭调试

```bash
export DEBUG_MODE=True
python -m pipeline.system

export DEBUG_MODE=False
python -m pipeline.system
```

也可以一次性写在命令前：

```bash
DEBUG_MODE=True python training/evaluate.py
DEBUG_MODE=False python training/evaluate.py
```

### PowerShell 开启 / 关闭调试

```powershell
$env:DEBUG_MODE="True"
python -m pipeline.system

$env:DEBUG_MODE="False"
python -m pipeline.system
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

脚本 `scripts/start_ollama.sh` 会自动完成：

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
export OLLAMA_PORT=11434
export OLLAMA_MODEL_TAG=gemma-4-26b-a4b-it-q4
source scripts/set_env.sh
python -m pipeline.system
```

建议显式指定 `OLLAMA_PORT`。以上示例中，`scripts/set_env.sh` 会设置：

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
export OLLAMA_PORT=11434
export OLLAMA_MODEL_TAG=your-model-tag
source scripts/set_env.sh
python -m pipeline.system
```

### 6. 常用 Ollama 参数

`scripts/start_ollama.sh` 支持这些可调参数：

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

此时 `scripts/set_env.sh` 会把环境变量切回 `vLLM` 对应配置。

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

| 组件 | 模型 | 用途 |
|---|---|---|
| **Embedding (Bi-Encoder)** | `harrier-oss-v1-0.6b` | Schema 列描述语义嵌入、ICL 模板检索 |
| **Reranker (Cross-Encoder)** | `jina-reranker-v3` | Schema Linking A 路精排基座 |
| **Schema Pruner** | `my_schema_pruner_model` | 基于 jina-reranker-v3 微调的领域精排模型 |
| **SQL 生成 LLM** | 开源 LLM (vLLM / Ollama) | 多路径 SQL 生成、实体提取、修正 |

### Reranker 微调流程

```text
jina-reranker-v3 (通用基座) → train_cross_encoder.py 领域适配 → my_schema_pruner_model (推理使用)
```

1. `prepare_data.py` 从标注数据生成训练集
2. `train_cross_encoder.py` 基于 `jina-reranker-v3` 做领域适配微调，输出至 `models/my_schema_pruner_model`
3. 推理时 `schema_linker.py` 自动加载微调后的 `my_schema_pruner_model`
4. 若暂无训练数据，可将 `SCHEMA_PRUNER_MODEL_PATH` 直接指向 `models/jina-reranker-v3` 使用开箱即用能力

## 环境变量

### 核心环境变量

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
| `ENABLE_SEMANTIC_VALUE_RETRIEVAL` | 是否启用 D 路语义值检索 | `True` |
| `SEMANTIC_VALUE_TOP_K` | 每个实体返回的语义相似值数量 | `2` |
| `SEMANTIC_VALUE_THRESHOLD` | D 路语义值最低相似度 | `0.62` |
| `SEMANTIC_VALUE_MAX_VALUES_PER_COLUMN` | 每列最多编码的唯一值数量 | `2000` |
| `LLM_MAX_GEN_TOKENS` | SQL 生成最大长度 | `1024` |
| `LLM_REQUEST_TIMEOUT_SEC` | 请求超时秒数 | `180` |
| `ENABLE_PLAN_PATH` | 是否启用多步 plan 慢路 | `True` |
| `ENTITY_MAX_TOKENS` | 实体提取最大长度 | `256` |
| `ENTITY_USE_GUIDED_JSON` | 实体提取是否启用 guided JSON | `True` |
| `ENTITY_PREFIX_BRACKET` | 实体提取是否预填 `[` 前缀 | `False` |
| `ENTITY_ENABLE_THINKING` | 实体提取是否启用 thinking | `False` |
| `REFINER_MAX_TOKENS` | Refiner 最大长度 | `2048` |
| `REFINER_ENFORCE_TIMEOUT` | Refiner 是否强制超时 | `False` |
| `REFINER_ENABLE_THINKING` | Refiner 是否启用 thinking | `False` |

### Ollama 相关环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `OLLAMA_HOST_BIND` | Ollama 监听地址 | `127.0.0.1` |
| `OLLAMA_PORT` | Ollama 端口 | 建议显式设置 |
| `OLLAMA_MODEL_TAG` | Ollama 注册模型名 | `gemma-4-26b-a4b-it-q4` |
| `OLLAMA_MODEL_PATH` | 本地 GGUF 文件路径 | 空 |
| `OLLAMA_CONTEXT_LENGTH` | 上下文长度 | `4096` |
| `OLLAMA_GPU_LAYERS` | GPU 层数 | `-1` |
| `OLLAMA_NUM_PREDICT` | 默认生成长度 | `512` |
| `OLLAMA_TEMPERATURE` | 生成温度 | `0` |

说明：当前 `set_env.sh` 与 `set_env.ps1` 对 `OLLAMA_PORT` 的默认值并不完全一致，因此更推荐你在切换到 Ollama 时显式设置 `OLLAMA_PORT`，避免不同 shell 下端口不一致。

### vLLM 相关环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `VLLM_MODEL_PATH` | vLLM 模型目录 | `models/Qwen3.5-9B` |
| `VLLM_PORT` | vLLM 端口 | `8000` |
| `VLLM_HOST` | vLLM 监听地址 | `0.0.0.0` |
| `VLLM_GPU_MEMORY_UTILIZATION` | GPU 显存利用率 | `0.72` |
| `VLLM_MAX_NUM_SEQS` | 最大并发序列数 | `6` |
| `VLLM_MAX_MODEL_LEN` | 最大上下文长度 | `8192` |
