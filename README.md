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
├── training/            # 训练与评估
│   ├── prepare_data.py  # CrossEncoder 数据准备
│   ├── train_cross_encoder.py
│   ├── evaluate.py      # 端到端评估
│   ├── evaluate_topk.py # Top-K 召回率评估
│   └── lora_train.py    # LoRA 微调
├── generation/          # 训练数据生成
├── scripts/             # 启动脚本
│   ├── start_vllm.sh
│   ├── set_env.sh
│   ├── set_env.ps1
│   └── profile_db.py
├── data/                # 数据目录
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

## 模型架构

| 组件 | 模型 | 用途 |
|---|---|---|
| **Embedding (Bi-Encoder)** | `harrier-oss-v1-0.6b` | Schema 列描述语义嵌入、ICL 模板检索 |
| **Reranker (Cross-Encoder)** | `jina-reranker-v3` | Schema Linking A路精排基座 |
| **Schema Pruner** | `my_schema_pruner_model` | 基于 jina-reranker-v3 微调的领域精排模型 |
| **SQL 生成 LLM** | Qwen 系列 (vLLM) | 三路 SQL 生成、实体提取、修正 |

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
| `LLM_BASE_URL` | vLLM 服务地址 | (必填) |
| `LLM_MODEL` | 模型名称 | (必填) |
| `LLM_API_KEY` | API Key | `EMPTY` |
| `DEBUG_MODE` | 调试输出 | `True` |
| `EMBED_MODEL_PATH` | Embedding 模型路径 | `models/harrier-oss-v1-0.6b` |
| `RERANKER_BASE_MODEL_PATH` | Reranker 基座模型路径 | `models/jina-reranker-v3` |
| `SCHEMA_PRUNER_MODEL_PATH` | 微调后精排模型路径 | `models/my_schema_pruner_model` |
