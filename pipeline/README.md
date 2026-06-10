# Pipeline 总体流程框架说明

本文档描述 `pipeline/` 目录下 Text-to-SQL 推理系统的整体运作方式，重点覆盖：

- 端到端执行流程
- 数据在各模块间的传递形式
- 模块之间的依赖关系与调用边界
- 失败场景下的回退与修复机制

---

## 1. 系统定位与入口

`pipeline/system.py` 中的 `TextToSQLSystem` 是主编排器，负责把一次自然语言问题转化为 SQL 执行结果。

系统核心策略是 **retrieval-first**：

1. 先做字段候选召回与证据实体抽取
2. 再构造结构化 plan JSON
3. 最后由 SQL 生成、检查、修复、选择组成闭环

运行入口示例：

```bash
python -m pipeline.system
```

---

## 2. 端到端流程总览

```
用户问题
  -> QuestionSplitter (可选拆分多子问题)
  -> SchemaLinker#1 (无实体初召回 Top20)
  -> EntityExtractor (抽取证据实体)
  -> SchemaLinker#2 (带实体二次召回 + 必须列补全)
  -> System 组装 final_schema + plan_json
  -> SQLGenerator (direct / icl / plan 并发生成)
  -> SQLRefiner + SQLCheckerChain (执行检查与修复)
  -> SQLSelector (按 confidence 与状态选最佳)
  -> DBEngine 执行结果整理去重
  -> 返回统一结果字典
```

若问题被拆成多个子问题，系统会对每个子问题独立执行上述单问题流程，最后在 `system.py` 中聚合。

---

## 3. 分层数据流（输入/输出）

### Layer 0：输入归一化与问题拆分

- 模块：`system.py` + `question_splitter.py`
- 输入：原始 `question: str`
- 输出：
  - 单问题：`[question]`
  - 多问题：`["子问题1", "子问题2", ...]`

关键点：

- `to_halfwidth()` 做全角半角统一
- `QuestionSplitter` 使用 LLM + guided JSON，输出结构：
  - `{"是否多问题": bool, "子问题": list[str]}`

---

### Layer 1：Schema 初次召回（无实体）

- 模块：`schema_linker.py`
- 输入：`question + extracted_entities=[]`
- 输出：`CandidateSchemaPack`（含 `Top20候选`、`证据详情`、`精简schema` 等）

关键点：

- CrossEncoder 对问题-字段 passage 做全量重排（A 路）
- 先得到语义相关的初始字段候选，为后续实体抽取提供上下文

---

### Layer 2：证据实体抽取

- 模块：`entity_extractor.py`
- 输入：
  - `question`
  - `{"召回schema": Top20候选}`
- 输出：`entities: list[str]`

关键点：

- LLM 仅输出 JSON（`{"提取实体":[...]}`）
- 后置过滤保证实体确实来自原问题、去重、排除字段名污染

---

### Layer 3：Schema 二次召回（带实体）

- 模块：`schema_linker.py`
- 输入：`question + entities`
- 输出：增强版 `CandidateSchemaPack`

新增信息：

- `必须列集合`：由实体精确/模糊/向量匹配命中的列
- `证据详情`：按“精确匹配/模糊匹配/向量匹配”分类的命中证据
- `精简schema`：在 TopK 基础上合并必须列后的供生成字段集

---

### Layer 4：计划数据结构组装

- 模块：`system.py`
- 输入：二次召回结果 + 实体
- 输出：`plan_json` + `schema_prompt`

`plan_json` 主要字段：

- `用户问题`
- `证据实体`
- `初始top20`
- `must_have`
- `证据详情`
- `精简schema`
- `最终schema`

`schema_prompt` 由 `generator.py` 构造，融合字段描述与 profiler 统计（类型/空值率/示例值/范围等）。

---

### Layer 5：多路径 SQL 生成

- 模块：`generator.py` + `fewshot_index.py`
- 输入：`question + schema_prompt + plan_json`
- 输出：候选 SQL（当前主路径返回 1 条最佳候选）

并发路径：

- `direct`：直接生成
- `icl`：few-shot 检索增强后生成
- `plan`：先规划再生成

关键点：

- Few-shot 由 FAISS 索引检索相似模板
- 生成后统一通过 `extract_sql()` 清洗为可执行 SQL 文本

---

### Layer 6：执行校验与自动修复

- 模块：`refiner.py` + `checkers.py` + `db_engine.py`
- 输入：候选 SQL + schema 信息 + plan_json
- 输出：带状态的候选列表（`success` / `needs_repair`）

检查器覆盖问题：

- 执行错误（语法/列名）
- Literal-Column 不一致
- 空结果风险
- `SELECT *`
- `ORDER BY` 缺少 NULL guard

修复方式：

- 若命中检查问题，Refiner 基于错误信息和 plan_json 再生成 SQL
- 再执行一次检查并更新状态

---

### Layer 7：候选选择与最终输出

- 模块：`selector.py` + `system.py`
- 输入：Refiner 处理后的候选
- 输出：统一响应字典

选择策略：

- 只在 `status == "success"` 的候选中选择
- 按 `confidence` 排序
- 若低于阈值返回 `low_confidence` 状态

输出核心字段：

- `final_sql`
- `execution_result`
- `reason`
- `token_usage`
- `证据实体`
- `候选字段包`
- `sql_generation_spec`
- `is_multi_question`

---

## 4. 模块依赖关系（调用视角）

`TextToSQLSystem` 初始化时注入并持有以下组件：

- `DBEngine`：CSV -> SQLite 内存表、SQL 执行与 literal 校验
- `SchemaLinker`：候选字段召回与证据对齐
- `EntityExtractor`：证据实体抽取
- `SQLGenerator`：多路径 SQL 生成
- `SQLRefiner`：基于 checker 的修复
- `SQLSelector`：最终候选选择
- `DatabaseProfiler`：列统计（由 SchemaLinker / Generator 使用）
- `QuestionSplitter`：多问题拆分

关系特征：

- **编排集中**：只有 `system.py` 做跨模块流程控制
- **模块单责**：每个模块聚焦一个阶段，不互相承担编排职责
- **数据显式传递**：通过 `plan_json`、`CandidateSchemaPack`、`schema_prompt` 串联上下游

---

## 5. 缓存与中间产物

缓存目录：`pipeline/similarity_cache/`

主要缓存：

- `schema_linker/exact_index.pkl`
- `schema_linker/lsh_index.pkl`
- `schema_linker/semantic_value_index.pkl`
- `schema_linker/faiss_index.bin` 与 `faiss_meta.pkl`
- `value_indexes/fewshot_index/*`（few-shot 检索索引）

作用：

- 减少每次启动时的索引重建耗时
- 保持 retrieval 阶段的稳定性与可复用性

---

## 6. 失败处理与降级策略

- 问题拆分失败：退化为单问题
- 实体抽取失败：返回空实体，仍继续流程
- few-shot 索引缺失：
  - 可自动构建（配置开启时）
  - 或降级为无 few-shot 召回
- SQL 生成失败：返回 `generation_failed`
- Refiner 修复失败：保持原候选并进入 Selector
- 所有候选失败：返回 `failed` 原因

系统设计目标是“**可继续执行优先**”，避免单点失败导致整条链路中断。

---

## 7. 可扩展点（建议）

- 在 `SchemaLinker.retrieve()` 增加新召回路由（如业务词典路由）
- 在 `SQLCheckerChain.check()` 增加领域规则检查器
- 在 `SQLSelector.select_best()` 增加执行结果簇投票或 rerank 打分
- 在 `TextToSQLSystem` 中调整流程开关（如拆分策略、Plan 路使用策略）

保持原则：新增能力尽量作为独立模块，通过 `system.py` 注入，而不是在现有模块中横向耦合。

