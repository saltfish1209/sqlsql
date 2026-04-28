"""
统一配置中心 —— 所有可调超参数、路径、模型名称均在此管理。
环境变量优先，缺省使用默认值。
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "pipeline" / "similarity_cache"

_ERR_HINT = (
    "请设置环境变量，或执行脚本: "
    "Bash: source scripts/set_env.sh ; "
    "PowerShell: . .\\scripts\\set_env.ps1"
)


def get_llm_base_url() -> str:
    v = os.getenv("LLM_BASE_URL", "").strip()
    if not v:
        raise RuntimeError(f"未设置 LLM_BASE_URL。{_ERR_HINT}")
    return v.rstrip("/")


def get_llm_api_key() -> str:
    return os.getenv("LLM_API_KEY", "EMPTY").strip() or "EMPTY"


def get_llm_model() -> str:
    v = os.getenv("LLM_MODEL", "").strip()
    if not v:
        raise RuntimeError(f"未设置 LLM_MODEL。{_ERR_HINT}")
    return v


@dataclass
class Settings:
    # ── 路径 ──
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    cache_dir: Path = CACHE_DIR

    csv_path: Path = field(default_factory=lambda: DATA_DIR / "一次二次物料长描述2.csv")
    schema_path: Path = field(default_factory=lambda: DATA_DIR / "m_schema.txt")
    schema_json_path: Path = field(default_factory=lambda: DATA_DIR / "m_schema.json")
    qa_template_csv: Path = field(default_factory=lambda: DATA_DIR / "train_dataset_with_sql_template.csv")
    train_csv: Path = field(default_factory=lambda: DATA_DIR / "train_dataset_with_sql_and_slots.csv")
    table_name: str = "procurement_table"

    # ── 本地模型路径 ──
    embed_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBED_MODEL_PATH",
            str(MODELS_DIR / "harrier-oss-v1-0.6b"),
        )
    )
    reranker_base_model: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_BASE_MODEL_PATH",
            str(MODELS_DIR / "jina-reranker-v3"),
        )
    )
    cross_encoder_model: str = field(
        default_factory=lambda: os.getenv(
            "SCHEMA_PRUNER_MODEL_PATH",
            str(MODELS_DIR / "my_schema_pruner_model"),
        )
    )

    # ── Schema Linking ──
    top_k_embed: int = 6
    lsh_threshold: float = 0.75
    lsh_num_perm: int = 128

    # C路二级校验阈值
    c_secondary_seq_ratio: float = 0.40
    c_secondary_jaccard: float = 0.32
    c_secondary_seq_with_jac: float = 0.26
    c_query_cover: float = 0.88

    # ── Embedding ──
    embed_query_prompt: str = (
        "Instruct: 给定一个关于数据库的自然语言问题，检索语义最相似的历史查询模板\n"
        "Query: "
    )

    # ── Generator ──
    num_sql_per_path: int = 1
    thinking_temperature: float = 0.5
    icl_temperature: float = 0.1
    # 与 ICL 路温度 (0.1) 拉开差距，使 ICL/Direct 两条快路在自洽性投票中保留多样性，
    # 避免 system._try_flow 早停 (两路一致即取消 thinking) 时退化为单路决策。
    direct_temperature: float = 0.4
    # 非思考路径的 max_tokens：SQL 本身很短，1024 足够
    max_gen_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_GEN_TOKENS", "1024"))
    )
    # thinking_path 的 max_tokens：设 0 表示不限（不向 API 传 max_tokens）
    thinking_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("THINKING_MAX_TOKENS", "0"))
    )
    llm_request_timeout_sec: int = field(
        default_factory=lambda: int(os.getenv("LLM_REQUEST_TIMEOUT_SEC", "180"))
    )
    # 三路策略：
    #   thinking_path 始终启用思考 (enable_thinking=True)
    #   ICL / Direct 始终禁用思考 (enable_thinking=False)
    # 通过 extra_body.chat_template_kwargs 传给 vLLM
    enable_thinking_for_entity: bool = field(
        default_factory=lambda: os.getenv("ENTITY_ENABLE_THINKING", "False").lower() == "true"
    )
    enable_thinking_for_refiner: bool = field(
        default_factory=lambda: os.getenv("REFINER_ENABLE_THINKING", "False").lower() == "true"
    )

    # ── Entity Extraction ──
    entity_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("ENTITY_MAX_TOKENS", "256"))
    )
    # vLLM guided decoding：extra_body.guided_json，强制输出 JSON 数组
    entity_use_guided_json: bool = field(
        default_factory=lambda: os.getenv("ENTITY_USE_GUIDED_JSON", "True").lower() == "true"
    )
    # 替模型开头：追加 assistant "[" + continue_final_message。
    # 默认关闭：vLLM + Qwen3 + guided_json 同时启用时会出现 "]]" 等 schema 尾巴污染，
    # 只用 guided_json 已经足够可靠。
    entity_prefix_bracket: bool = field(
        default_factory=lambda: os.getenv("ENTITY_PREFIX_BRACKET", "False").lower() == "true"
    )

    # ── Refiner ──
    max_repair_retries: int = 2
    refiner_temperature: float = 0.01
    # Refiner 同样需要给 Qwen3 的 CoT 留空间
    refiner_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("REFINER_MAX_TOKENS", "2048"))
    )
    # Refiner 不强制时间截停：max_tokens 已能自然限制生成长度
    refiner_enforce_timeout: bool = field(
        default_factory=lambda: os.getenv("REFINER_ENFORCE_TIMEOUT", "False").lower() == "true"
    )

    # ── SQL Generator 可选的 assistant 前缀（默认关闭） ──
    # 若开启：以 "```sql\n" 预填，强迫模型从 SQL 字面起笔
    generator_prefix_code_fence: bool = field(
        default_factory=lambda: os.getenv("GEN_PREFIX_CODE_FENCE", "False").lower() == "true"
    )

    # ── Profiler (论文新增) ──
    profile_sample_rows: int = 100
    profile_distinct_threshold: int = 80

    # ── 调试 ──
    debug_mode: bool = field(
        default_factory=lambda: os.getenv("DEBUG_MODE", "True").lower() == "true"
    )

    # ── 训练 ──
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_state: int = 42


settings = Settings()
