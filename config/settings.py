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
    schema_json_path: Path = field(default_factory=lambda: DATA_DIR / "m_schema.json")
    schema_path: Path = field(default_factory=lambda: DATA_DIR / "m_schema.txt")
    qa_template_csv: Path = field(default_factory=lambda: DATA_DIR / "train_dataset_template_only.csv")
    train_csv: Path = field(default_factory=lambda: DATA_DIR / "train_dataset_with_sql_and_slots.csv")
    train_split_jsonl: Path = field(default_factory=lambda: DATA_DIR / "train_split.jsonl")
    val_split_jsonl: Path = field(default_factory=lambda: DATA_DIR / "val_split.jsonl")
    test_split_jsonl: Path = field(default_factory=lambda: DATA_DIR / "test_split.jsonl")
    table_name: str = "procurement_table"
    fewshot_index_dir: Path = field(default_factory=lambda: CACHE_DIR / "fewshot_index")

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
    # ── Retrieval-first schema linking ──
    candidate_value_top_k: int = 3
    # candidate_exact_bonus: float = 0.35
    # candidate_semantic_bonus: float = 0.25
    # candidate_fuzzy_bonus: float = 0.05
    # candidate_min_score: float = 0.18
    # candidate_max_columns: int = 18
    # 断崖法与比例截取参数（支持环境变量覆盖）
    candidate_cliff_protect_ratio: float = field(
        default_factory=lambda: float(os.getenv("CANDIDATE_CLIFF_PROTECT_RATIO", "0.3"))
    )
    candidate_cliff_min_ratio: float = field(
        default_factory=lambda: float(os.getenv("CANDIDATE_CLIFF_MIN_RATIO", "0.05"))
    )
    candidate_top_k: int = field(
        default_factory=lambda: int(os.getenv("CANDIDATE_TOP_K", "8"))
    )
    lsh_threshold: float = 0.62
    lsh_num_perm: int = 64
    lsh_query_jaccard_threshold: float = 0.78
    lsh_query_seq_ratio: float = 0.84
    lsh_query_combined_threshold: float = 0.8
    c_secondary_seq_ratio: float = 0.82
    c_secondary_jaccard: float = 0.55
    c_secondary_seq_with_jac: float = 0.62
    c_query_cover: float = 0.45
    lsh_query_jaccard_threshold: float = 0.78
    lsh_query_seq_ratio: float = 0.84
    lsh_query_seq_with_jac: float = 0.67
    lsh_query_cover: float = 0.58
    semantic_value_top_k: int = 5
    semantic_value_threshold: float = 0.8
    semantic_value_max_values_per_column: int = 200
    enable_semantic_value_retrieval: bool = field(
        default_factory=lambda: os.getenv("ENABLE_SEMANTIC_VALUE_RETRIEVAL", "True").lower() == "true"
    )
    top_k_embed: int = 10
    index_cache_dir: Path = field(default_factory=lambda: CACHE_DIR / "value_indexes")
    # few-shot 索引与其他索引库同路径（统一挂在 value_indexes 下）
    fewshot_index_dir: Path = field(default_factory=lambda: CACHE_DIR / "value_indexes" / "fewshot_index")

    # 轻量检索后，交给 LLM 做证据实体与字段分类时的上下文上限
    evidence_schema_top_k: int = 10
    evidence_entity_max_items: int = 8
    evidence_json_max_tokens: int = 512
    evidence_use_guided_json: bool = field(
        default_factory=lambda: os.getenv("EVIDENCE_USE_GUIDED_JSON", "True").lower() == "true"
    )

    # ── Embedding / retrieval ──
    embed_query_prompt: str = (
        "Instruct: 给定一个关于数据库的自然语言问题，检索语义最相似的历史查询模板\n"
        "Query: "
    )

    # ── Generator ──
    num_sql_per_path: int = 1
    icl_few_shot_k: int = field(
        default_factory=lambda: int(os.getenv("ICL_FEW_SHOT_K", "3"))
    )
    fewshot_autobuild_on_start: bool = field(
        default_factory=lambda: os.getenv("FEWSHOT_AUTOBUILD_ON_START", "True").lower() == "true"
    )
    icl_temperature: float = 0.1
    direct_temperature: float = 0.3
    max_gen_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_GEN_TOKENS", "1024"))
    )
    generator_candidates_per_route: int = field(
        default_factory=lambda: int(os.getenv("GEN_CANDIDATES_PER_ROUTE", "2"))
    )
    intent_plan_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("INTENT_PLAN_MAX_TOKENS", "512"))
    )
    intent_plan_use_guided_json: bool = field(
        default_factory=lambda: os.getenv("INTENT_PLAN_USE_GUIDED_JSON", "True").lower() == "true"
    )
    llm_request_timeout_sec: int = field(
        default_factory=lambda: int(os.getenv("LLM_REQUEST_TIMEOUT_SEC", "180"))
    )
    enable_thinking_for_entity: bool = field(
        default_factory=lambda: os.getenv("ENTITY_ENABLE_THINKING", "False").lower() == "true"
    )
    enable_thinking_for_refiner: bool = field(
        default_factory=lambda: os.getenv("REFINER_ENABLE_THINKING", "False").lower() == "true"
    )
    baseline_enable_thinking: bool = field(
        default_factory=lambda: os.getenv("BASELINE_ENABLE_THINKING", "False").lower() == "true"
    )

    # ── Entity / evidence extraction ──
    entity_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("ENTITY_MAX_TOKENS", "384"))
    )
    entity_use_guided_json: bool = field(
        default_factory=lambda: os.getenv("ENTITY_USE_GUIDED_JSON", "True").lower() == "true"
    )
    entity_prefix_bracket: bool = field(
        default_factory=lambda: os.getenv("ENTITY_PREFIX_BRACKET", "False").lower() == "true"
    )
    enable_entity_extraction: bool = field(
        default_factory=lambda: os.getenv("ENABLE_ENTITY_EXTRACTION", "False").lower() == "true"
    )

    # ── Refiner / selector ──
    max_repair_retries: int = 2
    refiner_temperature: float = 0.01
    refiner_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("REFINER_MAX_TOKENS", "1024"))
    )
    refiner_enforce_timeout: bool = field(
        default_factory=lambda: os.getenv("REFINER_ENFORCE_TIMEOUT", "False").lower() == "true"
    )
    enable_sql_consistency_judge: bool = field(
        default_factory=lambda: os.getenv("ENABLE_SQL_CONSISTENCY_JUDGE", "True").lower() == "true"
    )
    generator_prefix_code_fence: bool = field(
        default_factory=lambda: os.getenv("GEN_PREFIX_CODE_FENCE", "False").lower() == "true"
    )

    # ── Profiler ──
    profile_sample_rows: int = 100
    profile_distinct_threshold: int = 80
    profile_enum_full_threshold: int = 15
    profile_example_k: int = field(
        default_factory=lambda: int(os.getenv("PROFILE_EXAMPLE_K", "2"))
    )
    deprecated_column_null_ratio_threshold: float = field(
        default_factory=lambda: float(os.getenv("DEPRECATED_COLUMN_NULL_RATIO_THRESHOLD", "0.95"))
    )

    # ── Flow control ──
    enable_question_split: bool = field(
        default_factory=lambda: os.getenv("ENABLE_QUESTION_SPLIT", "True").lower() == "true"
    )

    # ── 调试 ──
    debug_mode: bool = field(
        default_factory=lambda: os.getenv("DEBUG_MODE", "True").lower() == "true"
    )

    # ── 训练 ──
    train_split: float = field(
        default_factory=lambda: float(os.getenv("TRAIN_SPLIT", "0.8"))
    )
    val_split: float = field(
        default_factory=lambda: float(os.getenv("VAL_SPLIT", "0.1"))
    )
    test_split: float = field(
        default_factory=lambda: float(os.getenv("TEST_SPLIT", "0.1"))
    )
    eval_use_test_split_jsonl: bool = field(
        default_factory=lambda: os.getenv("EVAL_USE_TEST_SPLIT_JSONL", "True").lower() == "true"
    )
    random_state: int = 42


settings = Settings()
