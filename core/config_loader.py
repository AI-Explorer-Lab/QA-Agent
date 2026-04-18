from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "app.yaml"
DEFAULT_PROVIDER_KEY_ENV = {
    "anyrouter": "OPENAI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "QWEN_API_KEY",
    "zhipu": "ZHIPUAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
MODEL_SELECTOR_KEYS = (
    "current_model",
    "current",
    "active_model",
    "model_selector",
    "provider_model",
)


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_bool(value: Any, default: bool = False) -> bool:
    text = _clean_str(value).lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "on"}


def _normalize_openai_base_url(base_url: str) -> str:
    """
    Normalize OpenAI-compatible base URL.
    If no explicit path is provided, append `/v1`.
    """
    value = _clean_str(base_url)
    if not value:
        return ""

    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return value.rstrip("/")

    path = (parsed.path or "").rstrip("/")
    if not path:
        return value.rstrip("/") + "/v1"
    return value.rstrip("/")


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def _resolve_selected_model_selector(llm_config: Dict[str, Any]) -> str:
    for key in MODEL_SELECTOR_KEYS:
        selector = _clean_str(llm_config.get(key))
        if selector:
            return selector
    return ""


def _split_selector(selector: str, providers: Dict[str, Any]) -> Tuple[str, str]:
    selector = _clean_str(selector)
    if not selector:
        return "", ""

    provider_names = []
    if isinstance(providers, dict):
        provider_names = [_clean_str(name) for name in providers.keys() if _clean_str(name)]

    # Prefer exact provider prefix matching, e.g. anyrouter-gpt-5.3-codex
    for provider_name in sorted(provider_names, key=len, reverse=True):
        prefix = f"{provider_name}-"
        if selector.startswith(prefix):
            return provider_name, _clean_str(selector[len(prefix):])

    # Fallback to first '-' split.
    if "-" in selector:
        provider_name, model_key = selector.split("-", 1)
        return _clean_str(provider_name), _clean_str(model_key)

    # Selector only contains model-like value.
    return "", selector


def _resolve_provider_and_model_blocks(
    llm_config: Dict[str, Any],
) -> Tuple[str, str, str, Dict[str, Any], Dict[str, Any]]:
    providers = llm_config.get("providers", {})
    if not isinstance(providers, dict):
        providers = {}

    selector = _resolve_selected_model_selector(llm_config)
    provider_from_selector, model_key = _split_selector(selector, providers)

    provider_name = provider_from_selector or _clean_str(llm_config.get("provider"))
    provider_block = providers.get(provider_name, {}) if provider_name else {}
    if not isinstance(provider_block, dict):
        provider_block = {}

    models_block = provider_block.get("models", {})
    if not isinstance(models_block, dict):
        models_block = {}

    model_block: Dict[str, Any] = {}
    if model_key:
        block = models_block.get(model_key, {})
        if isinstance(block, dict):
            model_block = block

    if not model_key:
        fallback_model_key = _clean_str(
            provider_block.get("default_model_key") or llm_config.get("model_key")
        )
        if fallback_model_key:
            block = models_block.get(fallback_model_key, {})
            if isinstance(block, dict):
                model_key = fallback_model_key
                model_block = block

    if not model_key and not model_block:
        provider_level_model = _clean_str(provider_block.get("model"))
        if provider_level_model:
            model_key = provider_level_model

    return selector, provider_name, model_key, provider_block, model_block


def _pick_llm_value(
    key: str,
    llm_config: Dict[str, Any],
    provider_block: Dict[str, Any],
    model_block: Dict[str, Any],
) -> Any:
    if isinstance(model_block, dict) and key in model_block:
        return model_block.get(key)
    if isinstance(provider_block, dict) and key in provider_block:
        return provider_block.get(key)
    return llm_config.get(key)


def _resolve_llm_values(llm_config: Dict[str, Any]) -> Dict[str, str]:
    (
        selector,
        provider_name,
        model_key,
        provider_block,
        model_block,
    ) = _resolve_provider_and_model_blocks(llm_config)

    model = _clean_str(
        _pick_llm_value("model", llm_config, provider_block, model_block)
        or _pick_llm_value("model_name", llm_config, provider_block, model_block)
        or model_key
    )
    base_url = _normalize_openai_base_url(
        _clean_str(_pick_llm_value("base_url", llm_config, provider_block, model_block))
    )

    anthropic_base_url = _normalize_openai_base_url(
        _clean_str(
            _pick_llm_value(
                "anthropic_base_url", llm_config, provider_block, model_block
            )
            or base_url
        )
    )

    api_key = _clean_str(_pick_llm_value("api_key", llm_config, provider_block, model_block))
    api_key_env = _clean_str(
        _pick_llm_value("api_key_env", llm_config, provider_block, model_block)
    )
    if not api_key and api_key_env:
        api_key = _clean_str(os.getenv(api_key_env))
    if not api_key and provider_name:
        default_key_env = DEFAULT_PROVIDER_KEY_ENV.get(provider_name.lower(), "")
        if default_key_env:
            api_key = _clean_str(os.getenv(default_key_env))

    http_proxy = _clean_str(
        _pick_llm_value("http_proxy", llm_config, provider_block, model_block)
    )
    https_proxy = _clean_str(
        _pick_llm_value("https_proxy", llm_config, provider_block, model_block)
    )
    all_proxy = _clean_str(
        _pick_llm_value("all_proxy", llm_config, provider_block, model_block)
    )
    no_proxy = _clean_str(
        _pick_llm_value("no_proxy", llm_config, provider_block, model_block)
    )
    use_responses_api = _as_bool(
        _pick_llm_value("use_responses_api", llm_config, provider_block, model_block),
        default=False,
    )

    resolved_selector = selector
    if not resolved_selector and provider_name and model_key:
        resolved_selector = f"{provider_name}-{model_key}"

    resolved = {
        "LLM_PROVIDER": provider_name,
        "LLM_MODEL": model,
        "LLM_MODEL_KEY": model_key,
        "LLM_MODEL_SELECTOR": resolved_selector,
        "LLM_BASE_URL": base_url,
        "ANTHROPIC_BASE_URL": anthropic_base_url,
        "LLM_API_KEY": api_key,
        "LLM_API_KEY_ENV": api_key_env,
        "LLM_USE_RESPONSES_API": str(use_responses_api).lower(),
        "HTTP_PROXY": http_proxy,
        "HTTPS_PROXY": https_proxy,
        "ALL_PROXY": all_proxy,
        "NO_PROXY": no_proxy,
        # Lower-case aliases for libs that only read lower-case proxy vars.
        "http_proxy": http_proxy,
        "https_proxy": https_proxy,
        "all_proxy": all_proxy,
        "no_proxy": no_proxy,
    }
    return {k: _clean_str(v) for k, v in resolved.items() if _clean_str(v)}


def _map_to_env(config: Dict[str, Any]) -> Dict[str, str]:
    llm = config.get("llm", {}) if isinstance(config.get("llm"), dict) else {}
    keys = config.get("api_keys", {}) if isinstance(config.get("api_keys"), dict) else {}
    vector = config.get("vector", {}) if isinstance(config.get("vector"), dict) else {}
    storage = config.get("storage", {}) if isinstance(config.get("storage"), dict) else {}
    storage_pgvector = storage.get("pgvector", {}) if isinstance(storage.get("pgvector"), dict) else {}
    rag = config.get("rag", {}) if isinstance(config.get("rag"), dict) else {}
    retrieval = config.get("retrieval", {}) if isinstance(config.get("retrieval"), dict) else {}
    chunking = config.get("chunking", {}) if isinstance(config.get("chunking"), dict) else {}
    embedding = config.get("embedding", {}) if isinstance(config.get("embedding"), dict) else {}
    react_guardrails = (
        config.get("react_guardrails", {})
        if isinstance(config.get("react_guardrails"), dict)
        else {}
    )

    mineru_api_key_env = _clean_str(keys.get("mineru_api_key_env") or "MinerU_API_KEY")
    mineru_api_key = _clean_str(keys.get("mineru_api_key"))
    if not mineru_api_key and mineru_api_key_env:
        mineru_api_key = _clean_str(os.getenv(mineru_api_key_env))

    embedding_api_key_env = _clean_str(embedding.get("api_key_env") or "QWEN_API_KEY")
    embedding_api_key = _clean_str(embedding.get("api_key") or keys.get("qwen_api_key"))
    if not embedding_api_key and embedding_api_key_env:
        embedding_api_key = _clean_str(os.getenv(embedding_api_key_env))

    env_map = {
        "OPENAI_API_KEY": keys.get("openai_api_key"),
        "DEEPSEEK_API_KEY": keys.get("deepseek_api_key"),
        "QWEN_API_KEY": embedding_api_key,
        "LANGCHAIN_API_KEY": keys.get("langchain_api_key"),
        "ZHIPUAI_API_KEY": keys.get("zhipuai_api_key"),
        "ANTHROPIC_API_KEY": keys.get("anthropic_api_key"),
        "MinerU_API_KEY": mineru_api_key,
        "MINERU_API_KEY": mineru_api_key,
        "MINERU_API_KEY_ENV": mineru_api_key_env,
        "EMBEDDING_PROVIDER": embedding.get("provider"),
        "EMBEDDING_MODEL": embedding.get("model"),
        "EMBEDDING_DIMENSION": embedding.get("dimension"),
        "EMBEDDING_API_KEY": embedding_api_key,
        "EMBEDDING_API_KEY_ENV": embedding_api_key_env,
        "EMBEDDING_BASE_URL": embedding.get("base_url"),
        "STORAGE_BACKEND": storage.get("backend") or vector.get("backend"),
        "VECTOR_STORE_BACKEND": storage.get("backend") or vector.get("backend"),
        "PGVECTOR_DATABASE_URL": storage_pgvector.get("database_url") or vector.get("pgvector_database_url"),
        "PGVECTOR_EMBEDDING_DIM": storage_pgvector.get("embedding_dim") or vector.get("pgvector_embedding_dim"),
        "RAG_EXPAND_QUERY_NUM": rag.get("expand_query_num"),
        "RAG_RETRIEVED_ANSWERS": rag.get("retrieved_answers"),
        "HYBRID_DENSE_POOL_FACTOR": retrieval.get("hybrid_dense_pool_factor"),
        "HYBRID_SPARSE_POOL_FACTOR": retrieval.get("hybrid_sparse_pool_factor"),
        "HYBRID_SPARSE_ALGORITHM": retrieval.get("sparse_algorithm"),
        "HYBRID_SPARSE_SCAN_LIMIT": retrieval.get("hybrid_sparse_scan_limit"),
        "HYBRID_SPARSE_MIN_SCORE": retrieval.get("hybrid_sparse_min_score"),
        "HYBRID_BM25_K1": retrieval.get("bm25_k1"),
        "HYBRID_BM25_B": retrieval.get("bm25_b"),
        "HYBRID_BM25_MIN_SCORE": retrieval.get("bm25_min_score"),
        "HYBRID_TABLE_QUOTA": retrieval.get("hybrid_table_quota"),
        "HYBRID_TABLE_SCORE_FLOOR": retrieval.get("hybrid_table_score_floor"),
        "CHUNK_SIZE_TOKENS": chunking.get("chunk_size_tokens"),
        "CHUNK_OVERLAP_TOKENS": chunking.get("chunk_overlap_tokens"),
        "MAX_CHUNK_SIZE": chunking.get("max_chunk_size"),
        "CHUNK_HEADING_MAX_LEVEL": chunking.get("heading_max_level"),
        "REACT_CLARIFY_ENABLED": react_guardrails.get("clarify_enabled"),
        "REACT_CLARIFY_RAG_ONLY": react_guardrails.get("clarify_rag_only"),
        "REACT_CLARIFY_MAX_TURNS": react_guardrails.get("clarify_max_turns"),
        "REACT_EVIDENCE_MIN_DOCS": react_guardrails.get("evidence_min_docs"),
        "REACT_EVIDENCE_MIN_TOP_SIMILARITY": react_guardrails.get("evidence_min_top_similarity"),
        "REACT_EVIDENCE_MIN_AVG_SIMILARITY": react_guardrails.get("evidence_min_avg_similarity"),
        "REACT_EVIDENCE_MIN_OVERALL_SCORE": react_guardrails.get("evidence_min_overall_score"),
        "REACT_EVIDENCE_RETRY_LIMIT": react_guardrails.get("evidence_retry_limit"),
        "REACT_REFUSE_ON_LOW_EVIDENCE": react_guardrails.get("refuse_on_low_evidence"),
    }
    env_map.update(_resolve_llm_values(llm))
    if mineru_api_key and mineru_api_key_env:
        env_map[mineru_api_key_env] = mineru_api_key
    if embedding_api_key and embedding_api_key_env:
        env_map[embedding_api_key_env] = embedding_api_key

    cleaned: Dict[str, str] = {}
    for key, value in env_map.items():
        value_str = _clean_str(value)
        if not value_str:
            continue
        cleaned[key] = value_str
    return cleaned


@lru_cache(maxsize=1)
def load_runtime_env() -> Dict[str, str]:
    """
    Load YAML config and project it into environment variables.
    Priority:
    1) Existing process env / .env values
    2) YAML defaults
    """
    # Always load project-root .env so runtime does not depend on current working directory.
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    config_path = Path(os.getenv("APP_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
    yaml_config = _read_yaml(config_path)
    yaml_env = _map_to_env(yaml_config)

    for env_key, env_value in yaml_env.items():
        os.environ.setdefault(env_key, env_value)

    return yaml_env


def get_llm_runtime_config(
    default_model: str = "",
    default_base_url: str = "",
) -> Dict[str, Any]:
    """
    Resolve effective runtime LLM connection settings.
    Priority is still environment-first, with YAML providing defaults.
    """
    load_runtime_env()

    model_key = _clean_str(os.getenv("LLM_MODEL_KEY"))
    model = _clean_str(os.getenv("LLM_MODEL")) or model_key or default_model
    base_url = (
        _normalize_openai_base_url(_clean_str(os.getenv("LLM_BASE_URL")))
        or _normalize_openai_base_url(_clean_str(os.getenv("ANTHROPIC_BASE_URL")))
        or _normalize_openai_base_url(default_base_url)
    )

    api_key_candidates = (
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "QWEN_API_KEY",
        "ZHIPUAI_API_KEY",
        "ANTHROPIC_API_KEY",
    )
    api_key = ""
    for key_name in api_key_candidates:
        api_key = _clean_str(os.getenv(key_name))
        if api_key:
            break

    return {
        "provider": _clean_str(os.getenv("LLM_PROVIDER")),
        "model": model,
        "model_key": model_key,
        "model_selector": _clean_str(os.getenv("LLM_MODEL_SELECTOR")),
        "base_url": base_url,
        "api_key": api_key,
        "api_key_env": _clean_str(os.getenv("LLM_API_KEY_ENV")),
        "use_responses_api": _as_bool(os.getenv("LLM_USE_RESPONSES_API"), default=False),
    }
