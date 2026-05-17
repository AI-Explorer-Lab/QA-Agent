"""Load the project config/app.yaml file with graceful YAML fallback."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from exception import ConfigException

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "app.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "app": {
        "name": "Enterprise-Unstructured-Document-Trusted-Question-Answering-Agent",
        "env": "dev",
        "timezone": "Asia/Singapore",
    },
    "llm": {
        "current_model": "anyrouter-gpt-5.5",
        "temperature": 0.2,
        "max_tokens": 2048,
    },
    "agent": {
        "orchestration": "trusted_qa_workflow",
        "max_iterations": 6,
        "skill_trace_enabled": True,
        "default_skill": "fact_lookup",
    },
    "skills": {
        "enabled": [
            "fact_lookup",
            "table_qa",
            "citation_locate",
            "summarization",
            "report_generation",
            "multi_doc_compare",
        ],
        "clarify_before_skill": True,
        "fallback_skill": "fact_lookup",
    },
    "embedding": {
        "provider": "qwen",
        "model": "text-embedding-v4",
        "dimension": 1024,
        "api_key": "",
        "api_key_env": "QWEN_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "storage": {
        "backend": "pgvector",
        "pgvector": {
            "database_url": "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/trusted_qa",
            "embedding_dim": 1024,
        },
        "local_dev": {
            "enabled": False,
            "database_url": "sqlite:///database/local_dev.db",
        },
    },
    "pdf": {
        "parser": "mineru",
        "max_pages_per_task": 200,
    },
    "chunking": {
        "chunk_size_tokens": 1024,
        "chunk_overlap_tokens": 200,
        "max_chunk_size": 7000,
        "heading_max_level": 3,
    },
    "retrieval": {
        "strategy": "hybrid",
        "top_k": 5,
        "expand_query_num": 4,
        "max_concurrency": 6,
        "query_timeout_seconds": 20,
        "table_evidence_quota": 2,
    },
    "reranker": {
        "dense_weight": 0.50,
        "bm25_weight": 0.35,
        "metadata_boost_weight": 0.10,
        "table_boost_weight": 0.05,
        "top_n_factor": 4,
        "near_duplicate_threshold": 0.90,
        "cross_encoder_enabled": True,
        "cross_encoder_model": "BAAI/bge-reranker-base",
        "cross_encoder_candidate_pool": 30,
        "cross_encoder_batch_size": 8,
        "cross_encoder_max_length": 512,
        "cross_encoder_local_files_only": False,
    },
    "cache": {
        "enabled": True,
        "ttl_seconds": 3600,
        "max_items": 5000,
        "embedding_cache_enabled": True,
        "retrieval_cache_enabled": True,
        "document_parse_cache_enabled": True,
    },
    "api_keys": {
        "openai_api_key": "",
        "deepseek_api_key": "",
        "qwen_api_key": "",
        "zhipuai_api_key": "",
        "anthropic_api_key": "",
        "mineru_api_key": "",
        "mineru_api_key_env": "MinerU_API_KEY",
    },
    "guardrails": {
        "evidence_min_docs": 2,
        "evidence_min_top_score": 0.45,
        "evidence_min_avg_score": 0.30,
        "retry_limit": 2,
        "refuse_on_low_evidence": True,
    },
}


@dataclass(frozen=True)
class _YamlLine:
    indent: int
    text: str


def _strip_comment(line: str) -> str:
    in_single_quote = False
    in_double_quote = False
    for index, char in enumerate(line):
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if char == "#" and not in_single_quote and not in_double_quote:
            return line[:index]
    return line


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""

    if value in {"null", "Null", "NULL", "~"}:
        return None

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if (value.startswith("\"") and value.endswith("\"")) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _to_yaml_lines(text: str) -> list[_YamlLine]:
    result: list[_YamlLine] = []
    for raw_line in text.splitlines():
        without_comment = _strip_comment(raw_line.rstrip())
        if not without_comment.strip():
            continue
        indent = len(without_comment) - len(without_comment.lstrip(" "))
        if indent < 0:
            indent = 0
        result.append(_YamlLine(indent=indent, text=without_comment.strip()))
    return result


def _parse_yaml_node(lines: list[_YamlLine], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    line = lines[index]
    if line.indent != indent:
        raise ConfigException(
            message="Invalid YAML indentation in fallback parser.",
            detail={"line": line.text, "indent": line.indent, "expected": indent},
        )

    if line.text.startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_dict(lines, index, indent)


def _parse_yaml_dict(lines: list[_YamlLine], index: int, indent: int) -> tuple[dict[str, Any], int]:
    data: dict[str, Any] = {}

    while index < len(lines):
        line = lines[index]
        if line.indent < indent:
            break
        if line.indent > indent:
            raise ConfigException(
                message="Unexpected YAML indentation in fallback parser.",
                detail={"line": line.text, "indent": line.indent, "expected": indent},
            )
        if line.text.startswith("- "):
            break

        key, delimiter, remainder = line.text.partition(":")
        if not delimiter:
            raise ConfigException(
                message="Invalid YAML key in fallback parser.",
                detail={"line": line.text},
            )

        key = key.strip()
        remainder = remainder.strip()
        index += 1

        if remainder:
            data[key] = _parse_scalar(remainder)
            continue

        if index < len(lines) and lines[index].indent > indent:
            nested, index = _parse_yaml_node(lines, index, lines[index].indent)
            data[key] = nested
        else:
            data[key] = {}

    return data, index


def _parse_yaml_list(lines: list[_YamlLine], index: int, indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []

    while index < len(lines):
        line = lines[index]
        if line.indent < indent:
            break
        if line.indent > indent:
            raise ConfigException(
                message="Unexpected YAML indentation inside list in fallback parser.",
                detail={"line": line.text, "indent": line.indent, "expected": indent},
            )
        if not line.text.startswith("- "):
            break

        payload = line.text[2:].strip()
        index += 1

        if not payload:
            if index < len(lines) and lines[index].indent > indent:
                nested, index = _parse_yaml_node(lines, index, lines[index].indent)
                items.append(nested)
            else:
                items.append(None)
            continue

        if ":" in payload and not payload.startswith("'") and not payload.startswith('"'):
            inline_key, delimiter, inline_value = payload.partition(":")
            if delimiter:
                key = inline_key.strip()
                value_text = inline_value.strip()
                inline_obj: dict[str, Any] = {}
                if value_text:
                    inline_obj[key] = _parse_scalar(value_text)
                elif index < len(lines) and lines[index].indent > indent:
                    nested_value, index = _parse_yaml_node(lines, index, lines[index].indent)
                    inline_obj[key] = nested_value
                else:
                    inline_obj[key] = {}
                items.append(inline_obj)
                continue

        items.append(_parse_scalar(payload))

    return items, index


def _basic_yaml_load(text: str) -> dict[str, Any]:
    lines = _to_yaml_lines(text)
    if not lines:
        return {}

    data, index = _parse_yaml_node(lines, 0, lines[0].indent)
    if index != len(lines):
        raise ConfigException(
            message="Fallback YAML parser did not consume full document.",
            detail={"parsed_until": index, "line_count": len(lines)},
        )

    if isinstance(data, dict):
        return data
    raise ConfigException(message="Root YAML node must be a mapping.")


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in incoming.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}

    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return {}

    if yaml is not None:
        loaded = yaml.safe_load(raw_text) or {}
        if isinstance(loaded, dict):
            return loaded
        raise ConfigException(
            message="YAML root must be a mapping.",
            detail={"path": str(path)},
        )

    try:
        loaded_json = json.loads(raw_text)
        if isinstance(loaded_json, dict):
            return loaded_json
    except json.JSONDecodeError:
        pass

    return _basic_yaml_load(raw_text)


def _discover_yaml_paths(config_root: Path) -> list[Path]:
    if not config_root.exists() or not config_root.is_dir():
        return []

    yaml_paths = list(config_root.rglob("*.yaml")) + list(config_root.rglob("*.yml"))
    yaml_paths.sort(key=lambda item: str(item).lower())
    return yaml_paths


def _normalize_extra_paths(extra_paths: Iterable[str | Path] | None) -> list[Path]:
    if not extra_paths:
        return []

    normalized: list[Path] = []
    for item in extra_paths:
        normalized.append(Path(item).expanduser())
    return normalized


def load_yaml_config(
    config_root: str | Path | None = None,
    extra_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_CONFIG)
    explicit_config_path = os.getenv("APP_CONFIG_PATH", "").strip()

    if config_root:
        root = Path(config_root).expanduser()
        if root.is_dir():
            for yaml_path in _discover_yaml_paths(root):
                merged = _deep_merge(merged, _load_yaml_file(yaml_path))
        else:
            merged = _deep_merge(merged, _load_yaml_file(root))
    else:
        config_path = Path(
            explicit_config_path
            if explicit_config_path
            else DEFAULT_CONFIG_PATH
        ).expanduser()
        merged = _deep_merge(merged, _load_yaml_file(config_path))

        legacy_config_dir = os.getenv("APP_CONFIG_DIR", "").strip()
        if legacy_config_dir:
            legacy_root = Path(legacy_config_dir).expanduser()
            for yaml_path in _discover_yaml_paths(legacy_root):
                merged = _deep_merge(merged, _load_yaml_file(yaml_path))

    for extra_path in _normalize_extra_paths(extra_paths):
        merged = _deep_merge(merged, _load_yaml_file(extra_path))

    return merged


@lru_cache(maxsize=1)
def _cached_app_config() -> dict[str, Any]:
    return load_yaml_config()


def get_app_config(reload: bool = False) -> dict[str, Any]:
    if reload:
        _cached_app_config.cache_clear()
    return copy.deepcopy(_cached_app_config())
