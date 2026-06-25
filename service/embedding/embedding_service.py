from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import re
from hashlib import sha256
from http import HTTPStatus
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Sequence, Union

from core.config_loader import load_runtime_env
from middlewares.operation_log import log_operation_event
from utils.config_loader import get_app_config
from service.embedding.embedding_cache import ChunkEmbeddingCache, EmbeddingCache
from utils.async_utils import bounded_gather
from utils.content_normalizer import normalize_whitespace
from utils.hash_utils import build_cache_key

EMBEDDING_DIMENSION = 1024
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

EmbeddingProvider = Callable[[str], Union[Awaitable[Sequence[float]], Sequence[float]]]

try:  # pragma: no cover - optional runtime dependency
    import dashscope
except Exception:  # pragma: no cover
    dashscope = None


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_secret(value: Any) -> str:
    text = _clean_str(value)
    if not text or text.startswith("***REMOVED"):
        return ""
    return text


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = _clean_secret(value)
        if text:
            return text
    return ""


def _safe_message(value: Any, limit: int = 300) -> str:
    return str(value or "")[:limit]


def _resolve_embedding_runtime_config(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    load_runtime_env()
    app_config = config or get_app_config()
    embedding_cfg = app_config.get("embedding", {}) if isinstance(app_config.get("embedding"), dict) else {}
    key_cfg = app_config.get("api_keys", {}) if isinstance(app_config.get("api_keys"), dict) else {}

    api_key_env = _first_nonempty(
        os.getenv("EMBEDDING_API_KEY_ENV"),
        embedding_cfg.get("api_key_env"),
        "QWEN_API_KEY",
    )
    api_key = _first_nonempty(
        os.getenv("EMBEDDING_API_KEY"),
        os.getenv(api_key_env) if api_key_env else "",
        os.getenv("QWEN_API_KEY"),
        embedding_cfg.get("api_key"),
        key_cfg.get("qwen_api_key"),
    )
    timeout_seconds = _first_nonempty(
        os.getenv("EMBEDDING_TIMEOUT_SECONDS"),
        embedding_cfg.get("timeout_seconds"),
        20,
    )
    try:
        timeout_value = float(timeout_seconds)
    except Exception:
        timeout_value = 20.0

    return {
        "provider": _first_nonempty(os.getenv("EMBEDDING_PROVIDER"), embedding_cfg.get("provider"), "qwen"),
        "model": _first_nonempty(os.getenv("EMBEDDING_MODEL"), embedding_cfg.get("model"), "text-embedding-v4"),
        "api_key": api_key,
        "api_key_env": api_key_env,
        "base_url": _first_nonempty(os.getenv("EMBEDDING_BASE_URL"), embedding_cfg.get("base_url")),
        "timeout_seconds": timeout_value,
    }


def build_embedding_provider_from_config(config: Dict[str, Any] | None = None) -> EmbeddingProvider | None:
    runtime = _resolve_embedding_runtime_config(config)
    provider_name = str(runtime["provider"]).strip().lower()
    model = str(runtime["model"]).strip()
    api_key = str(runtime["api_key"]).strip()
    timeout_seconds = float(runtime["timeout_seconds"])

    if provider_name in {"deterministic", "deterministic_hash_embedding", "local", "none"}:
        log_operation_event(
            "index.embedding.provider",
            status="selected",
            provider="deterministic_hash_embedding",
            model="deterministic_hash_embedding",
            reason="local_embedding_provider",
        )
        return None

    if provider_name not in {"qwen", "dashscope"}:
        log_operation_event(
            "index.embedding.provider",
            status="disabled",
            provider=provider_name or "unknown",
            reason="unsupported_embedding_provider",
        )
        return None

    if dashscope is None:
        log_operation_event(
            "index.embedding.provider",
            status="warning",
            level=logging.WARNING,
            provider=provider_name,
            model=model,
            reason="dashscope_not_installed",
        )
        return None

    if not api_key:
        log_operation_event(
            "index.embedding.provider",
            status="warning",
            level=logging.WARNING,
            provider=provider_name,
            model=model,
            api_key_env=runtime.get("api_key_env"),
            reason="missing_embedding_api_key",
        )
        return None

    def _call_dashscope(text: str) -> Sequence[float]:
        response = dashscope.TextEmbedding.call(
            model=model,
            input=text,
            api_key=api_key,
            timeout=timeout_seconds,
        )
        status_code = getattr(response, "status_code", None)
        if status_code == HTTPStatus.OK or int(status_code or 0) == 200:
            output = getattr(response, "output", None) or {}
            embeddings = output.get("embeddings") if isinstance(output, dict) else None
            if embeddings and isinstance(embeddings, list):
                vector = embeddings[0].get("embedding") if isinstance(embeddings[0], dict) else None
                if isinstance(vector, list) and vector:
                    return vector
            raise RuntimeError("DashScope embedding returned an empty embedding payload.")

        message = getattr(response, "message", "")
        code = getattr(response, "code", "")
        raise RuntimeError(
            f"DashScope embedding failed: status={status_code}, code={_safe_message(code)}, message={_safe_message(message)}"
        )

    async def _provider(text: str) -> Sequence[float]:
        return await asyncio.to_thread(_call_dashscope, text)

    setattr(_provider, "_trusted_qa_provider_name", provider_name)
    setattr(_provider, "_trusted_qa_provider_model", model)
    log_operation_event(
        "index.embedding.provider",
        status="selected",
        provider=provider_name,
        model=model,
        base_url_set=bool(runtime.get("base_url")),
    )
    return _provider


class EmbeddingService:
    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
        cache_ttl_seconds: int = 3600,
        cache_max_items: int = 5000,
        embedding_cache: EmbeddingCache | None = None,
        chunk_embedding_cache: ChunkEmbeddingCache | None = None,
    ) -> None:
        self.provider = provider
        self.provider_name = (
            str(getattr(provider, "_trusted_qa_provider_name", "custom_provider"))
            if provider is not None
            else "deterministic_hash_embedding"
        )
        self.provider_model = (
            str(getattr(provider, "_trusted_qa_provider_model", ""))
            if provider is not None
            else ""
        )
        self.embedding_dim = EMBEDDING_DIMENSION
        self.embedding_cache = embedding_cache or EmbeddingCache(
            ttl_seconds=cache_ttl_seconds,
            max_items=cache_max_items,
        )
        self.chunk_embedding_cache = chunk_embedding_cache or ChunkEmbeddingCache(
            ttl_seconds=cache_ttl_seconds,
            max_items=cache_max_items,
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        if tokens:
            return tokens
        return list(text)

    def _deterministic_hash_embedding(self, text: str) -> List[float]:
        normalized = normalize_whitespace(text, preserve_newlines=False)
        if not normalized:
            return [0.0] * self.embedding_dim

        vector = [0.0] * self.embedding_dim
        for index, token in enumerate(self._tokenize(normalized)):
            digest = sha256(f"{token}|{index % 23}".encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.embedding_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            magnitude = 0.5 + (digest[5] / 255.0)
            vector[bucket] += sign * magnitude

        global_digest = sha256(normalized.encode("utf-8")).digest()
        for idx, byte in enumerate(global_digest):
            bucket = (idx * 67 + byte) % self.embedding_dim
            vector[bucket] += ((byte / 255.0) - 0.5) * 0.1

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return [0.0] * self.embedding_dim
        return [value / norm for value in vector]

    def _normalize_dimension(self, vector: Sequence[float]) -> List[float]:
        values = [float(item) for item in vector[: self.embedding_dim]]
        if len(values) < self.embedding_dim:
            values.extend([0.0] * (self.embedding_dim - len(values)))

        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 0:
            return [0.0] * self.embedding_dim
        return [value / norm for value in values]

    async def _embed_with_provider(self, text: str) -> List[float]:
        if self.provider is None:
            return self._deterministic_hash_embedding(text)

        try:
            response = self.provider(text)
            if inspect.isawaitable(response):
                response = await response  # type: ignore[assignment]
            return self._normalize_dimension(response)
        except Exception as exc:
            log_operation_event(
                "index.embedding.provider_fallback",
                status="warning",
                level=logging.WARNING,
                error_type=type(exc).__name__,
                error=str(exc),
                fallback="deterministic_hash_embedding",
            )
            return self._deterministic_hash_embedding(text)

    async def embed_text(self, text: str, use_cache: bool = True, chunk_text: bool = False) -> List[float]:
        normalized = normalize_whitespace(text, preserve_newlines=False)
        cache = self.chunk_embedding_cache if chunk_text else self.embedding_cache
        cache_prefix = "chunk_embedding" if chunk_text else "embedding"
        cache_key = build_cache_key(normalized, prefix=cache_prefix)

        if use_cache:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        vector = await self._embed_with_provider(normalized)
        if use_cache:
            cache.set(cache_key, vector)
        return vector

    async def embed_texts(
        self,
        texts: Sequence[str] | Iterable[str],
        use_cache: bool = True,
        chunk_text: bool = False,
        max_concurrency: int = 8,
        timeout_seconds: float | None = None,
    ) -> List[List[float]]:
        text_list = list(texts)
        coroutines = [self.embed_text(text, use_cache=use_cache, chunk_text=chunk_text) for text in text_list]
        if not coroutines:
            return []
        results = await bounded_gather(
            coroutines,
            limit=max(1, int(max_concurrency)),
            timeout_seconds=timeout_seconds,
            return_exceptions=False,
        )
        return [list(vector) for vector in results]


_default_embedding_service = EmbeddingService()


async def embed_text(text: str, use_cache: bool = True, chunk_text: bool = False) -> List[float]:
    return await _default_embedding_service.embed_text(text=text, use_cache=use_cache, chunk_text=chunk_text)


async def embed_texts(
    texts: Sequence[str] | Iterable[str],
    use_cache: bool = True,
    chunk_text: bool = False,
    max_concurrency: int = 8,
    timeout_seconds: float | None = None,
) -> List[List[float]]:
    return await _default_embedding_service.embed_texts(
        texts=texts,
        use_cache=use_cache,
        chunk_text=chunk_text,
        max_concurrency=max_concurrency,
        timeout_seconds=timeout_seconds,
    )

