from __future__ import annotations

import inspect
import math
import re
from hashlib import sha256
from typing import Awaitable, Callable, Iterable, List, Sequence, Union

from service.embedding.embedding_cache import ChunkEmbeddingCache, EmbeddingCache
from utils.async_utils import bounded_gather
from utils.content_normalizer import normalize_whitespace
from utils.hash_utils import build_cache_key

EMBEDDING_DIMENSION = 1024
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

EmbeddingProvider = Callable[[str], Union[Awaitable[Sequence[float]], Sequence[float]]]


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
        except Exception:
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

