from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, Iterable, List, Sequence

from .pgvector_repository import PgvectorRepository, deterministic_embedding
from .retrieval_cache import RetrievalResultCache
from .sparse_retriever import coarse_tokenize

try:  # pragma: no cover
    from utils.hash_utils import stable_sha256
except Exception:  # pragma: no cover
    import hashlib

    def stable_sha256(value: Any) -> str:
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _looks_like_table_query(query: str) -> bool:
    text = str(query or "").lower()
    if not text:
        return False

    hints = [
        "table",
        "指标",
        "单位",
        "同比",
        "环比",
        "表",
        "数据",
        "metric",
        "ratio",
        "revenue",
    ]
    if any(hint in text for hint in hints):
        return True

    token_count = len(coarse_tokenize(text))
    return token_count >= 4 and any(char.isdigit() for char in text)


def default_query_expander(question: str, expand_query_num: int) -> list[str]:
    total = max(0, int(expand_query_num))
    if total <= 0:
        return []

    text = str(question or "").strip()
    if not text:
        return []

    candidates: list[str] = []
    normalized = " ".join(coarse_tokenize(text))
    if normalized and normalized != text:
        candidates.append(normalized)

    zh_only = "".join(part for part in coarse_tokenize(text) if any("\u4e00" <= char <= "\u9fff" for char in part))
    if zh_only and zh_only != text:
        candidates.append(zh_only)

    en_tokens = [part for part in coarse_tokenize(text) if part.isascii()]
    if len(en_tokens) >= 2:
        candidates.append(" ".join(en_tokens[: min(6, len(en_tokens))]))

    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate != text and candidate not in unique:
            unique.append(candidate)
        if len(unique) >= total:
            break
    return unique[:total]


class ParallelQueryExecutor:
    def __init__(
        self,
        repository: PgvectorRepository,
        retrieval_cache: RetrievalResultCache | None = None,
        query_expander: Callable[[str, int], Iterable[str]] | None = None,
        embedding_builder: Callable[[str, int], Sequence[float]] | None = None,
        max_concurrency: int = 6,
        query_timeout_seconds: float = 20.0,
    ) -> None:
        self.repository = repository
        self.retrieval_cache = retrieval_cache
        self.query_expander = query_expander or default_query_expander
        self.embedding_builder = embedding_builder or deterministic_embedding
        self.max_concurrency = max(1, int(max_concurrency))
        self.query_timeout_seconds = max(0.01, float(query_timeout_seconds))

    async def execute(
        self,
        question: str,
        collection_name: str,
        top_k: int,
        query_type: str,
        expand_query_num: int,
        enable_cache: bool = True,
    ) -> Dict[str, Any]:
        effective_top_k = max(1, int(top_k))
        query_text = str(question or "").strip()
        effective_query_type = str(query_type or "fact_lookup")
        question_hash = stable_sha256(query_text)

        cache_key = None
        if self.retrieval_cache is not None:
            cache_key = self.retrieval_cache.build_key(
                collection_name=collection_name,
                question_hash=question_hash,
                query_type=effective_query_type,
                top_k=effective_top_k,
            )
            if enable_cache:
                cached = self.retrieval_cache.get(cache_key)
                if cached is not None:
                    trace = dict(cached.get("retrieval_trace") or {})
                    trace["cache_hit"] = True
                    trace["cache_key"] = cache_key.as_storage_key()
                    trace["cached_at"] = trace.get("generated_at")
                    trace["generated_at"] = time.time()
                    return {
                        "candidates": list(cached.get("candidates") or []),
                        "retrieval_trace": trace,
                    }

        query_variants = self._build_query_variants(query_text, expand_query_num)
        stage_top_n = max(effective_top_k * 4, effective_top_k)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        should_run_table = effective_query_type == "table_qa" or _looks_like_table_query(query_text)

        tasks: list[asyncio.Task[Dict[str, Any]]] = []
        for variant in query_variants:
            tasks.append(
                asyncio.create_task(
                    self._run_route_task(
                        route="dense",
                        query=variant,
                        collection_name=collection_name,
                        top_k=stage_top_n,
                        semaphore=semaphore,
                    )
                )
            )
            tasks.append(
                asyncio.create_task(
                    self._run_route_task(
                        route="bm25",
                        query=variant,
                        collection_name=collection_name,
                        top_k=stage_top_n,
                        semaphore=semaphore,
                    )
                )
            )
            if should_run_table:
                tasks.append(
                    asyncio.create_task(
                        self._run_route_task(
                            route="table",
                            query=variant,
                            collection_name=collection_name,
                            top_k=max(effective_top_k * 3, effective_top_k),
                            semaphore=semaphore,
                        )
                    )
                )

        raw_task_results = await asyncio.gather(*tasks)
        merged_candidates = self._merge_candidates(raw_task_results)

        retrieval_trace = {
            "collection_name": collection_name,
            "question_hash": question_hash,
            "query_type": effective_query_type,
            "top_k": effective_top_k,
            "query_variants": query_variants,
            "task_count": len(raw_task_results),
            "max_concurrency": self.max_concurrency,
            "query_timeout_seconds": self.query_timeout_seconds,
            "cache_hit": False,
            "cache_key": cache_key.as_storage_key() if cache_key is not None else "",
            "task_trace": [
                {
                    "route": item["route"],
                    "query": item["query"],
                    "duration_ms": item["duration_ms"],
                    "timed_out": item["timed_out"],
                    "error": item["error"],
                    "returned": item["returned"],
                }
                for item in raw_task_results
            ],
            "merged_candidate_count": len(merged_candidates),
            "generated_at": time.time(),
        }

        payload = {
            "candidates": merged_candidates,
            "retrieval_trace": retrieval_trace,
        }

        if enable_cache and self.retrieval_cache is not None and cache_key is not None:
            self.retrieval_cache.set(cache_key, payload)

        return payload

    def _build_query_variants(self, question: str, expand_query_num: int) -> list[str]:
        base = str(question or "").strip()
        expanded = list(self.query_expander(base, expand_query_num))
        variants: list[str] = []
        for item in [base] + expanded:
            value = str(item or "").strip()
            if value and value not in variants:
                variants.append(value)
        return variants

    async def _run_route_task(
        self,
        route: str,
        query: str,
        collection_name: str,
        top_k: int,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        items: list[Dict[str, Any]] = []
        timed_out = False
        error = ""

        async with semaphore:
            try:
                items = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._execute_route_sync,
                        route,
                        query,
                        collection_name,
                        top_k,
                    ),
                    timeout=self.query_timeout_seconds,
                )
            except asyncio.TimeoutError:
                timed_out = True
                items = []
            except Exception as exc:  # pragma: no cover - defensive safety path.
                error = str(exc)
                items = []

        duration_ms = int((time.perf_counter() - started) * 1000)
        return {
            "route": route,
            "query": query,
            "duration_ms": duration_ms,
            "timed_out": timed_out,
            "error": error,
            "returned": len(items),
            "items": items,
        }

    def _execute_route_sync(
        self,
        route: str,
        query: str,
        collection_name: str,
        top_k: int,
    ) -> list[Dict[str, Any]]:
        if route == "dense":
            embedding = self.embedding_builder(query, self.repository.embedding_dim)
            rows = self.repository.dense_search(
                collection_name=collection_name,
                query_embedding=embedding,
                query_text=query,
                top_k=top_k,
                chunk_type=None,
            )
            return [self._normalize_route_item(row, route, collection_name) for row in rows]

        if route == "table":
            rows = self.repository.table_search(
                collection_name=collection_name,
                query_text=query,
                top_k=top_k,
            )
            return [self._normalize_route_item(row, route, collection_name) for row in rows]

        rows = self.repository.keyword_search(
            collection_name=collection_name,
            query_text=query,
            top_k=top_k,
            chunk_type=None,
            table_only=False,
        )
        return [self._normalize_route_item(row, route, collection_name) for row in rows]

    def _normalize_route_item(self, item: Dict[str, Any], route: str, collection_name: str) -> Dict[str, Any]:
        payload = dict(item)
        payload.setdefault("collection_name", collection_name)
        payload.setdefault("chunk_id", str(payload.get("chunk_id") or ""))
        payload.setdefault("raw_doc", str(payload.get("raw_doc") or payload.get("content") or ""))
        payload.setdefault("source_channels", [])

        dense_score = _safe_float(payload.get("dense_score") or payload.get("similarity") or payload.get("score"))
        sparse_score = _safe_float(payload.get("bm25_score") or payload.get("score"))

        if route == "dense":
            payload["dense_score"] = _clip01(dense_score)
        elif route == "bm25":
            payload["bm25_score"] = _clip01(sparse_score)
        elif route == "table":
            payload["bm25_score"] = _clip01(sparse_score)
            payload["table_route_score"] = _clip01(sparse_score)
            payload["chunk_type"] = str(payload.get("chunk_type") or "table")

        channels = list(payload.get("source_channels") or [])
        if route not in channels:
            channels.append(route)
        payload["source_channels"] = channels
        return payload

    def _merge_candidates(self, task_results: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}

        for task in task_results:
            route = task.get("route")
            for item in task.get("items") or []:
                chunk_id = str(item.get("chunk_id") or "").strip()
                if not chunk_id:
                    basis = str(item.get("raw_doc") or "")
                    chunk_id = f"anon-{abs(hash((basis, route)))}"

                current = merged.get(chunk_id)
                if current is None:
                    current = dict(item)
                    current["chunk_id"] = chunk_id
                    current.setdefault("dense_score", 0.0)
                    current.setdefault("bm25_score", 0.0)
                    current.setdefault("table_route_score", 0.0)
                    current.setdefault("source_channels", list(item.get("source_channels") or []))
                    merged[chunk_id] = current
                else:
                    for key, value in item.items():
                        if current.get(key) in (None, "") and value not in (None, ""):
                            current[key] = value

                    current["dense_score"] = max(_safe_float(current.get("dense_score")), _safe_float(item.get("dense_score")))
                    current["bm25_score"] = max(_safe_float(current.get("bm25_score")), _safe_float(item.get("bm25_score")))
                    current["table_route_score"] = max(
                        _safe_float(current.get("table_route_score")),
                        _safe_float(item.get("table_route_score")),
                    )
                    existing_channels = list(current.get("source_channels") or [])
                    for channel in list(item.get("source_channels") or []):
                        if channel not in existing_channels:
                            existing_channels.append(channel)
                    current["source_channels"] = existing_channels

                if route == "dense":
                    current["dense_score"] = max(_safe_float(current.get("dense_score")), _safe_float(item.get("dense_score")))
                elif route == "table":
                    current["table_route_score"] = max(
                        _safe_float(current.get("table_route_score")),
                        _safe_float(item.get("table_route_score") or item.get("bm25_score")),
                    )
                else:
                    current["bm25_score"] = max(_safe_float(current.get("bm25_score")), _safe_float(item.get("bm25_score")))

        ranked = list(merged.values())
        for row in ranked:
            row["retrieval_score"] = max(
                _safe_float(row.get("dense_score")),
                _safe_float(row.get("bm25_score")),
                _safe_float(row.get("table_route_score")),
            )

        ranked.sort(key=lambda item: _safe_float(item.get("retrieval_score")), reverse=True)
        return ranked

