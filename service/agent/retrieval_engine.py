from __future__ import annotations

import asyncio
import importlib
import inspect
import math
import re
from copy import deepcopy
from typing import Any, Callable, Dict, List, Sequence

from utils.async_utils import bounded_gather
from utils.content_normalizer import normalize_whitespace

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall((text or "").lower())
    if tokens:
        return tokens
    return list((text or "").lower())


def _clip(text: str, size: int = 180) -> str:
    value = normalize_whitespace(text, preserve_newlines=False)
    if len(value) <= size:
        return value
    return value[: size - 3] + "..."


async def _invoke_callable(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(func)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            result = func(**kwargs)
        else:
            accepted = {name for name in signature.parameters.keys()}
            filtered = {key: value for key, value in kwargs.items() if key in accepted}
            result = func(**filtered)
    except (TypeError, ValueError):
        return None

    if inspect.isawaitable(result):
        return await result
    return result


class RetrievalEngine:
    async def _try_external_parallel_retrieval(self, **kwargs: Any) -> Dict[str, Any] | None:
        candidates = [
            ("service.retrieval.parallel_query_executor", ("parallel_hybrid_retrieval", "execute_parallel_hybrid_retrieval", "run_parallel_hybrid_retrieval")),
            ("service.retrieval.hybrid_retriever", ("parallel_hybrid_retrieval", "hybrid_retrieve", "retrieve")),
        ]

        for module_name, function_names in candidates:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue

            for function_name in function_names:
                fn = getattr(module, function_name, None)
                if not callable(fn):
                    continue

                try:
                    result = await _invoke_callable(fn, kwargs)
                except Exception:
                    continue

                if result is None:
                    continue

                if isinstance(result, dict):
                    if "candidates" in result:
                        result.setdefault("backend", f"external:{module_name}.{function_name}")
                        return result
                    if "evidence" in result:
                        return {
                            "backend": f"external:{module_name}.{function_name}",
                            "candidates": result.get("evidence", []),
                            "retrieval_trace": result,
                        }
                if isinstance(result, list):
                    return {
                        "backend": f"external:{module_name}.{function_name}",
                        "candidates": result,
                        "retrieval_trace": {
                            "backend": f"external:{module_name}.{function_name}",
                            "candidate_count": len(result),
                        },
                    }
        return None

    def _prepare_chunks(self, raw_chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for raw in raw_chunks:
            content = normalize_whitespace(raw.get("content") or raw.get("raw_doc") or "", preserve_newlines=False)
            if not content:
                continue
            chunks.append(
                {
                    "chunk_id": raw.get("chunk_id") or f"chunk_{len(chunks)+1}",
                    "doc_id": raw.get("doc_id") or raw.get("doc_source") or "unknown_doc",
                    "doc_source": raw.get("doc_source") or raw.get("doc_id") or "unknown_source",
                    "page_idx": int(raw.get("page_idx", 0) or 0),
                    "heading_path": raw.get("heading_path") or "front_matter",
                    "chunk_type": raw.get("chunk_type") or "text",
                    "content": content,
                    "table_header_text": raw.get("table_header_text") or "",
                    "table_context_text": raw.get("table_context_text") or "",
                }
            )
        return chunks

    def _dense_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        q_set = set(query_tokens)
        d_set = set(doc_tokens)
        overlap = len(q_set & d_set)
        return overlap / math.sqrt(max(1, len(q_set)) * max(1, len(d_set)))

    def _build_idf(self, chunks_tokens: List[List[str]]) -> Dict[str, float]:
        doc_count = max(1, len(chunks_tokens))
        df: Dict[str, int] = {}
        for tokens in chunks_tokens:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        idf: Dict[str, float] = {}
        for token, count in df.items():
            idf[token] = math.log((doc_count - count + 0.5) / (count + 0.5) + 1.0)
        return idf

    def _bm25_score(
        self,
        query_tokens: List[str],
        doc_tokens: List[str],
        idf: Dict[str, float],
        avg_doc_len: float,
    ) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        k1 = 1.2
        b = 0.75
        doc_len = len(doc_tokens)
        freqs: Dict[str, int] = {}
        for token in doc_tokens:
            freqs[token] = freqs.get(token, 0) + 1

        score = 0.0
        for token in query_tokens:
            tf = freqs.get(token, 0)
            if tf <= 0:
                continue
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / max(1.0, avg_doc_len)))
            score += idf.get(token, 0.0) * (numerator / max(1e-6, denominator))
        return score

    def _metadata_boost(self, query: str, chunk: Dict[str, Any]) -> float:
        score = 0.0
        lowered = query.lower()
        heading = str(chunk.get("heading_path") or "").lower()
        doc_source = str(chunk.get("doc_source") or "").lower()

        if heading and any(token in heading for token in _tokenize(lowered)):
            score += 0.15
        if doc_source and any(token in doc_source for token in _tokenize(lowered)):
            score += 0.1
        if str(chunk.get("page_idx", "")) and str(chunk.get("page_idx")) in lowered:
            score += 0.05
        return score

    def _table_boost(self, query_type: str, query: str, chunk: Dict[str, Any]) -> float:
        if query_type != "table_qa":
            return 0.0
        if str(chunk.get("chunk_type", "")).lower() != "table":
            return 0.0

        score = 0.2
        query_tokens = set(_tokenize(query))
        table_text = " ".join(
            [
                str(chunk.get("table_header_text") or ""),
                str(chunk.get("table_context_text") or ""),
                str(chunk.get("content") or ""),
            ]
        ).lower()
        if query_tokens and table_text:
            overlap = len(query_tokens & set(_tokenize(table_text)))
            score += min(0.25, overlap * 0.05)
        return score

    async def _score_chunk(
        self,
        chunk: Dict[str, Any],
        query: str,
        query_type: str,
        query_tokens: List[str],
        idf: Dict[str, float],
        avg_doc_len: float,
    ) -> Dict[str, Any]:
        doc_tokens = _tokenize(chunk["content"])
        dense_score = self._dense_score(query_tokens, doc_tokens)
        bm25_score = self._bm25_score(query_tokens, doc_tokens, idf=idf, avg_doc_len=avg_doc_len)
        metadata_boost = self._metadata_boost(query, chunk)
        table_boost = self._table_boost(query_type, query, chunk)
        retrieval_score = dense_score + bm25_score + metadata_boost + table_boost

        result = deepcopy(chunk)
        result.update(
            {
                "dense_score": round(dense_score, 6),
                "bm25_score": round(bm25_score, 6),
                "metadata_boost": round(metadata_boost, 6),
                "table_boost": round(table_boost, 6),
                "retrieval_score": round(retrieval_score, 6),
                "matched_query": query,
            }
        )
        return result

    async def _fallback_parallel_hybrid_retrieval(
        self,
        question: str,
        expanded_queries: List[str],
        query_type: str,
        collection_name: str,
        top_k: int,
        session_service: Any,
        max_concurrency: int = 6,
    ) -> Dict[str, Any]:
        queries: List[str] = []
        for query in [question] + (expanded_queries or []):
            value = normalize_whitespace(query, preserve_newlines=False)
            if value and value not in queries:
                queries.append(value)

        chunks = self._prepare_chunks(session_service.get_collection_chunks(collection_name))
        if not chunks:
            return {
                "backend": "local_dev_fallback",
                "queries": queries,
                "candidates": [],
                "retrieval_trace": {
                    "backend": "local_dev_fallback",
                    "collection_name": collection_name,
                    "query_routes": [],
                    "candidate_count": 0,
                },
            }

        chunks_tokens = [_tokenize(chunk["content"]) for chunk in chunks]
        avg_doc_len = sum(len(tokens) for tokens in chunks_tokens) / max(1, len(chunks_tokens))
        idf = self._build_idf(chunks_tokens)

        top_n = max(top_k * 4, top_k)
        merged: Dict[str, Dict[str, Any]] = {}
        query_routes: List[Dict[str, Any]] = []

        for query in queries:
            query_tokens = _tokenize(query)
            tasks = [
                self._score_chunk(
                    chunk=chunk,
                    query=query,
                    query_type=query_type,
                    query_tokens=query_tokens,
                    idf=idf,
                    avg_doc_len=avg_doc_len,
                )
                for chunk in chunks
            ]
            scored_chunks = await bounded_gather(tasks, limit=max(1, int(max_concurrency)))
            ranked = sorted(scored_chunks, key=lambda item: item["retrieval_score"], reverse=True)[:top_n]

            query_routes.append(
                {
                    "query": query,
                    "top_hits": [
                        {
                            "chunk_id": row["chunk_id"],
                            "doc_source": row.get("doc_source"),
                            "retrieval_score": row["retrieval_score"],
                        }
                        for row in ranked[: min(5, len(ranked))]
                    ],
                }
            )

            for row in ranked:
                chunk_id = row["chunk_id"]
                if chunk_id not in merged:
                    merged[chunk_id] = deepcopy(row)
                    merged[chunk_id]["matched_queries"] = [query]
                    continue

                target = merged[chunk_id]
                target["dense_score"] = max(target.get("dense_score", 0.0), row.get("dense_score", 0.0))
                target["bm25_score"] = max(target.get("bm25_score", 0.0), row.get("bm25_score", 0.0))
                target["metadata_boost"] = max(target.get("metadata_boost", 0.0), row.get("metadata_boost", 0.0))
                target["table_boost"] = max(target.get("table_boost", 0.0), row.get("table_boost", 0.0))
                target["retrieval_score"] = max(target.get("retrieval_score", 0.0), row.get("retrieval_score", 0.0))
                target.setdefault("matched_queries", []).append(query)

        candidates = sorted(merged.values(), key=lambda item: item.get("retrieval_score", 0.0), reverse=True)
        return {
            "backend": "local_dev_fallback",
            "queries": queries,
            "candidates": candidates,
            "retrieval_trace": {
                "backend": "local_dev_fallback",
                "collection_name": collection_name,
                "query_routes": query_routes,
                "candidate_count": len(candidates),
            },
        }

    async def parallel_hybrid_retrieval(
        self,
        question: str,
        expanded_queries: List[str],
        query_type: str,
        collection_name: str,
        top_k: int,
        session_service: Any,
        max_concurrency: int = 6,
    ) -> Dict[str, Any]:
        external_result = await self._try_external_parallel_retrieval(
            question=question,
            expanded_queries=expanded_queries,
            query_type=query_type,
            collection_name=collection_name,
            top_k=top_k,
            session_service=session_service,
            max_concurrency=max_concurrency,
        )
        if external_result:
            trace = external_result.setdefault("retrieval_trace", {})
            trace.setdefault("backend", external_result.get("backend", "external"))
            return external_result

        return await self._fallback_parallel_hybrid_retrieval(
            question=question,
            expanded_queries=expanded_queries,
            query_type=query_type,
            collection_name=collection_name,
            top_k=top_k,
            session_service=session_service,
            max_concurrency=max_concurrency,
        )

    def _normalize_column(self, rows: List[Dict[str, Any]], key: str) -> None:
        max_value = max((float(item.get(key, 0.0) or 0.0) for item in rows), default=0.0)
        for row in rows:
            value = float(row.get(key, 0.0) or 0.0)
            row[f"{key}_norm"] = 0.0 if max_value <= 0 else value / max_value

    def _dedupe_near_duplicates(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for row in rows:
            signature = normalize_whitespace(row.get("content", ""), preserve_newlines=False)[:180].lower()
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(row)
        return deduped

    def _enforce_table_quota(self, rows: List[Dict[str, Any]], all_rows: List[Dict[str, Any]], quota: int) -> List[Dict[str, Any]]:
        if quota <= 0:
            return rows
        selected = list(rows)
        current_table = [row for row in selected if str(row.get("chunk_type", "")).lower() == "table"]
        if len(current_table) >= quota:
            return selected

        for candidate in all_rows:
            if str(candidate.get("chunk_type", "")).lower() != "table":
                continue
            if any(candidate.get("chunk_id") == row.get("chunk_id") for row in selected):
                continue
            selected.append(candidate)
            current_table.append(candidate)
            if len(current_table) >= quota:
                break
        return selected

    def two_stage_hybrid_rerank(
        self,
        candidates: List[Dict[str, Any]],
        query_type: str,
        top_k: int,
        table_evidence_quota: int = 2,
    ) -> Dict[str, Any]:
        rows = [deepcopy(item) for item in candidates]
        self._normalize_column(rows, "dense_score")
        self._normalize_column(rows, "bm25_score")
        self._normalize_column(rows, "metadata_boost")
        self._normalize_column(rows, "table_boost")

        for row in rows:
            final_score = (
                0.50 * row.get("dense_score_norm", 0.0)
                + 0.35 * row.get("bm25_score_norm", 0.0)
                + 0.10 * row.get("metadata_boost_norm", 0.0)
                + 0.05 * row.get("table_boost_norm", 0.0)
            )
            row["final_score"] = round(final_score, 6)

        rows = sorted(rows, key=lambda item: item.get("final_score", 0.0), reverse=True)
        rows = self._dedupe_near_duplicates(rows)

        selected = rows[: max(1, int(top_k))]
        if query_type == "table_qa":
            selected = self._enforce_table_quota(selected, rows, quota=max(0, int(table_evidence_quota)))
            selected = sorted(selected, key=lambda item: item.get("final_score", 0.0), reverse=True)[: max(1, int(top_k))]

        rerank_trace = {
            "algorithm": "two_stage_hybrid_rerank_fallback",
            "weights": {
                "dense": 0.50,
                "bm25": 0.35,
                "metadata": 0.10,
                "table": 0.05,
            },
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "top_candidates": [
                {
                    "chunk_id": row.get("chunk_id"),
                    "doc_source": row.get("doc_source"),
                    "chunk_type": row.get("chunk_type"),
                    "final_score": row.get("final_score", 0.0),
                    "snippet": _clip(row.get("content", ""), 120),
                }
                for row in selected
            ],
        }

        return {
            "evidence": selected,
            "rerank_trace": rerank_trace,
        }


_DEFAULT_RETRIEVAL_ENGINE = RetrievalEngine()


async def parallel_hybrid_retrieval(**kwargs: Any) -> Dict[str, Any]:
    return await _DEFAULT_RETRIEVAL_ENGINE.parallel_hybrid_retrieval(**kwargs)


def two_stage_hybrid_rerank(**kwargs: Any) -> Dict[str, Any]:
    return _DEFAULT_RETRIEVAL_ENGINE.two_stage_hybrid_rerank(**kwargs)
