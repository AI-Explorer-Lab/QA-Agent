from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .cross_encoder import TransformersCrossEncoderScorer
from .sparse_retriever import coarse_tokenize


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


def _normalize_scores(values: Sequence[float]) -> List[float]:
    if not values:
        return []

    maximum = max(values)
    minimum = min(values)
    if maximum <= 0.0:
        return [0.0 for _ in values]
    if maximum == minimum:
        return [1.0 if value > 0 else 0.0 for value in values]

    return [_clip01((value - minimum) / (maximum - minimum)) if value > 0 else 0.0 for value in values]


def _token_overlap_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    tokens = set(coarse_tokenize(text))
    if not tokens:
        return 0.0

    overlap = len(query_tokens & tokens)
    if overlap <= 0:
        return 0.0

    precision = overlap / max(1, len(query_tokens))
    recall = overlap / max(1, len(tokens))
    return _clip01(0.7 * precision + 0.3 * recall)


def _extract_page_hint(query: str) -> int | None:
    text = str(query or "").lower()
    pattern = re.search(r"(?:page|p|第)\s*(\d{1,4})", text)
    if not pattern:
        return None
    try:
        return int(pattern.group(1))
    except Exception:
        return None


def _metadata_boost(query_tokens: set[str], page_hint: int | None, candidate: Dict[str, Any]) -> float:
    score = 0.0
    score += 0.35 * _token_overlap_score(query_tokens, str(candidate.get("heading_path") or ""))
    score += 0.15 * _token_overlap_score(query_tokens, str(candidate.get("level1_title") or ""))
    score += 0.15 * _token_overlap_score(query_tokens, str(candidate.get("level2_title") or ""))
    score += 0.15 * _token_overlap_score(query_tokens, str(candidate.get("level3_title") or ""))
    score += 0.20 * _token_overlap_score(query_tokens, str(candidate.get("doc_source") or ""))

    if page_hint is not None:
        page_idx = candidate.get("page_idx")
        try:
            page_value = int(page_idx)
        except Exception:
            page_value = None
        if page_value is not None and page_value in {page_hint, page_hint - 1, page_hint + 1}:
            score += 0.20

    return _clip01(score)


def _table_boost(query_tokens: set[str], candidate: Dict[str, Any], query_type: str) -> float:
    if str(query_type or "") != "table_qa":
        return 0.0

    score = 0.0
    chunk_type = str(candidate.get("chunk_type") or "").lower()
    if chunk_type == "table":
        score += 0.60

    score += 0.25 * _token_overlap_score(query_tokens, str(candidate.get("table_header_text") or ""))
    score += 0.15 * _token_overlap_score(query_tokens, str(candidate.get("table_context_text") or ""))
    return _clip01(score)


def _candidate_id(candidate: Dict[str, Any]) -> str:
    chunk_id = str(candidate.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id
    raw_doc = str(candidate.get("raw_doc") or "")
    return f"anon-{abs(hash(raw_doc))}"


def _candidate_pair_text(candidate: Dict[str, Any]) -> str:
    parts = [
        str(candidate.get("heading_path") or ""),
        str(candidate.get("level1_title") or ""),
        str(candidate.get("level2_title") or ""),
        str(candidate.get("level3_title") or ""),
        str(candidate.get("doc_source") or ""),
        str(candidate.get("table_header_text") or ""),
        str(candidate.get("table_context_text") or ""),
        str(candidate.get("raw_doc") or candidate.get("content") or ""),
    ]
    cleaned: List[str] = []
    seen: set[str] = set()
    for part in parts:
        value = re.sub(r"\s+", " ", str(part or "")).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return "\n".join(cleaned)


def _normalize_exact_text(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)
    return normalized


def _near_duplicate_score(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    containment = intersection / max(1, min(len(tokens_a), len(tokens_b)))
    jaccard = intersection / max(1, union)
    return max(containment, jaccard)


class TwoStageHybridReranker:
    def __init__(
        self,
        dense_weight: float = 0.50,
        bm25_weight: float = 0.35,
        metadata_boost_weight: float = 0.10,
        table_boost_weight: float = 0.05,
        near_duplicate_threshold: float = 0.90,
        table_evidence_quota: int = 2,
        cross_encoder_enabled: bool = True,
        cross_encoder_model: str = "BAAI/bge-reranker-base",
        cross_encoder_candidate_pool: int = 30,
        cross_encoder_batch_size: int = 8,
        cross_encoder_max_length: int = 512,
        cross_encoder_local_files_only: bool = False,
        cross_encoder_scorer: Any | None = None,
    ) -> None:
        self.dense_weight = float(dense_weight)
        self.bm25_weight = float(bm25_weight)
        self.metadata_boost_weight = float(metadata_boost_weight)
        self.table_boost_weight = float(table_boost_weight)
        self.near_duplicate_threshold = float(near_duplicate_threshold)
        self.table_evidence_quota = max(0, int(table_evidence_quota))
        self.cross_encoder_enabled = bool(cross_encoder_enabled)
        self.cross_encoder_model = str(cross_encoder_model or "BAAI/bge-reranker-base")
        self.cross_encoder_candidate_pool = max(1, int(cross_encoder_candidate_pool))
        self.cross_encoder_batch_size = max(1, int(cross_encoder_batch_size))
        self.cross_encoder_max_length = max(16, int(cross_encoder_max_length))
        self.cross_encoder_local_files_only = bool(cross_encoder_local_files_only)
        self._cross_encoder_scorer = cross_encoder_scorer
        self._cross_encoder_load_failed = ""

    def rerank(
        self,
        query: str,
        candidates: Iterable[Dict[str, Any]],
        top_k: int,
        query_type: str,
        table_evidence_quota: int | None = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        rows = [dict(item) for item in candidates if isinstance(item, dict)]
        if not rows:
            return [], self._empty_trace(top_k, query_type, table_evidence_quota)

        limit = max(1, int(top_k))
        quota = self.table_evidence_quota if table_evidence_quota is None else max(0, int(table_evidence_quota))
        query_tokens = set(coarse_tokenize(query))
        page_hint = _extract_page_hint(query)

        dense_values = [_safe_float(row.get("dense_score") or row.get("similarity") or 0.0) for row in rows]
        bm25_values = [_safe_float(row.get("bm25_score") or 0.0) for row in rows]
        dense_norm = _normalize_scores(dense_values)
        bm25_norm = _normalize_scores(bm25_values)

        scored: List[Dict[str, Any]] = []
        for index, row in enumerate(rows):
            payload = dict(row)
            payload.setdefault("chunk_id", _candidate_id(payload))
            payload.setdefault("source_channels", list(payload.get("source_channels") or []))

            payload["dense_score"] = _clip01(dense_norm[index])
            payload["bm25_score"] = _clip01(bm25_norm[index])
            payload["metadata_boost"] = _metadata_boost(query_tokens, page_hint, payload)
            payload["table_boost"] = _table_boost(query_tokens, payload, query_type)

            payload["final_score"] = _clip01(
                self.dense_weight * payload["dense_score"]
                + self.bm25_weight * payload["bm25_score"]
                + self.metadata_boost_weight * payload["metadata_boost"]
                + self.table_boost_weight * payload["table_boost"]
            )
            scored.append(payload)

        scored.sort(key=lambda item: _safe_float(item.get("final_score")), reverse=True)

        deduped: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_exact: set[str] = set()
        selected_token_sets: List[set[str]] = []

        for item in scored:
            cid = _candidate_id(item)
            if cid in seen_ids:
                continue

            normalized_text = _normalize_exact_text(str(item.get("raw_doc") or ""))
            if normalized_text and normalized_text in seen_exact:
                continue

            token_set = set(coarse_tokenize(str(item.get("raw_doc") or "")))
            near_duplicate = False
            if token_set:
                for existing in selected_token_sets:
                    if _near_duplicate_score(token_set, existing) >= self.near_duplicate_threshold:
                        near_duplicate = True
                        break
            if near_duplicate:
                continue

            deduped.append(item)
            seen_ids.add(cid)
            if normalized_text:
                seen_exact.add(normalized_text)
            selected_token_sets.append(token_set)

        candidate_pool_limit = max(limit, self.cross_encoder_candidate_pool)
        seed_pool = deduped[:limit]
        seed_pool, neighbor_supplemented = self._supplement_neighbors(seed_pool, deduped, limit)
        light_pool = self._merge_unique(seed_pool, deduped, candidate_pool_limit)
        light_pool = self._finalize_selection(light_pool, deduped, candidate_pool_limit, query_type, quota)
        selected, cross_encoder_trace = self._cross_encoder_rerank(query, light_pool, limit, query_type, quota)
        if not selected:
            selected = light_pool[:limit]

        trace = {
            "weights": {
                "dense_weight": self.dense_weight,
                "bm25_weight": self.bm25_weight,
                "metadata_boost_weight": self.metadata_boost_weight,
                "table_boost_weight": self.table_boost_weight,
            },
            "query_type": query_type,
            "input_candidates": len(rows),
            "after_near_duplicate": len(deduped),
            "light_candidate_pool_size": len(light_pool),
            "cross_encoder_candidate_pool": candidate_pool_limit,
            "cross_encoder": cross_encoder_trace,
            "table_evidence_quota": quota,
            "table_evidence_selected": sum(1 for row in selected if str(row.get("chunk_type") or "") == "table"),
            "neighbor_supplemented": neighbor_supplemented,
            "top": [
                {
                    "chunk_id": row.get("chunk_id"),
                    "final_score": row.get("final_score"),
                    "dense_score": row.get("dense_score"),
                    "bm25_score": row.get("bm25_score"),
                    "metadata_boost": row.get("metadata_boost"),
                    "table_boost": row.get("table_boost"),
                    "cross_encoder_score": row.get("cross_encoder_score"),
                }
                for row in selected[: min(len(selected), 10)]
            ],
        }

        return selected, trace

    def _empty_trace(self, top_k: int, query_type: str, table_evidence_quota: int | None) -> Dict[str, Any]:
        quota = self.table_evidence_quota if table_evidence_quota is None else max(0, int(table_evidence_quota))
        return {
            "weights": {
                "dense_weight": self.dense_weight,
                "bm25_weight": self.bm25_weight,
                "metadata_boost_weight": self.metadata_boost_weight,
                "table_boost_weight": self.table_boost_weight,
            },
            "query_type": query_type,
            "input_candidates": 0,
            "after_near_duplicate": 0,
            "light_candidate_pool_size": 0,
            "cross_encoder_candidate_pool": self.cross_encoder_candidate_pool,
            "cross_encoder": {"status": "skipped", "reason": "empty_candidates"},
            "table_evidence_quota": quota,
            "table_evidence_selected": 0,
            "neighbor_supplemented": 0,
            "top": [],
            "top_k": max(1, int(top_k)),
        }

    def _cross_encoder_rerank(
        self,
        query: str,
        light_pool: Sequence[Dict[str, Any]],
        limit: int,
        query_type: str,
        quota: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.cross_encoder_enabled:
            return list(light_pool[:limit]), {"status": "disabled"}
        if not light_pool:
            return [], {"status": "skipped", "reason": "empty_candidate_pool"}

        scorer = self._get_cross_encoder_scorer()
        if scorer is None:
            reason = self._cross_encoder_load_failed or "scorer_unavailable"
            return list(light_pool[:limit]), {"status": "fallback", "reason": reason, "model": self.cross_encoder_model}

        texts = [_candidate_pair_text(item) for item in light_pool]
        try:
            scores, score_trace = scorer.score(query, texts)
        except Exception as exc:
            score_trace = {"status": "fallback", "reason": f"{type(exc).__name__}: {exc}"[:500]}
            scores = []

        if not scores or len(scores) != len(light_pool):
            trace = dict(score_trace or {})
            trace.setdefault("status", "fallback")
            trace.setdefault("reason", "invalid_score_count")
            trace.setdefault("model", self.cross_encoder_model)
            return list(light_pool[:limit]), trace

        scored_rows: List[Dict[str, Any]] = []
        for row, score in zip(light_pool, scores):
            payload = dict(row)
            payload["light_final_score"] = _safe_float(payload.get("final_score"))
            payload["cross_encoder_score"] = float(score)
            payload["final_score"] = float(score)
            scored_rows.append(payload)

        scored_rows.sort(key=lambda item: _safe_float(item.get("cross_encoder_score")), reverse=True)
        selected = self._finalize_selection(scored_rows[:limit], scored_rows, limit, query_type, quota)
        trace = dict(score_trace or {})
        trace.setdefault("status", "applied")
        trace.setdefault("model", self.cross_encoder_model)
        trace["input_candidates"] = len(light_pool)
        trace["selected"] = len(selected)
        return selected, trace

    def _get_cross_encoder_scorer(self) -> Any | None:
        if self._cross_encoder_scorer is not None:
            return self._cross_encoder_scorer
        if self._cross_encoder_load_failed:
            return None

        scorer = TransformersCrossEncoderScorer(
            model_name=self.cross_encoder_model,
            batch_size=self.cross_encoder_batch_size,
            max_length=self.cross_encoder_max_length,
            local_files_only=self.cross_encoder_local_files_only,
        )
        if not scorer._load():
            self._cross_encoder_load_failed = scorer.last_error or "model_load_failed"
            return None
        self._cross_encoder_scorer = scorer
        return self._cross_encoder_scorer

    def _merge_unique(
        self,
        preferred: Sequence[Dict[str, Any]],
        ranked: Sequence[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for source in (preferred, ranked):
            for item in source:
                cid = _candidate_id(item)
                if cid in seen:
                    continue
                merged.append(dict(item))
                seen.add(cid)
                if len(merged) >= limit:
                    return merged
        return merged

    def _finalize_selection(
        self,
        selected: List[Dict[str, Any]],
        ranked: Sequence[Dict[str, Any]],
        limit: int,
        query_type: str,
        quota: int,
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for source in (selected, ranked):
            for item in source:
                cid = _candidate_id(item)
                if cid in seen:
                    continue
                merged.append(item)
                seen.add(cid)

        merged.sort(key=lambda row: _safe_float(row.get("final_score")), reverse=True)
        if str(query_type or "") != "table_qa" or quota <= 0:
            return merged[:limit]

        table_rows = [row for row in merged if str(row.get("chunk_type") or "") == "table"]
        required_tables = min(quota, limit, len(table_rows))

        final_rows: List[Dict[str, Any]] = []
        used: set[str] = set()
        for row in table_rows[:required_tables]:
            cid = _candidate_id(row)
            final_rows.append(row)
            used.add(cid)

        for row in merged:
            cid = _candidate_id(row)
            if cid in used:
                continue
            final_rows.append(row)
            used.add(cid)
            if len(final_rows) >= limit:
                break

        return final_rows[:limit]

    def _supplement_neighbors(
        self,
        selected: List[Dict[str, Any]],
        ranked: Sequence[Dict[str, Any]],
        top_k: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        if not selected or not ranked:
            return selected, 0

        chosen: Dict[str, Dict[str, Any]] = {_candidate_id(item): dict(item) for item in selected}
        supplemented = 0
        max_pool_size = max(2 * max(1, int(top_k)), max(1, int(top_k)) + 2)

        for item in list(selected):
            if len(chosen) >= max_pool_size:
                break

            page_idx = item.get("page_idx")
            chunk_index = item.get("chunk_index")
            doc_ref = str(item.get("doc_id") or item.get("doc_source") or "")

            try:
                current_page = int(page_idx)
                current_chunk = int(chunk_index)
            except Exception:
                continue

            neighbor: Dict[str, Any] | None = None
            neighbor_score = -1.0
            for candidate in ranked:
                cid = _candidate_id(candidate)
                if cid in chosen:
                    continue
                candidate_doc_ref = str(candidate.get("doc_id") or candidate.get("doc_source") or "")
                if candidate_doc_ref != doc_ref:
                    continue

                try:
                    candidate_page = int(candidate.get("page_idx"))
                    candidate_chunk = int(candidate.get("chunk_index"))
                except Exception:
                    continue

                if candidate_page != current_page:
                    continue
                if abs(candidate_chunk - current_chunk) > 1:
                    continue

                score = _safe_float(candidate.get("final_score"))
                if score > neighbor_score:
                    neighbor = dict(candidate)
                    neighbor_score = score

            if neighbor is None:
                continue

            neighbor["final_score"] = _clip01(_safe_float(neighbor.get("final_score")) + 0.02)
            chosen[_candidate_id(neighbor)] = neighbor
            supplemented += 1

        final_rows = list(chosen.values())
        final_rows.sort(key=lambda item: _safe_float(item.get("final_score")), reverse=True)
        return final_rows, supplemented
