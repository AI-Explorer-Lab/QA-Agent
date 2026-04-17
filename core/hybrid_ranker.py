from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from core.retrieval_deduper import dedupe_ranked_documents


_CN_OR_WORD_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9_]+", flags=re.IGNORECASE)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _extract_token_sequence(text: str) -> List[str]:
    value = str(text or "").strip().lower()
    if not value:
        return []

    tokens: List[str] = []
    segments = _CN_OR_WORD_PATTERN.findall(value)

    for segment in segments:
        if not segment:
            continue

        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            if len(segment) <= 3:
                tokens.append(segment)
                continue

            # Generate Chinese n-gram tokens to improve sparse recall.
            for n in (2, 3, 4):
                if len(segment) < n:
                    continue
                for i in range(len(segment) - n + 1):
                    tokens.append(segment[i : i + n])
            tokens.append(segment)
            continue

        tokens.append(segment)

    return tokens


def _extract_tokens(text: str) -> Set[str]:
    return set(_extract_token_sequence(text))


def _token_overlap_score(query_tokens: Set[str], text_tokens: Set[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    hit = len(query_tokens & text_tokens)
    if hit <= 0:
        return 0.0
    precision_like = hit / max(1, len(query_tokens))
    recall_like = hit / max(1, len(text_tokens))
    return _clip01(0.7 * precision_like + 0.3 * recall_like)


def _bm25_scores(
    query_terms: Sequence[str],
    docs_tokens: Sequence[Sequence[str]],
    k1: float,
    b: float,
) -> List[float]:
    if not query_terms or not docs_tokens:
        return [0.0 for _ in docs_tokens]

    corpus_size = len(docs_tokens)
    if corpus_size <= 0:
        return [0.0 for _ in docs_tokens]

    avg_doc_len = sum(len(tokens) for tokens in docs_tokens) / max(1, corpus_size)
    avg_doc_len = max(avg_doc_len, 1e-9)

    document_frequency: Counter[str] = Counter()
    for tokens in docs_tokens:
        document_frequency.update(set(tokens))

    unique_query_terms = list(dict.fromkeys(term for term in query_terms if term))
    scores: List[float] = []
    for tokens in docs_tokens:
        if not tokens:
            scores.append(0.0)
            continue

        term_frequency = Counter(tokens)
        doc_len = max(1, len(tokens))
        score = 0.0

        for term in unique_query_terms:
            tf = float(term_frequency.get(term, 0))
            if tf <= 0:
                continue

            df = float(document_frequency.get(term, 0))
            # Okapi BM25 idf with +1 smoothing for numerical stability.
            idf = math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
            if denom <= 0:
                continue
            score += idf * ((tf * (k1 + 1.0)) / denom)

        scores.append(float(score))

    return scores


def _normalize_bm25_scores(raw_scores: Sequence[float]) -> List[float]:
    if not raw_scores:
        return []

    max_score = max(raw_scores)
    if max_score <= 0:
        return [0.0 for _ in raw_scores]

    return [_clip01(score / max_score) if score > 0 else 0.0 for score in raw_scores]


def _is_table_intent(query: str) -> bool:
    q = str(query or "").lower()
    if not q:
        return False

    lexical_hints = [
        "table",
        "row",
        "column",
        "metric",
        "ratio",
        "同比",
        "环比",
        "增长",
        "下降",
        "金额",
        "占比",
        "比例",
        "指标",
        "单位",
        "明细",
        "数据",
        "表格",
        "表",
    ]
    if any(hint in q for hint in lexical_hints):
        return True

    if re.search(r"\b20\d{2}\b", q):
        return True
    if re.search(r"\d+(?:\.\d+)?\s*%", q):
        return True
    if re.search(r"\d+(?:\.\d+)?\s*(?:万元|亿元|元|万|亿)", q):
        return True

    return False


def _dense_normalized(doc: Dict[str, Any]) -> float:
    return _clip01(_safe_float(doc.get("similarity", 0.0)))


def _field_hit_score(query_tokens: Set[str], doc: Dict[str, Any]) -> float:
    fields_with_weight: Sequence[Tuple[str, float]] = (
        ("heading_path", 0.30),
        ("level1_title", 0.08),
        ("level2_title", 0.08),
        ("level3_title", 0.08),
        ("table_context_text", 0.20),
        ("table_header_text", 0.14),
        ("raw_doc", 0.12),
    )
    score = 0.0
    for field, weight in fields_with_weight:
        tokens = _extract_tokens(str(doc.get(field, "")))
        score += weight * _token_overlap_score(query_tokens, tokens)
    return _clip01(score)


def _table_bonus(doc: Dict[str, Any]) -> float:
    chunk_type = str(doc.get("chunk_type", "")).strip().lower()
    if chunk_type != "table":
        raw_doc = str(doc.get("raw_doc", ""))
        text_len = len(raw_doc.strip())
        return 0.15 if text_len >= 40 else 0.0

    has_context = bool(doc.get("table_context_text"))
    has_header = bool(doc.get("table_header_text"))

    if has_context and has_header:
        return 1.0
    if has_context:
        return 0.85
    return 0.45


def keyword_rank_documents(
    query: str,
    docs: Iterable[Dict[str, Any]],
    k: int,
    min_score: float = 0.05,
    algorithm: str = "overlap",
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> List[Dict[str, Any]]:
    algo = str(algorithm or "overlap").strip().lower()
    if algo not in {"overlap", "bm25"}:
        algo = "overlap"

    candidates: List[Tuple[Dict[str, Any], str]] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue

        searchable = " ".join(
            [
                str(doc.get("raw_doc", "")),
                str(doc.get("heading_path", "")),
                str(doc.get("level1_title", "")),
                str(doc.get("level2_title", "")),
                str(doc.get("level3_title", "")),
                str(doc.get("table_context_text", "")),
                str(doc.get("table_header_text", "")),
            ]
        ).strip()
        if not searchable:
            continue

        candidates.append((doc, searchable))

    if not candidates:
        return []

    scored: List[Dict[str, Any]] = []
    if algo == "bm25":
        query_terms = _extract_token_sequence(query)
        if not query_terms:
            return []

        docs_tokens = [_extract_token_sequence(searchable) for _, searchable in candidates]
        raw_scores = _bm25_scores(
            query_terms=query_terms,
            docs_tokens=docs_tokens,
            k1=max(0.1, float(bm25_k1)),
            b=max(0.0, min(1.0, float(bm25_b))),
        )
        normalized_scores = _normalize_bm25_scores(raw_scores)

        for (doc, _), sparse in zip(candidates, normalized_scores):
            if sparse < min_score:
                continue
            item = dict(doc)
            item["sparse_score"] = _clip01(sparse)
            scored.append(item)
    else:
        query_tokens = _extract_tokens(query)
        if not query_tokens:
            return []

        for doc, searchable in candidates:
            sparse = _token_overlap_score(query_tokens, _extract_tokens(searchable))
            if sparse < min_score:
                continue

            item = dict(doc)
            item["sparse_score"] = _clip01(sparse)
            scored.append(item)

    scored.sort(key=lambda item: _safe_float(item.get("sparse_score", 0.0)), reverse=True)
    return scored[: max(1, int(k))]


def _is_complete_table_context(doc: Dict[str, Any]) -> bool:
    return str(doc.get("chunk_type", "")).strip().lower() == "table" and bool(doc.get("table_context_text"))


def _doc_identity(doc: Dict[str, Any]) -> str:
    chunk_id = str(doc.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id
    fallback = str(doc.get("raw_doc") or "").strip()
    return fallback[:200]


def fuse_hybrid_results(
    query: str,
    dense_docs: Sequence[Dict[str, Any]],
    sparse_docs: Sequence[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    limit = max(1, int(k))
    query_tokens = _extract_tokens(query)

    merged: Dict[str, Dict[str, Any]] = {}
    for doc in list(dense_docs) + list(sparse_docs):
        if not isinstance(doc, dict):
            continue
        key = _doc_identity(doc)
        if not key:
            continue

        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(doc)
            continue

        for field in (
            "raw_doc",
            "doc_id",
            "doc_source",
            "chunk_id",
            "chunk_type",
            "chunk_index",
            "heading_path",
            "level1_title",
            "level2_title",
            "level3_title",
            "table_id",
            "sub_table_id",
            "sub_table_index",
            "table_id_subtable_count",
            "table_context_text",
            "table_header_text",
        ):
            if existing.get(field) in {None, ""} and doc.get(field) not in {None, ""}:
                existing[field] = doc.get(field)

        existing["sparse_score"] = max(
            _safe_float(existing.get("sparse_score", 0.0)),
            _safe_float(doc.get("sparse_score", 0.0)),
        )
        existing["similarity"] = max(
            _safe_float(existing.get("similarity", 0.0)),
            _safe_float(doc.get("similarity", 0.0)),
        )

    if not merged:
        return []

    table_intent = _is_table_intent(query)
    if table_intent:
        # dense, sparse, field, table_bonus
        weights = (0.40, 0.35, 0.15, 0.10)
    else:
        weights = (0.55, 0.25, 0.15, 0.05)

    ranked: List[Dict[str, Any]] = []
    for item in merged.values():
        dense_score = _dense_normalized(item)
        sparse_score = _clip01(_safe_float(item.get("sparse_score", 0.0)))
        field_score = _field_hit_score(query_tokens, item)
        table_score = _table_bonus(item)

        final = (
            weights[0] * dense_score
            + weights[1] * sparse_score
            + weights[2] * field_score
            + weights[3] * table_score
        )

        item["dense_score"] = dense_score
        item["sparse_score"] = sparse_score
        item["field_hit_score"] = field_score
        item["table_bonus"] = table_score
        item["similarity"] = _clip01(final)
        ranked.append(item)

    ranked.sort(key=lambda doc: _safe_float(doc.get("similarity", 0.0)), reverse=True)
    deduped = dedupe_ranked_documents(ranked, k=max(limit * 4, 20))
    if not deduped:
        return []

    table_quota = _safe_int(os.getenv("HYBRID_TABLE_QUOTA", "2"), 2)
    table_quota = max(0, min(table_quota, limit))
    table_floor = _safe_float(os.getenv("HYBRID_TABLE_SCORE_FLOOR", "0.25"), 0.25)

    top_score = _safe_float(deduped[0].get("similarity", 0.0), 0.0)
    adaptive_threshold = max(table_floor, top_score * 0.45)

    selected: List[Dict[str, Any]] = []
    selected_keys: Set[str] = set()

    for doc in deduped:
        if len(selected) >= table_quota:
            break
        if not _is_complete_table_context(doc):
            continue
        if _safe_float(doc.get("similarity", 0.0)) < adaptive_threshold:
            continue
        key = _doc_identity(doc)
        if key in selected_keys:
            continue
        selected.append(doc)
        selected_keys.add(key)

    # Fallback: if quota is not satisfied due score threshold, backfill with
    # best remaining complete table-context chunks.
    if len(selected) < table_quota:
        for doc in deduped:
            if len(selected) >= table_quota:
                break
            if not _is_complete_table_context(doc):
                continue
            key = _doc_identity(doc)
            if key in selected_keys:
                continue
            selected.append(doc)
            selected_keys.add(key)

    for doc in deduped:
        if len(selected) >= limit:
            break
        key = _doc_identity(doc)
        if key in selected_keys:
            continue
        selected.append(doc)
        selected_keys.add(key)

    return selected[:limit]
