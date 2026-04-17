from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Set


def _safe_similarity(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _normalize_exact_text(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)
    return normalized


def _tokenize_text(text: str) -> Set[str]:
    tokens = re.split(r"[^\w\u4e00-\u9fff]+", str(text or "").lower())
    return {token for token in tokens if token}


def _near_duplicate_score(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    min_len = min(len(tokens_a), len(tokens_b))
    jaccard = intersection / max(1, union)
    containment = intersection / max(1, min_len)
    return max(jaccard, containment)


def dedupe_ranked_documents(
    docs: Sequence[Dict[str, Any]],
    k: int,
    near_duplicate_threshold: float = 0.9,
) -> List[Dict[str, Any]]:
    """
    Final-stage dedupe with backfill:
    1) sort by similarity desc
    2) drop exact duplicates (chunk_id / normalized text)
    3) drop near-duplicates by token overlap
    4) continue scanning and backfill from lower-ranked docs until top-k is filled
    """
    limit = max(1, int(k))
    ranked = sorted(list(docs), key=lambda item: _safe_similarity(item.get("similarity", 0.0)), reverse=True)

    selected: List[Dict[str, Any]] = []
    seen_chunk_ids: Set[str] = set()
    seen_exact_text: Set[str] = set()
    selected_token_sets: List[Set[str]] = []

    for doc in ranked:
        chunk_id = str(doc.get("chunk_id") or "").strip()
        if chunk_id and chunk_id in seen_chunk_ids:
            continue

        raw_doc = str(doc.get("raw_doc", "") or "")
        exact_key = _normalize_exact_text(raw_doc)
        if exact_key and exact_key in seen_exact_text:
            continue

        token_set = _tokenize_text(raw_doc)
        is_near_duplicate = False
        if token_set:
            for existing in selected_token_sets:
                if _near_duplicate_score(token_set, existing) >= near_duplicate_threshold:
                    is_near_duplicate = True
                    break
        if is_near_duplicate:
            continue

        selected.append(doc)
        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        if exact_key:
            seen_exact_text.add(exact_key)
        selected_token_sets.append(token_set)

        if len(selected) >= limit:
            break

    return selected
