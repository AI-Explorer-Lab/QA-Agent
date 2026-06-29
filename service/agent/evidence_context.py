from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence

from service.retrieval.sparse_retriever import coarse_tokenize
from utils.content_normalizer import normalize_whitespace


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.%+\-]*")
_CJK_SEQ_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


def _clean_text(value: Any, *, preserve_newlines: bool = False) -> str:
    return normalize_whitespace(str(value or ""), preserve_newlines=preserve_newlines).strip()


def _unique(items: Sequence[str], limit: int = 80) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def _cjk_ngrams(text: str) -> List[str]:
    grams: List[str] = []
    for sequence in _CJK_SEQ_RE.findall(text):
        if len(sequence) <= 8:
            grams.append(sequence)
        for width in (2, 3, 4):
            if len(sequence) < width:
                continue
            for index in range(len(sequence) - width + 1):
                grams.append(sequence[index : index + width])
    return grams


def build_query_terms(question: str, slots: Mapping[str, Any] | None = None) -> List[str]:
    """Extract terms that are useful for locating a focused snippet inside a chunk."""

    slot_values = slots or {}
    parts = [_clean_text(question)]
    for key in ("metric", "period", "target_statement", "scope", "table_name", "unit", "focus"):
        value = slot_values.get(key)
        if value:
            parts.append(_clean_text(value))
    for key in ("years", "compare_targets"):
        values = slot_values.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            parts.extend(_clean_text(item) for item in values if item)

    joined = " ".join(part for part in parts if part)
    terms: List[str] = []
    terms.extend(_NUMBER_RE.findall(joined))
    terms.extend(_ASCII_TOKEN_RE.findall(joined))
    terms.extend(_cjk_ngrams(joined))
    terms.extend(token for token in coarse_tokenize(joined) if len(token) >= 2)

    filtered: List[str] = []
    for term in terms:
        value = term.strip()
        if len(value) < 2:
            continue
        if value in {"什么", "多少", "哪里", "这个", "公司", "报告", "表格", "分别", "出处"}:
            continue
        filtered.append(value)
    filtered.sort(key=len, reverse=True)
    return _unique(filtered)


def _find_term_spans(text: str, terms: Sequence[str]) -> List[tuple[int, int, str]]:
    lowered = text.lower()
    spans: List[tuple[int, int, str]] = []
    for term in terms:
        token = str(term or "").strip()
        if not token:
            continue
        start = 0
        needle = token.lower()
        while True:
            index = lowered.find(needle, start)
            if index < 0:
                break
            spans.append((index, index + len(token), token))
            start = index + max(1, len(token))
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    return spans


def _best_window(text: str, spans: Sequence[tuple[int, int, str]], max_chars: int) -> tuple[int, int, List[str]]:
    if not spans:
        return 0, min(len(text), max_chars), []

    best_start = 0
    best_end = min(len(text), max_chars)
    best_terms: List[str] = []
    best_score = -1
    half = max(120, max_chars // 2)
    for start, end, _term in spans:
        center = (start + end) // 2
        window_start = max(0, center - half)
        window_end = min(len(text), window_start + max_chars)
        window_start = max(0, window_end - max_chars)
        matched = _unique([term for s, e, term in spans if s < window_end and e > window_start])
        score = len(matched) * 100 + sum(len(term) for term in matched)
        if score > best_score:
            best_score = score
            best_start = window_start
            best_end = window_end
            best_terms = matched
    return best_start, best_end, best_terms


def _clip_window(text: str, start: int, end: int) -> str:
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}".strip()


def _table_header(text: str, max_chars: int = 420) -> str:
    cleaned = text.replace("<TABLE_START>", "").replace("<TABLE_END>", "").strip()
    if not cleaned:
        return ""
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) >= 2:
        return "\n".join(lines[:2])[:max_chars].strip()
    return cleaned[:max_chars].strip()


def build_audit_snippet(
    content: str,
    *,
    question: str,
    query_type: str,
    chunk_type: str = "",
    slots: Mapping[str, Any] | None = None,
    max_chars: int | None = None,
) -> Dict[str, Any]:
    text = _clean_text(content, preserve_newlines=True)
    if not text:
        return {"content": "", "reason": "empty_content", "matched_terms": []}

    terms = build_query_terms(question, slots)
    spans = _find_term_spans(text, terms)
    is_table = str(chunk_type or "").lower() == "table" or "<TABLE_START>" in text
    effective_max = int(max_chars or (1400 if is_table else 1100))
    effective_max = max(500, effective_max)

    if spans:
        start, end, matched = _best_window(text, spans, effective_max)
        snippet = _clip_window(text, start, end)
        reason = "table_keyword_window" if is_table else "keyword_window"
        if is_table:
            header = _table_header(text)
            if header and header not in snippet:
                snippet = f"{header}\n{snippet}"
        return {
            "content": snippet[: max(effective_max + 500, effective_max)],
            "reason": reason,
            "matched_terms": matched[:20],
        }

    fallback = text[:effective_max].strip()
    return {
        "content": fallback,
        "reason": "fallback_head",
        "matched_terms": [],
    }


def build_audit_evidence_brief(
    question: str,
    query_type: str,
    slots: Mapping[str, Any] | None,
    evidence: Sequence[Mapping[str, Any]],
    *,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    brief: List[Dict[str, Any]] = []
    for index, item in enumerate(list(evidence)[: max(1, int(limit))], start=1):
        content = _clean_text(item.get("content") or item.get("raw_doc"), preserve_newlines=True)
        snippet = build_audit_snippet(
            content,
            question=question,
            query_type=query_type,
            chunk_type=str(item.get("chunk_type") or ""),
            slots=slots,
        )
        brief.append(
            {
                "evidence_id": f"C{index}",
                "chunk_id": item.get("chunk_id", ""),
                "rank": index,
                "chunk_type": item.get("chunk_type", ""),
                "doc_source": item.get("doc_source", ""),
                "heading_path": item.get("heading_path", ""),
                "content": snippet["content"],
                "snippet_reason": snippet["reason"],
                "matched_terms": snippet["matched_terms"],
                "original_content_chars": len(content),
                "audit_content_chars": len(str(snippet["content"] or "")),
                "score": item.get("final_score") or item.get("score"),
            }
        )
    return brief

