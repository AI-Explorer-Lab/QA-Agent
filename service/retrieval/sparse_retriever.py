from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .types import ensure_candidate_dict


_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9_]+", flags=re.IGNORECASE)


def coarse_tokenize(text: str) -> List[str]:
    value = str(text or "").strip().lower()
    if not value:
        return []

    tokens: List[str] = []
    for segment in _TOKEN_PATTERN.findall(value):
        if not segment:
            continue

        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            tokens.append(segment)
            if len(segment) >= 2:
                for width in (2, 3):
                    if len(segment) < width:
                        continue
                    for index in range(len(segment) - width + 1):
                        tokens.append(segment[index : index + width])
            continue

        tokens.append(segment)

    return tokens


class SparseBM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = max(0.1, float(k1))
        self.b = min(1.0, max(0.0, float(b)))
        self._documents: List[Dict[str, Any]] = []
        self._tokenized_docs: List[List[str]] = []
        self._df: Counter[str] = Counter()
        self._avg_doc_len: float = 0.0

    def index_chunks(self, chunks: Iterable[Mapping[str, Any]]) -> None:
        self._documents = [ensure_candidate_dict(item) for item in chunks]
        self._tokenized_docs = []
        self._df = Counter()

        for chunk in self._documents:
            searchable_text = " ".join(
                [
                    str(chunk.get("raw_doc") or chunk.get("content") or ""),
                    str(chunk.get("heading_path") or ""),
                    str(chunk.get("level1_title") or ""),
                    str(chunk.get("level2_title") or ""),
                    str(chunk.get("level3_title") or ""),
                    str(chunk.get("table_header_text") or ""),
                    str(chunk.get("table_context_text") or ""),
                ]
            ).strip()
            tokens = coarse_tokenize(searchable_text)
            self._tokenized_docs.append(tokens)
            self._df.update(set(tokens))

        total_tokens = sum(len(tokens) for tokens in self._tokenized_docs)
        self._avg_doc_len = total_tokens / max(1, len(self._tokenized_docs))

    def search(
        self,
        query: str,
        top_k: int,
        collection_name: str = "",
        chunk_type: str | None = None,
    ) -> List[Dict[str, Any]]:
        if not self._documents:
            return []

        query_tokens = coarse_tokenize(query)
        if not query_tokens:
            return []

        unique_query_tokens = list(dict.fromkeys(query_tokens))
        corpus_size = len(self._documents)
        results: List[Dict[str, Any]] = []

        for index, document in enumerate(self._documents):
            if collection_name and str(document.get("collection_name") or "") != collection_name:
                continue
            if chunk_type and str(document.get("chunk_type") or "text") != chunk_type:
                continue

            tokens = self._tokenized_docs[index]
            if not tokens:
                continue

            tf = Counter(tokens)
            doc_len = max(1, len(tokens))
            score = 0.0
            for term in unique_query_tokens:
                term_tf = float(tf.get(term, 0))
                if term_tf <= 0:
                    continue

                df = float(self._df.get(term, 0))
                idf = math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))
                denom = term_tf + self.k1 * (1.0 - self.b + self.b * doc_len / max(1e-6, self._avg_doc_len))
                if denom <= 0:
                    continue
                score += idf * ((term_tf * (self.k1 + 1.0)) / denom)

            if score <= 0:
                continue

            item = dict(document)
            item["bm25_score"] = float(score)
            item["score"] = float(score)
            results.append(item)

        results.sort(key=lambda row: float(row.get("bm25_score") or 0.0), reverse=True)

        normalized: List[Dict[str, Any]] = []
        max_score = float(results[0].get("bm25_score") or 0.0) if results else 0.0
        for row in results[: max(1, int(top_k))]:
            value = float(row.get("bm25_score") or 0.0)
            normalized_score = value / max_score if max_score > 0 else 0.0
            payload = dict(row)
            payload["bm25_score"] = normalized_score
            payload["score"] = normalized_score
            payload.setdefault("chunk_id", str(payload.get("chunk_id") or ""))
            normalized.append(payload)

        return normalized
