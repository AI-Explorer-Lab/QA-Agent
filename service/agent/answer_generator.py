from __future__ import annotations

import re
from typing import Any, Dict, List


def _content(row: Dict[str, Any]) -> str:
    return str(row.get("raw_doc") or row.get("content") or "").strip()


def _clip(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_evidence_payload(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence = []
    for index, row in enumerate(rows, start=1):
        content = _content(row)
        item = {
            "evidence_id": f"E{index}",
            "chunk_id": str(row.get("chunk_id") or ""),
            "doc_id": str(row.get("doc_id") or ""),
            "doc_source": str(row.get("doc_source") or ""),
            "chunk_type": str(row.get("chunk_type") or "text"),
            "content": content,
            "score": float(row.get("final_score") or row.get("score") or 0.0),
            "rank": index,
            "metadata": {
                "page_idx": row.get("page_idx"),
                "page_range": row.get("page_range", ""),
                "heading_path": row.get("heading_path", ""),
                "collection_name": row.get("collection_name", ""),
                "source_channels": row.get("source_channels", []),
                "dense_score": row.get("dense_score", 0.0),
                "bm25_score": row.get("bm25_score", 0.0),
            },
        }
        evidence.append(item)
    return evidence


def build_citations(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations = []
    for index, item in enumerate(evidence, start=1):
        meta = item.get("metadata", {})
        citations.append(
            {
                "citation_id": f"C{index}",
                "chunk_id": item.get("chunk_id", ""),
                "doc_id": item.get("doc_id", ""),
                "doc_source": item.get("doc_source", ""),
                "collection_name": meta.get("collection_name", ""),
                "page_idx": meta.get("page_idx"),
                "page_range": meta.get("page_range", ""),
                "heading_path": meta.get("heading_path", ""),
                "quote": _clip(item.get("content", ""), 260),
                "confidence": round(float(item.get("score") or 0.0), 4),
            }
        )
    return citations


class AnswerGenerator:
    def generate(
        self,
        question: str,
        query_type: str,
        evidence: List[Dict[str, Any]],
        decision: str = "answer",
        gate_reason: str = "",
    ) -> Dict[str, Any]:
        evidence_payload = build_evidence_payload(evidence)
        citations = build_citations(evidence_payload)

        if decision == "clarify":
            answer = self._clarify_answer(query_type, gate_reason)
            return {"answer": answer, "citations": [], "evidence": evidence_payload, "confidence": 0.0}
        if decision == "refuse":
            answer = "????????????? PDF ?????????????????????????? PDF?"
            return {"answer": answer, "citations": citations, "evidence": evidence_payload, "confidence": 0.0}

        if query_type == "table_qa":
            answer = self._table_answer(evidence_payload)
        elif query_type == "citation_locate":
            answer = self._citation_locate_answer(evidence_payload)
        elif query_type == "multi_doc_compare":
            answer = self._compare_answer(evidence_payload)
        elif query_type in {"summarization", "report_generation"}:
            answer = self._summary_answer(query_type, evidence_payload)
        else:
            answer = self._fact_answer(evidence_payload)

        confidence = max([float(item.get("score") or 0.0) for item in evidence_payload] or [0.0])
        return {
            "answer": answer,
            "citations": citations,
            "evidence": evidence_payload,
            "confidence": round(min(1.0, max(0.0, confidence)), 4),
        }

    def _fact_answer(self, evidence: List[Dict[str, Any]]) -> str:
        lines = []
        for item in evidence[:3]:
            lines.append(f"- {_clip(item.get('content', ''), 180)} [{self._citation_label(item)}]")
        return "??????? PDF ???\n" + "\n".join(lines)

    def _table_answer(self, evidence: List[Dict[str, Any]]) -> str:
        lines = ["???????????"]
        for item in evidence:
            if item.get("chunk_type") != "table":
                continue
            source = self._citation_label(item)
            lines.append(f"- ??/??/??/??/???{_clip(item.get('content', ''), 220)} [{source}]")
        if len(lines) == 1:
            return self._fact_answer(evidence)
        return "\n".join(lines)

    def _citation_locate_answer(self, evidence: List[Dict[str, Any]]) -> str:
        lines = ["???????"]
        for item in evidence[:5]:
            meta = item.get("metadata", {})
            lines.append(
                f"- ?????{_clip(item.get('content', ''), 180)}????{meta.get('page_idx')}??????{meta.get('heading_path', '')}?chunk_id?{item.get('chunk_id')} [{self._citation_label(item)}]"
            )
        return "\n".join(lines)

    def _summary_answer(self, query_type: str, evidence: List[Dict[str, Any]]) -> str:
        title = "?????" if query_type == "report_generation" else "??"
        lines = [f"{title}??????????????"]
        for item in evidence[:5]:
            lines.append(f"- {_clip(item.get('content', ''), 170)} [{self._citation_label(item)}]")
        return "\n".join(lines)

    def _compare_answer(self, evidence: List[Dict[str, Any]]) -> str:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in evidence:
            grouped.setdefault(str(item.get("doc_source") or item.get("doc_id") or "unknown"), []).append(item)
        lines = ["??????????"]
        for source, rows in grouped.items():
            lines.append(f"- {source}: {_clip(rows[0].get('content', ''), 180)} [{self._citation_label(rows[0])}]")
        return "\n".join(lines)

    @staticmethod
    def _clarify_answer(query_type: str, gate_reason: str) -> str:
        if query_type == "table_qa":
            return "????????????????????????"
        if query_type == "multi_doc_compare":
            return "????????????????? PDF ??????"
        return f"???????????????????????????{gate_reason or 'missing_slots'}"

    @staticmethod
    def _citation_label(item: Dict[str, Any]) -> str:
        rank = item.get("rank") or 1
        return f"C{rank}"
