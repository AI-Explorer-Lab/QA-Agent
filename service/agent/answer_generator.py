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
        del question
        evidence_payload = build_evidence_payload(evidence)
        citations = build_citations(evidence_payload)

        if decision == "clarify":
            answer = self._clarify_answer(query_type, gate_reason)
            return {"answer": answer, "citations": [], "evidence": evidence_payload, "confidence": 0.0}
        if decision == "refuse":
            answer = self._refuse_answer(query_type, gate_reason)
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

    @staticmethod
    def _refuse_answer(query_type: str, gate_reason: str) -> str:
        reason = gate_reason or "low_evidence"
        if reason in {"no_evidence", "no_evidence_after_retry"}:
            return (
                "\u672a\u68c0\u7d22\u5230\u8db3\u591f\u7684 PDF \u8bc1\u636e\uff0c\u65e0\u6cd5\u57fa\u4e8e\u6587\u6863\u53ef\u9760\u56de\u7b54\u3002"
                "\u8bf7\u786e\u8ba4\u5df2\u7ecf\u5b8c\u6210\u6587\u6863\u7d22\u5f15\uff0c\u5e76\u4e14 ask \u8bf7\u6c42\u4f7f\u7528\u4e86\u6b63\u786e\u7684 collection_name\u3002"
            )
        if reason in {"missing_table_evidence", "missing_table_evidence_after_retry"}:
            return "\u672a\u68c0\u7d22\u5230\u8db3\u591f\u7684\u8868\u683c\u8bc1\u636e\uff0c\u65e0\u6cd5\u53ef\u9760\u56de\u7b54\u8be5\u8868\u683c\u95ee\u9898\u3002\u8bf7\u786e\u8ba4\u6587\u6863\u4e2d\u5b58\u5728\u76f8\u5173\u8868\u683c\uff0c\u6216\u8865\u5145\u6307\u6807\u548c\u671f\u95f4\u3002"
        if reason in {"multi_doc_evidence_missing", "multi_doc_evidence_missing_after_retry"}:
            return "\u672a\u68c0\u7d22\u5230\u8db3\u591f\u7684\u591a\u6587\u6863\u8bc1\u636e\uff0c\u65e0\u6cd5\u8fdb\u884c\u53ef\u9760\u5bf9\u6bd4\u3002\u8bf7\u786e\u8ba4\u8981\u5bf9\u6bd4\u7684\u6587\u6863\u90fd\u5df2\u7d22\u5f15\u5230\u540c\u4e00\u4e2a collection_name\u3002"
        if reason in {"insufficient_doc_coverage", "insufficient_doc_coverage_after_retry"}:
            return "\u68c0\u7d22\u5230\u7684\u8bc1\u636e\u8986\u76d6\u4e0d\u8db3\uff0c\u65e0\u6cd5\u751f\u6210\u53ef\u9760\u7684\u603b\u7ed3\u6216\u62a5\u544a\u3002\u8bf7\u6269\u5927\u68c0\u7d22\u8303\u56f4\u6216\u786e\u8ba4\u6587\u6863\u96c6\u5408\u662f\u5426\u5b8c\u6574\u3002"
        if query_type == "citation_locate":
            return "\u672a\u627e\u5230\u53ef\u5b9a\u4f4d\u7684\u539f\u6587\u8bc1\u636e\uff0c\u56e0\u6b64\u65e0\u6cd5\u7ed9\u51fa\u9875\u7801\u3001\u6807\u9898\u8def\u5f84\u6216 chunk_id\u3002"
        return "\u68c0\u7d22\u5230\u7684\u8bc1\u636e\u76f8\u5173\u6027\u4e0d\u8db3\uff0c\u65e0\u6cd5\u57fa\u4e8e PDF \u6587\u6863\u53ef\u9760\u56de\u7b54\u3002\u8bf7\u6362\u4e00\u79cd\u66f4\u5177\u4f53\u7684\u95ee\u6cd5\uff0c\u6216\u786e\u8ba4\u7d22\u5f15\u548c collection_name \u662f\u5426\u6b63\u786e\u3002"

    def _fact_answer(self, evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return "\u672a\u68c0\u7d22\u5230\u53ef\u7528\u4e8e\u56de\u7b54\u7684 PDF \u8bc1\u636e\u3002"
        lines = []
        for item in evidence[:3]:
            lines.append(f"- {_clip(item.get('content', ''), 180)} [{self._citation_label(item)}]")
        return "\u57fa\u4e8e PDF \u8bc1\u636e\uff1a\n" + "\n".join(lines)

    def _table_answer(self, evidence: List[Dict[str, Any]]) -> str:
        lines = ["\u57fa\u4e8e\u8868\u683c\u8bc1\u636e\uff1a"]
        for item in evidence:
            if item.get("chunk_type") != "table":
                continue
            source = self._citation_label(item)
            lines.append(f"- \u6307\u6807/\u6570\u503c/\u5355\u4f4d/\u671f\u95f4\uff1a{_clip(item.get('content', ''), 220)} [{source}]")
        if len(lines) == 1:
            return self._fact_answer(evidence)
        return "\n".join(lines)

    def _citation_locate_answer(self, evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return "\u672a\u627e\u5230\u53ef\u5b9a\u4f4d\u7684\u539f\u6587\u8bc1\u636e\u3002"
        lines = ["\u8bc1\u636e\u4f4d\u7f6e\u5982\u4e0b\uff1a"]
        for item in evidence[:5]:
            meta = item.get("metadata", {})
            lines.append(
                f"- \u76f8\u5173\u5185\u5bb9\uff1a{_clip(item.get('content', ''), 180)}\uff1b\u9875\u7801\uff1a{meta.get('page_idx')}\uff1b"
                f"\u6807\u9898\u8def\u5f84\uff1a{meta.get('heading_path', '')}\uff1bchunk_id\uff1a{item.get('chunk_id')} [{self._citation_label(item)}]"
            )
        return "\n".join(lines)

    def _summary_answer(self, query_type: str, evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return "\u672a\u68c0\u7d22\u5230\u53ef\u7528\u4e8e\u603b\u7ed3\u7684 PDF \u8bc1\u636e\u3002"
        title = "\u62a5\u544a" if query_type == "report_generation" else "\u6458\u8981"
        lines = [f"{title}\uff08\u57fa\u4e8e\u68c0\u7d22\u5230\u7684 PDF \u8bc1\u636e\uff09\uff1a"]
        for item in evidence[:5]:
            lines.append(f"- {_clip(item.get('content', ''), 170)} [{self._citation_label(item)}]")
        return "\n".join(lines)

    def _compare_answer(self, evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return "\u672a\u68c0\u7d22\u5230\u53ef\u7528\u4e8e\u5bf9\u6bd4\u7684 PDF \u8bc1\u636e\u3002"
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in evidence:
            grouped.setdefault(str(item.get("doc_source") or item.get("doc_id") or "unknown"), []).append(item)
        lines = ["\u591a\u6587\u6863\u5bf9\u6bd4\u5982\u4e0b\uff1a"]
        for source, rows in grouped.items():
            lines.append(f"- {source}: {_clip(rows[0].get('content', ''), 180)} [{self._citation_label(rows[0])}]")
        return "\n".join(lines)

    @staticmethod
    def _clarify_answer(query_type: str, gate_reason: str) -> str:
        if query_type == "table_qa":
            return "\u8bf7\u8865\u5145\u8981\u67e5\u8be2\u7684\u6307\u6807\u548c\u671f\u95f4\uff0c\u4f8b\u5982\uff1a2025 \u5e74\u8425\u4e1a\u6536\u5165\u662f\u591a\u5c11\uff1f"
        if query_type == "multi_doc_compare":
            return "\u8bf7\u8bf4\u660e\u8981\u5bf9\u6bd4\u7684\u81f3\u5c11\u4e24\u4e2a PDF\u3001\u516c\u53f8\u6216\u62a5\u544a\u540d\u79f0\u3002"
        return f"\u95ee\u9898\u8fd8\u7f3a\u5c11\u5173\u952e\u4fe1\u606f\uff0c\u8bf7\u8865\u5145\u540e\u518d\u67e5\u8be2\u3002\u7f3a\u5931\u539f\u56e0\uff1a{gate_reason or 'missing_slots'}"

    @staticmethod
    def _citation_label(item: Dict[str, Any]) -> str:
        rank = item.get("rank") or 1
        return f"C{rank}"
