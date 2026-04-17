"""Citation mapper utilities for message storage and response formatting."""

from __future__ import annotations

from typing import Any, Iterable

from domain import Citation, Evidence


class CitationMapper:
    @staticmethod
    def serialize(citations: Iterable[Citation | dict[str, Any]]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for citation in citations:
            if isinstance(citation, Citation):
                serialized.append(citation.model_dump(mode="json"))
            else:
                serialized.append(dict(citation))
        return serialized

    @staticmethod
    def deserialize(citations: Iterable[dict[str, Any]]) -> list[Citation]:
        return [Citation.model_validate(citation) for citation in citations]

    @staticmethod
    def build_from_evidence(evidence_list: Iterable[Evidence | dict[str, Any]]) -> list[Citation]:
        citations: list[Citation] = []
        for evidence in evidence_list:
            if isinstance(evidence, Evidence):
                if evidence.citation is not None:
                    citations.append(evidence.citation)
            else:
                citation_raw = evidence.get("citation") if isinstance(evidence, dict) else None
                if citation_raw:
                    citations.append(Citation.model_validate(citation_raw))
        return citations
