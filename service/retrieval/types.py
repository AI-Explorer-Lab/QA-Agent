from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, MutableMapping


try:  # pragma: no cover - integrated projects may provide richer models.
    from domain.retrieval import RetrievalCandidate as DomainRetrievalCandidate  # type: ignore
except Exception:  # pragma: no cover
    DomainRetrievalCandidate = None


@dataclass
class RetrievalCandidate:
    chunk_id: str
    raw_doc: str = ""
    collection_name: str = ""
    doc_id: str = ""
    doc_source: str = ""
    page_idx: int | None = None
    chunk_index: int | None = None
    chunk_type: str = "text"
    heading_path: str = ""
    level1_title: str = ""
    level2_title: str = ""
    level3_title: str = ""
    table_id: str = ""
    sub_table_id: str = ""
    table_header_text: str = ""
    table_context_text: str = ""
    dense_score: float = 0.0
    bm25_score: float = 0.0
    metadata_boost: float = 0.0
    table_boost: float = 0.0
    table_route_score: float = 0.0
    final_score: float = 0.0
    source_channels: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RetrievalCandidate":
        if DomainRetrievalCandidate is not None:
            try:
                domain_obj = DomainRetrievalCandidate(**dict(value))
                if hasattr(domain_obj, "model_dump"):
                    return cls(**domain_obj.model_dump())
                if hasattr(domain_obj, "dict"):
                    return cls(**domain_obj.dict())
                if hasattr(domain_obj, "__dict__"):
                    return cls(**domain_obj.__dict__)
            except Exception:
                pass

        payload: Dict[str, Any] = {k: v for k, v in dict(value).items()}
        payload.setdefault("chunk_id", str(payload.get("chunk_id") or ""))
        if not payload["chunk_id"]:
            payload["chunk_id"] = str(payload.get("id") or payload.get("_id") or "")
        if not payload["chunk_id"]:
            payload["chunk_id"] = f"anon-{abs(hash(str(payload)))}"
        payload.setdefault("raw_doc", str(payload.get("raw_doc") or payload.get("content") or ""))
        payload.setdefault("chunk_type", str(payload.get("chunk_type") or "text"))
        payload.setdefault("source_channels", list(payload.get("source_channels") or []))
        return cls(**{k: payload.get(k) for k in cls.__dataclass_fields__.keys()})


def ensure_candidate_dict(value: RetrievalCandidate | Mapping[str, Any] | MutableMapping[str, Any]) -> Dict[str, Any]:
    if isinstance(value, RetrievalCandidate):
        return value.to_dict()
    return RetrievalCandidate.from_mapping(value).to_dict()
