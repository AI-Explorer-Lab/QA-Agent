from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class SkillDefinition:
    skill_name: str
    query_types: Tuple[str, ...]
    required_slots: Tuple[str, ...]
    input_schema: Dict[str, Any]
    tool_chain: Tuple[str, ...]
    output_schema: Dict[str, Any]
    guardrails: Dict[str, Any]
    trace_fields: Tuple[str, ...]

    def get_missing_slots(self, slots: Dict[str, Any]) -> List[str]:
        missing: List[str] = []
        for slot in self.required_slots:
            value = slots.get(slot)
            if isinstance(value, list):
                if len(value) == 0:
                    missing.append(slot)
                elif slot == "compare_targets" and len(value) < 2:
                    missing.append(slot)
                continue
            if not value:
                missing.append(slot)
        return missing


FactLookupSkill = SkillDefinition(
    skill_name="FactLookupSkill",
    query_types=("fact_lookup",),
    required_slots=tuple(),
    input_schema={"question": "str", "collection_name": "str"},
    tool_chain=(
        "query_expander",
        "parallel_hybrid_retrieval",
        "two_stage_hybrid_rerank",
        "evidence_gate",
        "answer_generator",
    ),
    output_schema={"answer": "str", "citations": "list", "evidence": "list", "decision": "str", "schema": "dict"},
    guardrails={"refuse_on_low_evidence": True},
    trace_fields=("selected_skill", "tool_chain", "observations", "gate_decision"),
)

TableQASkill = SkillDefinition(
    skill_name="TableQASkill",
    query_types=("table_qa",),
    required_slots=("metric", "period"),
    input_schema={"question": "str", "metric": "str", "period": "str", "collection_name": "str"},
    tool_chain=(
        "query_expander",
        "parallel_hybrid_retrieval",
        "table_prioritized_retrieval",
        "two_stage_hybrid_rerank",
        "evidence_gate",
        "answer_generator",
    ),
    output_schema={"answer": "str", "citations": "list", "evidence": "list", "decision": "str", "schema": "dict"},
    guardrails={"table_evidence_quota": 2},
    trace_fields=("selected_skill", "tool_chain", "observations", "table_evidence_count"),
)

CitationLocateSkill = SkillDefinition(
    skill_name="CitationLocateSkill",
    query_types=("citation_locate",),
    required_slots=("target_statement",),
    input_schema={"question": "str", "target_statement": "str", "collection_name": "str"},
    tool_chain=(
        "query_expander",
        "parallel_hybrid_retrieval",
        "two_stage_hybrid_rerank",
        "evidence_gate",
        "answer_generator",
    ),
    output_schema={"answer": "str", "citations": "list", "evidence": "list", "decision": "str", "schema": "dict"},
    guardrails={"require_snippet": True},
    trace_fields=("selected_skill", "tool_chain", "observations", "citation_locations"),
)

SummarizationSkill = SkillDefinition(
    skill_name="SummarizationSkill",
    query_types=("summarization",),
    required_slots=("scope",),
    input_schema={"question": "str", "scope": "str", "collection_name": "str"},
    tool_chain=(
        "query_expander",
        "parallel_hybrid_retrieval",
        "two_stage_hybrid_rerank",
        "evidence_gate",
        "answer_generator",
    ),
    output_schema={"answer": "str", "citations": "list", "evidence": "list", "decision": "str", "schema": "dict"},
    guardrails={"min_evidence": 2},
    trace_fields=("selected_skill", "tool_chain", "observations", "summary_topics"),
)

ReportGenerationSkill = SkillDefinition(
    skill_name="ReportGenerationSkill",
    query_types=("report_generation",),
    required_slots=("scope",),
    input_schema={"question": "str", "scope": "str", "collection_name": "str"},
    tool_chain=(
        "query_expander",
        "parallel_hybrid_retrieval",
        "two_stage_hybrid_rerank",
        "evidence_gate",
        "answer_generator",
    ),
    output_schema={"answer": "str", "citations": "list", "evidence": "list", "decision": "str", "schema": "dict"},
    guardrails={"structured_report": True},
    trace_fields=("selected_skill", "tool_chain", "observations", "report_outline"),
)

MultiDocCompareSkill = SkillDefinition(
    skill_name="MultiDocCompareSkill",
    query_types=("multi_doc_compare",),
    required_slots=("compare_targets",),
    input_schema={"question": "str", "compare_targets": "list", "collection_name": "str"},
    tool_chain=(
        "query_expander",
        "parallel_hybrid_retrieval",
        "two_stage_hybrid_rerank",
        "evidence_gate",
        "answer_generator",
    ),
    output_schema={"answer": "str", "citations": "list", "evidence": "list", "decision": "str", "schema": "dict"},
    guardrails={"require_multi_doc_evidence": True},
    trace_fields=("selected_skill", "tool_chain", "observations", "doc_coverage"),
)


ALL_SKILLS = (
    FactLookupSkill,
    TableQASkill,
    CitationLocateSkill,
    SummarizationSkill,
    ReportGenerationSkill,
    MultiDocCompareSkill,
)
