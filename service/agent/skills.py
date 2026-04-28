from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple


@dataclass(frozen=True)
class SkillPackage:
    task_description: str
    prompt_template: str
    few_shot_examples: Tuple[Mapping[str, Any], ...] = tuple()
    slot_schema: Dict[str, Any] = field(default_factory=dict)
    tool_constraints: Dict[str, Any] = field(default_factory=dict)
    execution_config: Dict[str, Any] = field(default_factory=dict)


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
    package: SkillPackage

    def get_missing_slots(self, slots: Dict[str, Any]) -> List[str]:
        missing: List[str] = []
        for slot in self._effective_required_slots():
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

    def _effective_required_slots(self) -> Tuple[str, ...]:
        if self.required_slots:
            return self.required_slots
        raw_required = self.package.slot_schema.get("required", [])
        if not isinstance(raw_required, list):
            return tuple()
        return tuple(str(item) for item in raw_required if str(item).strip())

    def package_metadata(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "query_types": list(self.query_types),
            "task_description": self.package.task_description,
            "prompt_template": self.package.prompt_template,
            "few_shot_examples": list(self.package.few_shot_examples),
            "slot_schema": dict(self.package.slot_schema),
            "tool_constraints": dict(self.package.tool_constraints),
            "execution_config": dict(self.package.execution_config),
            "tool_chain": list(self.tool_chain),
            "guardrails": dict(self.guardrails),
        }


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
    package=SkillPackage(
        task_description="Answer precise fact lookup questions grounded in retrieved evidence.",
        prompt_template=(
            "You are the FactLookup skill. Return concise, citation-grounded factual answers. "
            "If evidence is insufficient, trigger conservative decisioning."
        ),
        few_shot_examples=(
            {
                "question": "What is the registered capital of Company A?",
                "answer_style": "single-fact with citation",
            },
        ),
        slot_schema={"required": [], "optional": ["scope", "entity"]},
        tool_constraints={"allowed_tools": ["query_expander", "retriever", "reranker", "answer_generator"]},
        execution_config={"max_iterations": 1, "retry_on_low_evidence": True},
    ),
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
    package=SkillPackage(
        task_description="Answer table-centric numerical questions with strong table evidence coverage.",
        prompt_template=(
            "You are the TableQA skill. Prioritize table rows/cells, preserve period and metric alignment, "
            "and include units when available."
        ),
        few_shot_examples=(
            {
                "question": "What was 2025 revenue?",
                "expected_slots": {"metric": "revenue", "period": "2025"},
            },
        ),
        slot_schema={"required": ["metric", "period"], "optional": ["unit", "scope", "table_name"]},
        tool_constraints={"must_include_table_evidence": True, "minimum_table_chunks": 2},
        execution_config={"table_priority": "high", "retry_with_table_terms": True},
    ),
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
    package=SkillPackage(
        task_description="Locate source citations for a target statement and return traceable evidence snippets.",
        prompt_template=(
            "You are the CitationLocate skill. Find source-aligned snippets and return citation spans with "
            "high precision."
        ),
        few_shot_examples=(
            {
                "question": "Find evidence for: Net profit increased by 12%.",
                "expected_slots": {"target_statement": "Net profit increased by 12%"},
            },
        ),
        slot_schema={"required": ["target_statement"], "optional": ["scope", "period"]},
        tool_constraints={"require_snippet": True, "snippet_max_length": 280},
        execution_config={"precision_mode": True},
    ),
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
    package=SkillPackage(
        task_description="Summarize evidence within a defined scope while preserving citation traceability.",
        prompt_template=(
            "You are the Summarization skill. Organize key points by topic, avoid unsupported claims, "
            "and retain citation grounding."
        ),
        few_shot_examples=(
            {
                "question": "Summarize the 2025 operations section.",
                "expected_slots": {"scope": "2025 operations section"},
            },
        ),
        slot_schema={"required": ["scope"], "optional": ["period", "focus"]},
        tool_constraints={"require_multi_evidence": True},
        execution_config={"summary_style": "bullet", "max_points": 6},
    ),
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
    package=SkillPackage(
        task_description="Generate structured reports from evidence with sectioned output and citations.",
        prompt_template=(
            "You are the ReportGeneration skill. Produce sectioned outputs (overview, findings, risks) "
            "and ensure each section is evidence-grounded."
        ),
        few_shot_examples=(
            {
                "question": "Generate a compliance report for Q4 controls.",
                "expected_slots": {"scope": "Q4 controls"},
            },
        ),
        slot_schema={"required": ["scope"], "optional": ["period", "audience"]},
        tool_constraints={"output_must_be_structured": True},
        execution_config={"output_format": "sections", "include_risk_section": True},
    ),
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
    package=SkillPackage(
        task_description="Compare claims across multiple documents and highlight differences with citations.",
        prompt_template=(
            "You are the MultiDocCompare skill. Produce side-by-side comparisons and explicitly call out "
            "agreement/disagreement with source references."
        ),
        few_shot_examples=(
            {
                "question": "Compare supplier risk disclosures for Vendor A and Vendor B.",
                "expected_slots": {"compare_targets": ["Vendor A", "Vendor B"]},
            },
        ),
        slot_schema={"required": ["compare_targets"], "optional": ["period", "scope"]},
        tool_constraints={"min_distinct_documents": 2},
        execution_config={"comparison_layout": "side_by_side"},
    ),
)


ALL_SKILLS = (
    FactLookupSkill,
    TableQASkill,
    CitationLocateSkill,
    SummarizationSkill,
    ReportGenerationSkill,
    MultiDocCompareSkill,
)
