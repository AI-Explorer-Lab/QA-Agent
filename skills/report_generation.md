# ReportGenerationSkill

```json
{
  "skill_name": "ReportGenerationSkill",
  "query_types": ["report_generation"],
  "required_slots": ["scope"],
  "input_schema": {
    "question": "str",
    "scope": "str",
    "collection_name": "str"
  },
  "tool_chain": [
    "clarify_gate",
    "query_expander",
    "parallel_hybrid_retrieval",
    "two_stage_hybrid_rerank",
    "evidence_gate",
    "answer_generator"
  ],
  "output_schema": {
    "answer": "str",
    "citations": "list",
    "evidence": "list",
    "decision": "str",
    "schema": "dict"
  },
  "guardrails": {
    "structured_report": true
  },
  "trace_fields": [
    "selected_skill",
    "tool_chain",
    "observations",
    "report_outline"
  ],
  "few_shot_examples": [
    {
      "question": "Generate a compliance report for Q4 controls.",
      "expected_slots": {
        "scope": "Q4 controls"
      }
    }
  ],
  "slot_schema": {
    "required": ["scope"],
    "optional": ["period", "audience"]
  },
  "tool_constraints": {
    "output_must_be_structured": true
  },
  "execution_config": {
    "output_format": "sections",
    "include_risk_section": true
  }
}
```

## Task Description
Generate structured reports from evidence with sectioned output and citations.

## Prompt Template
You are the ReportGeneration skill. Produce sectioned outputs such as overview, findings, and risks, and ensure each section is evidence-grounded.
