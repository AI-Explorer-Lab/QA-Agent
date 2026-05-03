# SummarizationSkill

```json
{
  "skill_name": "SummarizationSkill",
  "query_types": ["summarization"],
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
    "min_evidence": 2
  },
  "trace_fields": [
    "selected_skill",
    "tool_chain",
    "observations",
    "summary_topics"
  ],
  "few_shot_examples": [
    {
      "question": "Summarize the 2025 operations section.",
      "expected_slots": {
        "scope": "2025 operations section"
      }
    }
  ],
  "slot_schema": {
    "required": ["scope"],
    "optional": ["period", "focus"]
  },
  "tool_constraints": {
    "require_multi_evidence": true
  },
  "execution_config": {
    "summary_style": "bullet",
    "max_points": 6
  }
}
```

## Task Description
Summarize evidence within a defined scope while preserving citation traceability.

## Prompt Template
You are the Summarization skill. Organize key points by topic, avoid unsupported claims, and retain citation grounding.
