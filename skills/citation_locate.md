# CitationLocateSkill

```json
{
  "skill_name": "CitationLocateSkill",
  "query_types": ["citation_locate"],
  "required_slots": ["target_statement"],
  "input_schema": {
    "question": "str",
    "target_statement": "str",
    "collection_name": "str"
  },
  "tool_chain": [
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
    "require_snippet": true
  },
  "trace_fields": [
    "selected_skill",
    "tool_chain",
    "observations",
    "citation_locations"
  ],
  "few_shot_examples": [
    {
      "question": "Find evidence for: Net profit increased by 12%.",
      "expected_slots": {
        "target_statement": "Net profit increased by 12%"
      }
    }
  ],
  "slot_schema": {
    "required": ["target_statement"],
    "optional": ["scope", "period"]
  },
  "tool_constraints": {
    "require_snippet": true,
    "snippet_max_length": 280
  },
  "execution_config": {
    "precision_mode": true
  }
}
```

## Task Description
Locate source citations for a target statement and return traceable evidence snippets.

## Prompt Template
You are the CitationLocate skill. Find source-aligned snippets and return citation spans with high precision.

