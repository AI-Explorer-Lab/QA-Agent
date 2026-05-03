# TableQASkill

```json
{
  "skill_name": "TableQASkill",
  "query_types": ["table_qa"],
  "required_slots": ["metric", "period"],
  "input_schema": {
    "question": "str",
    "metric": "str",
    "period": "str",
    "collection_name": "str"
  },
  "tool_chain": [
    "clarify_gate",
    "query_expander",
    "parallel_hybrid_retrieval",
    "table_prioritized_retrieval",
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
    "table_evidence_quota": 2
  },
  "trace_fields": [
    "selected_skill",
    "tool_chain",
    "observations",
    "table_evidence_count"
  ],
  "few_shot_examples": [
    {
      "question": "What was 2025 revenue?",
      "expected_slots": {
        "metric": "revenue",
        "period": "2025"
      }
    }
  ],
  "slot_schema": {
    "required": ["metric", "period"],
    "optional": ["unit", "scope", "table_name"]
  },
  "tool_constraints": {
    "must_include_table_evidence": true,
    "minimum_table_chunks": 2
  },
  "execution_config": {
    "table_priority": "high",
    "retry_with_table_terms": true
  }
}
```

## Task Description
Answer table-centric numerical questions with strong table evidence coverage.

## Prompt Template
You are the TableQA skill. Prioritize table rows and cells, preserve period and metric alignment, and include units when available.
