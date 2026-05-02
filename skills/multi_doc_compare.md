# MultiDocCompareSkill

```json
{
  "skill_name": "MultiDocCompareSkill",
  "query_types": ["multi_doc_compare"],
  "required_slots": ["compare_targets"],
  "input_schema": {
    "question": "str",
    "compare_targets": "list",
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
    "require_multi_doc_evidence": true
  },
  "trace_fields": [
    "selected_skill",
    "tool_chain",
    "observations",
    "doc_coverage"
  ],
  "few_shot_examples": [
    {
      "question": "Compare supplier risk disclosures for Vendor A and Vendor B.",
      "expected_slots": {
        "compare_targets": ["Vendor A", "Vendor B"]
      }
    }
  ],
  "slot_schema": {
    "required": ["compare_targets"],
    "optional": ["period", "scope"]
  },
  "tool_constraints": {
    "min_distinct_documents": 2
  },
  "execution_config": {
    "comparison_layout": "side_by_side"
  }
}
```

## Task Description
Compare claims across multiple documents and highlight differences with citations.

## Prompt Template
You are the MultiDocCompare skill. Produce side-by-side comparisons and explicitly call out agreement or disagreement with source references.
