# FactLookupSkill

```json
{
  "skill_name": "FactLookupSkill",
  "query_types": ["fact_lookup"],
  "required_slots": [],
  "input_schema": {
    "question": "str",
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
    "refuse_on_low_evidence": true
  },
  "trace_fields": [
    "selected_skill",
    "tool_chain",
    "observations",
    "gate_decision"
  ],
  "few_shot_examples": [
    {
      "question": "What is the registered capital of Company A?",
      "answer_style": "single-fact with citation"
    }
  ],
  "slot_schema": {
    "required": [],
    "optional": ["scope", "entity"]
  },
  "tool_constraints": {
    "allowed_tools": ["query_expander", "retriever", "reranker", "answer_generator"]
  },
  "execution_config": {
    "max_iterations": 1,
    "retry_on_low_evidence": true
  }
}
```

## Task Description
Answer precise fact lookup questions grounded in retrieved evidence.

## Prompt Template
You are the FactLookup skill. Return concise, citation-grounded factual answers. If evidence is insufficient, trigger conservative decisioning.
