# Enterprise Unstructured Document Trusted QA Agent

Trusted QA service for enterprise PDF documents. The active system is centered on the `service.*` stack: PDF indexing, hybrid retrieval, evidence gating, grounded answer generation, citations, and session traces.

## Active API

```text
GET  /health
POST /documents/index
POST /qa/ask
GET  /qa/sessions/{session_id}
```

Legacy chat-llm endpoints are intentionally not mounted:

```text
POST /askLLM
POST /team-leader-task
POST /react-ask
```

## Runtime Path

```text
main.py
  -> controller.apis
  -> controller.apis.document_controller / qa_controller / session_controller
  -> service.pdf / service.agent.trusted_qa_workflow / service.retrieval
```

## Run

```bash
python -m uvicorn main:app --reload
```

## Test

```bash
python -m pytest tests -q
python scripts/workflow_acceptance.py
python scripts/e2e_acceptance.py
python scripts/pgvector_smoke.py
```

## Configuration

Primary configuration lives in `config/app.yaml`:

- `llm`: provider/model/API endpoint settings.
- `embedding`: embedding provider and dimension.
- `storage`: pgvector/local development storage backend.
- `retrieval`: hybrid retrieval and ranking controls.
- `guardrails`: evidence thresholds and retry policy.

Use `.env.template` for local environment overrides.