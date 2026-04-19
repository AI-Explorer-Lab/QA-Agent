# Current Project Plan

This project is now maintained as a trusted enterprise document QA agent. The active implementation is the controller/service architecture under `controller/`, `service/`, `domain/`, `mapper/`, `database/`, `core/`, and `utils/`.

## Active Runtime

```text
main.py
  -> controller.apis.router
  -> /documents/index: service.pdf.document_indexer
  -> /qa/ask: service.agent.trusted_qa_workflow
  -> /qa/sessions/{session_id}: service.session.session_service
```

## Core Capabilities

- Parse and chunk PDF documents through `service.pdf`.
- Embed chunks through `service.embedding`.
- Store and retrieve evidence through `service.retrieval` and pgvector/local runtime repositories.
- Classify questions, expand queries, rerank evidence, run evidence guardrails, and generate grounded answers through `service.agent.trusted_qa_workflow`.
- Return answer, decision, citations, evidence, retrieval trace, rerank trace, skill trace, session id, and observations.

## Removed Legacy Surface

The old chat-llm compatibility layer has been removed from the active project tree:

- `api/routes.py`
- `chat_langchain.py`
- `workflow/*` LangGraph/ReAct files
- `db_service/*` FAISS/SQLite compatibility files
- old `chunking_service`, `embedding_service`, `ingest_service`, `regular_service`, and MCP demo files
- old `/askLLM`, `/team-leader-task`, and `/react-ask` routes

## Guardrails For Future Changes

- Keep new API routes under `controller/apis`.
- Keep business logic under `service/*`.
- Do not reintroduce legacy route names or LangGraph workflow modules unless they are deliberately rebuilt as current functionality.
- Update `tests/` when adding or removing public API surface.