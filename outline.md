# Enterprise Unstructured Document Trusted QA Agent 代码库分析

## 1. 整体目标

这个项目面向企业 PDF 非结构化文档，解决“用户基于已索引文档提问时，系统必须从 PDF 证据中检索、引用、判断证据是否充足，并生成可信答案”的问题。

项目类型：PDF-only RAG + Trusted QA Agent + Document QA。

业务场景来自 `service/agent/query_classifier.py` 和 `service/agent/skills.py`：

- 事实问答：`fact_lookup`
- 表格问答：`table_qa`
- 摘要总结：`summarization`
- 引用定位：`citation_locate`
- 报告生成：`report_generation`
- 多文档差异对比：`multi_doc_compare`
- 信息不足澄清：`ambiguous_query` 或缺少必要槽位时进入 clarify

## 2. 核心数据流

### 2.1 文档入库：PDF -> pgvector

入口：`POST /documents/index`

调用链：

`用户提交 pdf_path/collection_name`
-> `controller/apis/document_controller.py:index_documents()`
-> `service/pdf/document_indexer.py:DocumentIndexingService.index_documents()`
-> `service/pdf/pdf_loader.py:collect_pdf_documents()`
-> `service/pdf/mineru_client.py:MinerUClient.parse_pdf_to_mineru_json()`
-> `service/pdf/structured_chunker.py:StructuredChunker.chunk_mineru_payload()`
-> `service/embedding/embedding_service.py:EmbeddingService.embed_texts()`
-> `service/retrieval/runtime.py:replace_collection_chunks()/upsert_runtime_chunks()`
-> `service/retrieval/pgvector_repository.py:PgvectorRepository.upsert_chunks()/replace_collection_chunks()`
-> `pdf_documents/pdf_chunks` 持久化到 PostgreSQL + pgvector

步骤：

1. API 接收 `pdf_path`、`collection_name`、`force_rebuild`、`doc_source`。
2. `collect_pdf_documents()` 校验输入必须是 PDF 文件或包含 PDF 的目录，并计算文件 hash、大小。
3. 非强制重建时，通过 `get_latest_document_by_source()` 判断同一 `collection_name + doc_source` 是否已经索引且 hash 相同；相同则跳过。
4. `MinerUClient.parse_pdf_to_mineru_json()` 解析 PDF：优先读 MinerU JSON，其次调用 MinerU remote OCR，再 fallback 到 PyMuPDF，最后 fallback 到 minimal payload。
5. `StructuredChunker.chunk_mineru_payload()` 将 MinerU payload 转换为 chunk：文本按标题状态和 token 长度切分，表格转 Markdown，并保留页码、标题路径、表格上下文等元数据。
6. `EmbeddingService.embed_texts()` 对 chunk 内容生成 1024 维 embedding；配置可用时走 DashScope/Qwen，不可用或失败时 fallback 到 deterministic hash embedding。
7. `DocumentIndexingService` 规范化 chunk 字段，补齐 `raw_doc/content/embedding/page_range/heading_path/chunk_type/metadata`。
8. `runtime.py` 要求运行时后端为 `pgvector`，然后写入 PostgreSQL。
9. 返回索引结果：成功状态、collection、文档数、chunk 数、跳过文档、实际存储后端等。

### 2.2 通用问答：用户问题 -> 可信答案

入口：`POST /qa/ask`

调用链：

`用户提交 question/collection_name/top_k`
-> `controller/apis/qa_controller.py:ask()`
-> `service/agent/trusted_qa_workflow.py:TrustedQAWorkflow.ask()`
-> `SessionService.load_session()`
-> `classify_query_type()`
-> `SkillRegistry.select_skill()`
-> `LLMService.expand_queries()` 或 `query_expander.expand_queries()`
-> `run_clarify_gate()`
-> `HybridRetriever.retrieve()`
-> `ParallelQueryExecutor.execute()`
-> `PgvectorRepository.dense_search()/keyword_search()/table_search()`
-> `TwoStageHybridReranker.rerank()`
-> `EvidenceGate.evaluate()`
-> `AnswerGenerator.generate()`
-> 可选 `LLMService.generate_grounded_answer()`
-> `evaluate_qa_result()`
-> `SessionService.save_session()`
-> 返回 `answer/citations/evidence/retrieval_trace/rerank_trace/session_id`

步骤：

1. `qa_controller.ask()` 将请求参数传给全局 `TrustedQAWorkflow`。
2. `TrustedQAWorkflow.ask()` 加载或创建 session。
3. `classify_query_type(question)` 用关键词规则识别问题类型。
4. `SkillRegistry.select_skill(query_type)` 根据类型选择技能定义，如 `TableQASkill`、`ReportGenerationSkill`。
5. 查询扩展优先调用 `LLMService.expand_queries()`；LLM 不可用时使用 `service/agent/query_expander.py` 的模板扩展。
6. `run_clarify_gate()` 抽取槽位并判断是否需要澄清。
7. 若需要澄清，直接构造 clarify response，不进入检索。
8. 若可以回答，进入 `HybridRetriever.retrieve()`。
9. `ParallelQueryExecutor.execute()` 对多个 query variant 并发执行 dense、BM25，表格问题额外执行 table route。
10. 合并相同 `chunk_id` 候选，保留来源 channel 和最高 dense/BM25/table 分。
11. `TwoStageHybridReranker.rerank()` 按 dense/BM25/metadata/table boost 排序，去重，补相邻 chunk，并对表格问答保证表格证据 quota。
12. `EvidenceGate.evaluate()` 判断证据是否足够；不足时 retry 或 refuse，缺槽位时 clarify。
13. 如果 gate 返回 retry，workflow 会改写问题追加 gate reason，再重新检索，最多 `retry_limit` 次。
14. `AnswerGenerator.generate()` 先基于证据生成模板化答案、引用、证据 payload。
15. 如果 decision 是 `answer`，再调用 `LLMService.generate_grounded_answer()` 生成基于证据的中文最终答案；LLM 不可用时保留模板答案。
16. `evaluate_qa_result()` 给出评估结果并提升 confidence。
17. `SessionService.save_session()` 写入 user message、assistant message、retrieval trace、evaluation records。
18. API 返回完整 QA response。

### 2.3 表格问答流程

触发条件：`classify_query_type()` 命中“表、表格、指标、数据、同比、环比、毛利率、收入、成本、利润、参数、table、metric”等关键词，query type 为 `table_qa`。

流程：

`用户问题包含指标/数值/表格意图`
-> `classify_query_type() = table_qa`
-> `SkillRegistry.select_skill() = TableQASkill`
-> `run_clarify_gate()` 要求 `metric + period`
-> `expand_queries()` 添加“指标 数值 单位 期间 来源 / 财务表格 / 表头 字段”
-> `ParallelQueryExecutor` 并发执行 dense + BM25 + table route
-> `PgvectorRepository.table_search()` 或 `keyword_search(table_only=True)` 获取表格 chunk
-> `TwoStageHybridReranker` 增加 `table_boost`，并通过 `table_evidence_quota` 优先保留表格证据
-> `EvidenceGate` 检查表格证据数量
-> `AnswerGenerator._table_answer()` 输出“基于表格证据”的答案和引用
-> 可选 LLM 生成 grounded answer

关键点：

- 表格 chunk 在入库阶段由 `StructuredChunker._table_to_chunk_text()` 转 Markdown 保存。
- 表格 chunk 保存 `table_header_text`、`table_context_text`、`table_id`、`sub_table_id`。
- 证据门控中 `table_qa` 要求表格证据数量达到 `table_evidence_quota`。

### 2.4 报告生成流程

触发条件：`classify_query_type()` 命中“报告、汇报、生成报告、分析报告、report”，query type 为 `report_generation`。

流程：

`用户要求生成报告`
-> `classify_query_type() = report_generation`
-> `SkillRegistry.select_skill() = ReportGenerationSkill`
-> `run_clarify_gate()` 要求 `scope`
-> `expand_queries()` 添加“报告提纲 / 风险与机会 / 关键数据引用”
-> `HybridRetriever` 检索候选证据
-> `TwoStageHybridReranker` 重排并去重
-> `EvidenceGate` 按 coverage-sensitive 类型检查证据覆盖
-> `AnswerGenerator._summary_answer(query_type='report_generation')` 输出“报告（基于检索到的 PDF 证据）”
-> 可选 `LLMService.generate_grounded_answer()` 基于证据生成最终报告式回答
-> 保存 session 和 trace

注意：`ReportGenerationSkill` 定义了 `structured_report` guardrail，但当前 workflow 没有独立报告生成器；它复用 `AnswerGenerator._summary_answer()` 和可选 LLM grounded answer。

### 2.5 差异对比 / 多文档对比流程

触发条件：`classify_query_type()` 命中“对比、比较、差异、区别、versus、vs、compare”，query type 为 `multi_doc_compare`。

流程：

`用户要求 A 和 B 差异/对比`
-> `classify_query_type() = multi_doc_compare`
-> `SkillRegistry.select_skill() = MultiDocCompareSkill`
-> `run_clarify_gate()` 要求 `compare_targets` 至少两个
-> `expand_queries()` 添加“差异点 对照 / 多文档 逐项比较 / 版本变化”
-> `HybridRetriever` 检索多个 query variant
-> `TwoStageHybridReranker` 重排和去重
-> `EvidenceGate` 检查证据是否覆盖至少两个 `doc_id/doc_source`
-> 不足时 retry，仍不足则 refuse 或 clarify
-> `AnswerGenerator._compare_answer()` 按 `doc_source/doc_id` 分组输出多文档对比
-> 可选 LLM 生成 grounded answer
-> 保存 session 和 trace

关键点：对比是否可靠不只看分数，还检查证据来自几个文档。

### 2.6 引用定位流程

触发条件：命中“出处、原文、哪一页、页码、引用、citation、source、locate”，query type 为 `citation_locate`。

流程：

`用户问出处/页码/原文位置`
-> `classify_query_type() = citation_locate`
-> `CitationLocateSkill`
-> `run_clarify_gate()` 要求 `target_statement`
-> 混合检索 + 重排
-> `EvidenceGate` 判断相关性
-> `AnswerGenerator._citation_locate_answer()` 输出相关内容、页码、标题路径、chunk_id、引用 ID

### 2.7 信息不足澄清流程

触发条件：问题太短、指代不清、缺必要槽位，例如表格问答缺 `metric/period`，对比缺两个对象，报告缺 `scope`。

流程：

`用户问题不完整`
-> `classify_query_type()` 返回 `ambiguous_query` 或具体类型
-> `run_clarify_gate()` 提取 slots
-> 缺槽位时 decision = `clarify`
-> `AnswerGenerator.generate(decision='clarify')`
-> response.answer 被替换为 `clarify_question`
-> 保存 session
-> 返回澄清问题，不执行检索

## 3. 核心模块拆解

### 3.1 API / Controller

位置：`main.py`、`controller/apis/*.py`

做什么：创建 FastAPI app，注册中间件、异常处理和路由，暴露文档索引、问答、会话、健康检查 API。

输入：HTTP request。

输出：JSON response。

### 3.2 PDF Ingestion / Indexing

位置：`service/pdf/document_indexer.py`、`pdf_loader.py`、`mineru_client.py`、`mineru_parser.py`、`structured_chunker.py`、`heading_recovery.py`

做什么：收集 PDF、解析为 MinerU-like JSON、恢复标题层级、切文本/表格 chunk、保存页码/标题/表格上下文/文档 hash 等元数据。

输入：PDF 文件路径或目录、collection_name、doc_source、force_rebuild。

输出：结构化 chunk 列表。

### 3.3 Embedding

位置：`service/embedding/embedding_service.py`、`service/embedding/embedding_cache.py`

做什么：将文本转成 1024 维向量，支持 DashScope/Qwen 和 deterministic hash fallback，并做缓存。

输入：文本或文本列表。

输出：`List[float]` 或 `List[List[float]]`。

### 3.4 Storage / Repository

位置：`database/connection.py`、`database/pgvector_schema.sql`、`service/retrieval/runtime.py`、`service/retrieval/pgvector_repository.py`

做什么：读取存储配置，构建 PostgreSQL + pgvector repository，保存文档和 chunk，提供 dense search、keyword search、table search。

输入：入库 chunks；检索时输入 collection、query embedding、query text、top_k。

输出：入库计数或候选 chunk 列表。

### 3.5 Retrieval

位置：`service/retrieval/hybrid_retriever.py`、`parallel_query_executor.py`、`sparse_retriever.py`、`two_stage_hybrid_reranker.py`、`retrieval_cache.py`

做什么：对扩展查询并发执行 dense/BM25/table 检索，合并候选，重排，去重，返回证据和 trace。

输入：question、collection_name、top_k、query_type、expanded_queries。

输出：evidence/candidates、retrieval_trace、rerank_trace。

### 3.6 Agent Workflow / Skill Orchestration

位置：`service/agent/trusted_qa_workflow.py`、`query_classifier.py`、`query_expander.py`、`clarify_gate.py`、`evidence_gate.py`、`skill_registry.py`、`skills.py`、`answer_generator.py`

做什么：编排一次 QA 请求的完整生命周期：分类、选技能、澄清、检索、证据门控、答案生成、trace 记录。

输入：question、collection_name、session_id、top_k、expand_query_num、enable_cache。

输出：answer、decision、query_type、citations、evidence、retrieval_trace、rerank_trace、skill_trace。

### 3.7 LLM

位置：`service/llm/llm_client.py`

做什么：读取 YAML 和环境变量配置，创建 OpenAI-compatible `AsyncOpenAI` client，用于 LLM query expansion 和 grounded answer generation。

输入：prompt，或 question/query_type/evidence/citations。

输出：扩展查询列表或基于证据的最终中文答案；不可用时返回 None，由 workflow fallback。

### 3.8 Session / Trace / Evaluation

位置：`service/session/session_service.py`、`service/evaluation/ragas_evaluator.py`

做什么：管理 QA session，保存用户消息、助手消息、retrieval trace、rerank trace、evaluation records，提供会话查询。

输入：session_id、question、assistant_payload、retrieval_trace。

输出：session 详情和 evaluation metadata。

### 3.9 Middleware / Observability

位置：`middlewares/request_log.py`、`exception_handler.py`、`operation_log.py`、`trace_context.py`

做什么：请求日志、异常处理、操作步骤计时、trace id 管理。

输入：HTTP 请求、业务操作事件、异常。

输出：日志、标准化异常响应、trace_id。

## 4. 模块之间如何连接

### 4.1 FastAPI 装配链

`main.py`
-> `FastAPI(...)`
-> `RequestLogMiddleware`
-> `app_exception_handler`
-> `controller.apis.router`
-> `document_router / qa_router / session_router / health_router`

### 4.2 文档索引调用链

`document_controller.index_documents()`
-> `get_document_indexing_service().index_documents()`
-> `collect_pdf_documents()`
-> `MinerUClient.parse_pdf_to_mineru_json()`
-> `StructuredChunker.chunk_mineru_payload()`
-> `EmbeddingService.embed_texts()`
-> `_normalize_chunk_for_retrieval()`
-> `get_runtime_repository()`
-> `replace_collection_chunks()` 或 `upsert_runtime_chunks()`
-> `PgvectorRepository.replace_collection_chunks()/upsert_chunks()`
-> PostgreSQL tables

### 4.3 问答调用链

`qa_controller.ask()`
-> `TrustedQAWorkflow.ask()`
-> `SessionService.load_session()`
-> `classify_query_type()`
-> `SkillRegistry.select_skill()`
-> `LLMService.expand_queries()`
-> fallback `expand_queries()`
-> `run_clarify_gate()`
-> `HybridRetriever.retrieve()`
-> `ParallelQueryExecutor.execute()`
-> `PgvectorRepository.dense_search()/keyword_search()/table_search()`
-> `TwoStageHybridReranker.rerank()`
-> `EvidenceGate.evaluate()`
-> retry retrieval if needed
-> `AnswerGenerator.generate()`
-> optional `LLMService.generate_grounded_answer()`
-> `evaluate_qa_result()`
-> `SessionService.save_session()`
-> API response

### 4.4 检索内部调用链

`HybridRetriever.retrieve()`
-> `ParallelQueryExecutor.execute(top_k * 4)`
-> `_build_query_variants()`
-> `_run_route_task(route='dense')`
-> `EmbeddingService.embed_text()`
-> `PgvectorRepository.dense_search()`
-> `_run_route_task(route='bm25')`
-> `PgvectorRepository.keyword_search()`
-> 条件满足时 `_run_route_task(route='table')`
-> `PgvectorRepository.table_search()`
-> `_merge_candidates()`
-> `TwoStageHybridReranker.rerank()`
-> evidence

### 4.5 数据写入关系

文档索引写入：

- `pdf_documents`：文档级元数据，如 doc_id、collection、doc_source、hash、page_count。
- `pdf_chunks`：chunk 内容、embedding、页码、标题路径、chunk_type、表格信息。

问答写入：

- `qa_sessions`：会话。
- `qa_messages`：用户消息和助手消息。
- `retrieval_traces`：扩展查询、检索 trace、重排 trace、最终证据。
- `evaluation_records`：QA 评估指标。

## 5. 入口在哪里

### 5.1 服务启动入口

主入口文件：`main.py`

启动方式：

```bash
python -m uvicorn main:app --reload
```

关键对象：

- `app = FastAPI(...)`
- `app.include_router(router)`
- `app_chat_llm = app` 是旧命令兼容变量。

### 5.2 API 入口

路由聚合：`controller/apis/__init__.py`、`controller/apis/router.py`

核心 API：

- `GET /health`
- `POST /documents/index`
- `POST /qa/ask`
- `GET /qa/sessions/{session_id}`

### 5.3 文档索引入口函数

- `controller/apis/document_controller.py:index_documents()`
- `service/pdf/document_indexer.py:DocumentIndexingService.index_documents()`

### 5.4 问答入口函数

- `controller/apis/qa_controller.py:ask()`
- `service/agent/trusted_qa_workflow.py:TrustedQAWorkflow.ask()`

### 5.5 核心工作流入口对象

全局单例：

- `service/agent/trusted_qa_workflow.py:_DEFAULT_WORKFLOW`
- `get_trusted_qa_workflow()`

`/qa/ask` 每次请求都会进入同一个 workflow 实例，该实例内部持有 session service、skill registry、LLM service、embedding service、hybrid retriever、evidence gate、answer generator。

## 6. 一句话总流程图

文档索引：

`PDF 文件/目录`
-> `PDF 校验与 hash`
-> `MinerU JSON / remote OCR / PyMuPDF fallback`
-> `解析 blocks`
-> `标题恢复 + 文本/表格 chunk`
-> `embedding`
-> `pgvector 写入 pdf_documents/pdf_chunks`
-> `返回 indexed_chunks`

用户问答：

`用户问题`
-> `会话加载`
-> `问题分类`
-> `技能选择`
-> `查询扩展`
-> `槽位澄清判断`
-> `并行混合检索`
-> `两阶段重排`
-> `证据门控`
-> `模板答案 + 可选 LLM grounded answer`
-> `引用/证据/trace/evaluation`
-> `保存会话`
-> `返回最终答案`

## 7. 架构判断

### 当前设计优点

- 数据流清晰，`TrustedQAWorkflow.ask()` 是 QA 主编排点。
- 文档入库和问答链路分离。
- 表格 chunk 是一等公民，有独立 route、boost 和 gate。
- 输出不只有答案，还包含 citations、evidence、retrieval_trace、rerank_trace，适合可信 QA。
- LLM 不可用时仍可运行，embedding 和答案生成都有 fallback。

### 当前实现边界

- `skills.py` 定义了多技能，但当前没有动态 ReAct 工具调用循环；实际是固定 workflow 编排 + skill trace。
- 报告生成和总结共用 `AnswerGenerator._summary_answer()`，报告结构主要依赖可选 LLM。
- `runtime.py` 当前强制 pgvector，README 中提到的 FAISS/local_dev 更多是历史或测试兼容路径。
- 查询分类是关键词规则，不是模型分类。

# Q&A

## 1. 生成回答的 confidence 怎么给的，用的什么算法？

当前项目里的 `confidence` 不是 LLM 自己打分，也不是完整 RAGAS 在线评估，而是两个来源取最大值：

`AnswerGenerator.generate()` 生成的证据分数 confidence
-> `evaluate_qa_result()` 生成的本地评估 confidence
-> `TrustedQAWorkflow.ask()` 里执行 `response["confidence"] = max(response["confidence"], evaluation["confidence"])`

代码位置：

- `service/agent/answer_generator.py:AnswerGenerator.generate()`
- `service/evaluation/ragas_evaluator.py:evaluate_qa_result()`
- `service/agent/trusted_qa_workflow.py:TrustedQAWorkflow.ask()`

### 1.1 AnswerGenerator 的 confidence

当 `decision == "answer"` 时：

```python
confidence = max([float(item.get("score") or 0.0) for item in evidence_payload] or [0.0])
```

也就是说，它取最终 evidence 里最高的 `score`。

这个 `score` 来自 evidence payload：

```python
"score": float(row.get("final_score") or row.get("score") or 0.0)
```

通常是 `TwoStageHybridReranker` 产出的 `final_score`。

`final_score` 的计算在 `service/retrieval/two_stage_hybrid_reranker.py`：

```text
final_score =
  dense_weight * dense_score
+ bm25_weight * bm25_score
+ metadata_boost_weight * metadata_boost
+ table_boost_weight * table_boost
```

默认权重来自类初始化：

```text
dense_weight = 0.50
bm25_weight = 0.35
metadata_boost_weight = 0.10
table_boost_weight = 0.05
```

所以第一层 confidence 本质是“最强证据 chunk 的重排相关性分数”。

如果 `decision == "clarify"` 或 `decision == "refuse"`，`AnswerGenerator` 返回的 confidence 是 `0.0`。

### 1.2 evaluate_qa_result 的 confidence

`evaluate_qa_result()` 是一个 local RAGAS-compatible 的轻量评估，不调用真实 RAGAS 模型。

它计算三个指标：

```text
grounding_score
completeness_score
consistency_score
```

然后：

```text
overall_score = (grounding + completeness + consistency) / 3
confidence = clamp(overall_score, 0, 1)
```

具体逻辑：

- `grounding_score`：看 citation 数量和 evidence 数量的比例，`citation_count / evidence_count`，最多 1。
- `completeness_score`：看 answer 是否为空，以及不同 decision 下是否完整。
- `consistency_score`：简单规则判断，比如 answer 但没有 evidence，则 consistency 为 0。

### 1.3 最终 confidence

最终返回值：

```python
response["confidence"] = max(answer_payload_confidence, evaluation_confidence)
```

这意味着：

- 如果检索证据分数高，confidence 会被 evidence top score 拉高。
- 如果模板回答完整、引用和证据数量匹配，local evaluation 也可能拉高 confidence。
- 它不是事实正确性的强保证，更像“检索证据强度 + 输出结构完整度”的综合信号。

## 2. workflow 设计是怎么样的？入参是什么，经过了什么变化，出参是什么？

核心 workflow 是 `service/agent/trusted_qa_workflow.py:TrustedQAWorkflow.ask()`。

### 2.1 入参

```python
async def ask(
    self,
    question: str,
    collection_name: str = "default",
    session_id: str | None = None,
    top_k: int = 5,
    expand_query_num: int = 3,
    enable_cache: bool = True,
) -> Dict[str, Any]
```

含义：

- `question`：用户问题。
- `collection_name`：检索哪个文档集合。
- `session_id`：已有会话 ID；为空则创建新会话。
- `top_k`：最终希望返回的证据数量。
- `expand_query_num`：查询扩展数量。
- `enable_cache`：是否启用 retrieval cache。

### 2.2 过程中的状态变化

整体变化链：

`raw question`
-> `session_id`
-> `query_type`
-> `selected_skill`
-> `expanded_queries`
-> `slots/clarify decision`
-> `retrieval candidates`
-> `reranked evidence`
-> `gate decision`
-> `answer_payload`
-> `response`
-> `evaluation`
-> `saved session`

详细步骤：

1. 加载会话：`SessionService.load_session()` 生成或恢复 `session_id`。
2. 问题分类：`classify_query_type(question)` 得到 `query_type`。
3. 技能选择：`SkillRegistry.select_skill(query_type)` 得到 `selected_skill`。
4. 查询扩展：优先 `LLMService.expand_queries()`，失败则 `expand_queries()` 模板扩展。
5. 记录 observations：先记录 session、分类、技能选择。
6. 澄清门控：`run_clarify_gate(question, query_type, collection_name)` 抽取 slots，缺槽位则直接返回 clarify。
7. 检索：`HybridRetriever.retrieve()` 执行并行混合检索和重排。
8. 证据门控：`EvidenceGate.evaluate()` 判断 evidence 是否可回答。
9. 重试：如果 gate 是 `retry`，则追加 gate reason 到 question 后重新检索，最多 `retry_limit` 次。
10. 决策规整：如果最后还是 retry，则转成 refuse。
11. 生成答案：`AnswerGenerator.generate()` 生成答案、citations、evidence、confidence。
12. 可选 LLM grounded answer：如果 decision 是 answer，并且 LLM 可用，则用证据重写最终答案。
13. 构造 response：`_build_response()` 组装 answer、decision、trace、skill_trace。
14. 评估：`evaluate_qa_result()` 生成 local evaluation，并更新 confidence。
15. 保存：`SessionService.save_session()` 写入消息、trace、evaluation。

### 2.3 出参

返回的是 `Dict[str, Any]`，结构与 `domain/qa.py:QAResponse` 对齐，主要字段：

```text
answer: 最终答案
decision: answer / clarify / refuse
query_type: fact_lookup / table_qa / citation_locate / ...
confidence: 最终置信度
citations: 引用列表
evidence: 证据列表
retrieval_trace: 检索过程 trace
rerank_trace: 重排过程 trace
skill_trace: 技能选择和工具链 trace
react_observations: workflow 关键阶段观察记录
session_id: 会话 ID
```

## 3. 如果命中多个 type 的关键词，例如“承诺事项履行情况表格出处”，怎么处理？判断用什么 skill？回答和引用如何对上？

当前分类器是顺序优先级规则，不是多标签分类。

代码位置：`service/agent/query_classifier.py:classify_query_type()`。

判断顺序是：

```text
1. compare keywords -> multi_doc_compare
2. report keywords -> report_generation
3. citation keywords -> citation_locate
4. table keywords -> table_qa
5. summary keywords -> summarization
6. ambiguous_query 判断
7. 默认 fact_lookup
```

所以“承诺事项履行情况表格出处”同时命中：

- “表格” -> `table_qa`
- “出处” -> `citation_locate`

由于 `citation_locate` 的判断在 `table_qa` 前面，最终 query_type 会是：

```text
citation_locate
```

对应 skill 是：

```text
CitationLocateSkill
```

这意味着当前系统会优先理解为“帮我定位出处/页码/原文位置”，而不是“做表格数值问答”。

### 回答和引用如何对上？

对应关系由 `AnswerGenerator` 生成：

1. `build_evidence_payload(rows)` 按重排后的 evidence 顺序生成：`E1、E2、E3...`
2. `build_citations(evidence)` 用同样顺序生成：`C1、C2、C3...`
3. 每个 citation 和 evidence 都带相同的 `chunk_id`、`doc_id`、`doc_source`。
4. 答案模板里通过 `rank` 生成 `[C1]`、`[C2]` 等引用标记。

核心对应规则：

```text
evidence[0] -> evidence_id E1 -> citation C1 -> answer 中 [C1]
evidence[1] -> evidence_id E2 -> citation C2 -> answer 中 [C2]
```

对于 `citation_locate`，答案会输出：

```text
相关内容；页码；标题路径；chunk_id [C1]
```

然后 `citations` 列表里的 `C1` 包含：

```text
chunk_id
doc_id
doc_source
collection_name
page_idx
page_range
heading_path
quote
confidence
```

所以答案中的 `[C1]` 可以通过 `citation_id == C1` 对到具体来源，再通过 `chunk_id` 对到 evidence。

### 当前限制

这个分类策略一次只选一个 type，所以不会同时执行 `TableQASkill + CitationLocateSkill`。像“表格出处”这种复合意图，当前更偏向出处定位。

更理想的设计是：

```text
primary_intent = citation_locate
secondary_intents = [table_qa]
```

然后检索阶段启用 table route，同时答案阶段输出表格证据的位置。这是后续可改进点。

## 4. 为什么当前不需要 ReAct？出于什么考虑？

当前项目不是“不需要 Agent 思想”，而是把 Agent 的决策空间收窄成固定可信 workflow。

原因主要有几个：

### 4.1 任务链路稳定

这个项目的核心任务是企业 PDF 可信问答，主路径相对固定：

```text
分类 -> 扩展 -> 检索 -> 重排 -> 证据门控 -> 回答 -> 引用 -> 保存 trace
```

这里没有太多需要 LLM 自主决定的开放工具选择。例如不像通用 Agent 需要临时决定调用浏览器、数据库、代码解释器、搜索引擎等工具。

### 4.2 可信 QA 更需要可控性

ReAct 的优点是灵活，但缺点是：

- 每轮推理可能不稳定。
- 工具调用路径难以完全预测。
- 延迟和 token 成本更高。
- 更容易出现“看起来调用了工具，但证据不充分”的情况。

本项目强调 citation、evidence、retrieval_trace、rerank_trace，因此固定 workflow 更方便做审计和调试。

### 4.3 当前 skill 只是“策略标签 + trace”，不是可执行工具循环

`skills.py` 里的 skill 定义了：

- 支持哪些 query_type。
- 需要哪些 slots。
- tool_chain 应该是什么。
- guardrails 是什么。
- trace_fields 是什么。

但真正执行还是 `TrustedQAWorkflow.ask()` 写死的统一流程。

所以它更像：

```text
Skill as policy metadata
```

而不是：

```text
Skill as executable agent tool
```

这种做法对面试讲架构是成立的，因为它体现了“先把高风险 RAG 主链路做稳定，再逐步开放 Agent 决策”的工程取舍。

## 5. 在 agent、skills 上未来可以怎么改进？multi-agent 一定必要吗？

### 5.1 可以改进的方向

1. 多标签意图识别

当前是单 query_type。可以改成：

```text
primary_intent + secondary_intents + intent_confidence
```

例如“表格出处”可以是：

```text
primary_intent = citation_locate
secondary_intents = [table_qa]
```

这样检索时启用 table route，回答时按 citation locate 格式输出表格来源。

2. Skill 可执行化

把现在的 `SkillDefinition` 从元数据升级为可执行接口：

```python
class Skill:
    async def plan(context): ...
    async def run(context): ...
    async def validate(output): ...
```

不同技能可以有不同检索策略、slot schema、输出 schema。

3. Query planner

在 workflow 前增加 planner，负责把复杂问题拆成子问题：

```text
用户问题 -> 子查询列表 -> 分别检索 -> 汇总答案
```

适合报告生成、多文档对比、跨章节分析。

4. Evidence-aware answer verifier

当前 evidence gate 在生成前，缺少生成后的 answer verifier。可以增加：

```text
答案 claim 抽取 -> 每个 claim 对齐 evidence/citation -> 不可支持的 claim 删除或降级
```

5. 更细粒度表格 QA

当前表格是 chunk-level 检索。可以增强为：

```text
表格结构解析 -> 行列定位 -> 单元格级证据 -> 计算型问题支持
```

6. Conversation-aware retrieval

当前 session 保存历史，但 workflow 没明显把历史用于 query rewrite。可以增加上下文改写：

```text
当前问题 + 历史对话 -> standalone question
```

7. Skill 级评估指标

不同 skill 应该有不同评估：

- TableQA：指标、数值、单位、期间是否齐全。
- CitationLocate：页码、chunk_id、quote 是否准确。
- Compare：是否覆盖两个以上文档。
- Report：结构完整度和证据覆盖度。

### 5.2 multi-agent 一定必要吗？

不一定。

对这个项目而言，multi-agent 不是第一优先级。因为核心瓶颈不是“缺多个 Agent 角色”，而是：

```text
检索质量、证据覆盖、表格结构理解、答案可验证性、评估闭环
```

什么时候需要 multi-agent？

- 报告生成很复杂，需要“研究员 -> 结构规划 -> 证据审计 -> 写作 -> 复核”多阶段协作。
- 多文档对比需要每个文档一个 analyst，再由 coordinator 汇总。
- 需要把检索、计算、引用审计、合规审查拆成独立角色。

什么时候不需要？

- 普通事实问答。
- 表格定位问答。
- 引用出处查询。
- 单文档摘要。

工程上更推荐演进路线：

```text
固定 workflow
-> 可执行 skill
-> planner + verifier
-> 必要时再 multi-agent
```

不要为了“听起来高级”硬上 multi-agent。面试里能讲清楚这个取舍，反而更像后端架构师。

## 6. 这个项目对当下面试够不够？RAG 会不会过时？

先说结论：这个项目对 LLM 应用开发 / 后端架构 / RAG 工程化面试仍然是有价值的，但你需要把它讲成“可信文档 QA 系统”，不要只讲成“我做了一个 RAG demo”。

### 6.1 现在普通 RAG 确实没那么稀缺了

如果项目只是：

```text
PDF -> chunk -> embedding -> vector search -> prompt -> answer
```

那在今年的面试里确实会显得普通。因为这条链路已经非常标准化，很多候选人都能做。

### 6.2 你的项目比普通 RAG 多的东西

这个代码库里能讲出差异化的点：

1. PDF-only 企业文档场景，不是泛泛聊天。
2. MinerU/PyMuPDF fallback 的文档解析链路。
3. 文本 + 表格结构化 chunk。
4. dense + BM25 + table route 的混合检索。
5. two-stage reranker，包含 metadata boost、table boost、近重复去重、邻居补充。
6. EvidenceGate：证据不足时 retry/refuse/clarify，不是强行回答。
7. citation/evidence/retrieval_trace/rerank_trace/session 保存，强调可审计。
8. 多 query_type 和 skill registry，虽然不是动态 ReAct，但有清晰的技能策略层。
9. LLM fallback：真实 LLM 不可用时仍能跑通，工程鲁棒性不错。
10. pgvector 持久化，而不是纯内存 demo。

这些点已经能支撑一轮不错的项目深挖。

### 6.3 面试时应该怎么讲，避免“过时”

不要说：

```text
我做了一个 RAG 项目，可以上传 PDF 问答。
```

要说：

```text
我做的是企业非结构化 PDF 的可信问答 Agent。重点不是简单向量检索，而是围绕可信输出做了结构化解析、混合检索、表格证据优先、重排、证据门控、引用追踪和会话审计。
```

面试官追问时，你可以重点讲：

- 为什么表格 QA 不能只靠普通 chunk。
- 为什么需要 BM25 + dense，而不是只用向量。
- 为什么 evidence gate 比“直接让 LLM 回答”更可靠。
- confidence 是怎么来的，它的局限是什么。
- 为什么当前没上 full ReAct，是工程可控性考虑。
- 如果继续演进，如何做 planner、verifier、claim-level citation、多标签 intent。

### 6.4 这个项目还差什么会更强？

如果为了面试更稳，优先补这几块：

1. 写一份架构图和核心流程图。
2. 准备一个真实财报表格 QA demo。
3. 加一个“answer verifier / claim citation checker”的设计或实现。
4. 把多关键词意图冲突处理升级为 primary + secondary intents。
5. 准备性能数据：索引耗时、chunk 数、检索耗时、top_k、缓存命中。
6. 准备失败案例：证据不足如何 refuse，表格证据不足如何 retry。

### 6.5 心态上不用怕

RAG 这个词热度会变化，但“企业知识库、文档问答、证据可追溯、结构化检索、表格理解、回答可信性”不会过时。

真正过时的是 demo 级 RAG，不是工程化可信 QA。

你这个项目的方向没问题。接下来要做的是把它从“RAG 项目”包装和打磨成：

```text
面向企业 PDF 的可信文档问答系统，核心能力是证据驱动回答和可审计链路。
```


## 7. 补充：LLM 生成正文时，citation 是怎么引用上的？

你这里问的是更关键的问题：最后正文如果是 `LLMService.generate_grounded_answer()` 生成的，那 LLM 怎么知道某句话该引用 `[C1]` 还是 `[C2]`？

当前实现的答案是：靠 prompt 中显式注入“带 citation_id 的证据块”，不是靠后处理自动对齐。

代码位置：`service/llm/llm_client.py:LLMService.generate_grounded_answer()`。

### 7.1 LLM 生成前，citation 已经先确定了

在进入 LLM 之前，`AnswerGenerator.generate()` 已经根据重排后的 evidence 顺序生成了两份结构：

```text
evidence[0] -> E1
citations[0] -> C1

evidence[1] -> E2
citations[1] -> C2
```

也就是说，`C1/C2/C3` 不是 LLM 临时创造的，而是系统先按 evidence 顺序分配好的。

### 7.2 LLM prompt 里把每条证据显式标成 [C1]

`generate_grounded_answer()` 会遍历 `evidence[:8]`，然后从 `citations[index - 1]` 取 citation_id：

```python
citation_id = citations[index - 1].get("citation_id", f"C{index}")
```

然后拼成 evidence_lines：

```text
[C1] doc=xxx page=3 heading=xxx content=xxx
[C2] doc=xxx page=5 heading=xxx content=xxx
```

接着 system prompt 会要求：

```text
Answer only from the provided evidence.
Every key claim must cite citation ids like [C1].
```

所以 LLM 能知道 citation 的原因是：它看到的上下文不是裸文本，而是已经带标签的证据：

```text
[C1] 证据内容 A
[C2] 证据内容 B
```

LLM 在组织自然语言答案时，根据自己引用了哪条证据内容，把对应的 `[C1]`、`[C2]` 写进正文。

### 7.3 最终 response 里的 citations 不会被 LLM 改写

LLM 只改写：

```python
answer_payload["answer"] = llm_answer
```

但 `answer_payload["citations"]` 和 `answer_payload["evidence"]` 仍然是前面 `AnswerGenerator` 基于检索结果确定好的。

所以最终对应关系是：

```text
正文里的 [C1]
-> response.citations 中 citation_id == C1 的对象
-> 该对象包含 chunk_id/doc_source/page_idx/heading_path/quote
-> response.evidence 中同 chunk_id 的证据内容
```

### 7.4 当前实现的风险

这里有一个重要边界：当前系统没有做“LLM 输出后的 citation 校验”。

也就是说，它相信 LLM 会遵守 prompt，但没有强制验证：

- LLM 是否引用了不存在的 `[C9]`。
- LLM 是否把来自 `[C1]` 的事实错误标成 `[C2]`。
- LLM 是否某个关键 claim 没有引用。
- LLM 是否引用了 `[C1]`，但正文 claim 实际并不能从 `[C1]` 支撑。

所以当前 citation 对齐方式是：

```text
prompt-time citation grounding
```

还不是：

```text
post-generation claim-level citation verification
```

### 7.5 更严谨的改进方式

如果要让这个项目更像生产级可信 QA，后面可以加一个 `AnswerCitationVerifier`：

```text
LLM answer
-> 抽取每个 claim
-> 解析 claim 后面的 [C1]/[C2]
-> 找到对应 evidence content
-> 判断 claim 是否能由该 evidence 支撑
-> 删除/降级 unsupported claim
-> 补齐 missing citation
-> 如果引用不存在，拒答或重写
```

还可以加一个简单的规则校验作为第一步：

1. 正则提取正文里的所有 `[C数字]`。
2. 检查每个 citation id 是否存在于 `response.citations`。
3. 检查 answer 中是否至少引用了一个 citation。
4. 对没有出现在正文里的 citations 做标记，避免前端误以为全部被使用。
5. 对 `decision == answer` 但没有 citation 的情况降级 confidence。

更强的版本可以让 LLM 输出结构化 JSON：

```json
{
  "claims": [
    {
      "text": "公司 2025 年收入为 123 亿元",
      "citation_ids": ["C1"]
    }
  ],
  "final_answer": "... [C1]"
}
```

这样 citation 不是散落在自然语言里，而是可以被程序验证和重组。

### 7.6 面试时应该怎么说

可以这样讲：

```text
当前实现里，citation id 是在检索和重排后由系统按 evidence 顺序确定的。LLM 生成最终答案时，并不是自己发明 citation，而是在 prompt 里看到带 [C1]/[C2] 标签的 evidence blocks，并被要求每个关键结论引用这些 id。最终 response 的 citations 数组仍然由系统保留，用正文里的 [C1] 去反查 citations 数组即可。

但这仍然是 prompt-level grounding，不能百分百保证 claim 和 citation 严格对齐。生产级改进是增加 answer citation verifier，对 LLM 输出后的 claim 和 citation 做逐条校验，发现不存在的 citation、缺 citation 或 claim unsupported 时重写或拒答。
```
