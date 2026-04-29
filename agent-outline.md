# 项目链路与核心梳理（面试复习版）

## 一、项目一句话定位

这是一个面向企业 PDF 文档的“可信问答 Agent”。它不是简单把问题丢给大模型，而是先完成文档解析与索引，再通过混合检索拿证据，用规则门控和语义审计判断证据是否够可靠，最后再生成带引用的答案，并把整条链路落到 session / trace 里，方便追踪与复盘。

## 二、项目主链路

### 1. 文档入库链路

`/documents/index` 进入 `DocumentIndexingService.index_documents()`，依次完成：

1. 收集 PDF 文件
2. 调用 MinerU 做解析/OCR
3. 结构化切块（保留标题层级、页码、表格信息）
4. 为 chunk 生成 embedding
5. 写入 pgvector 与会话侧存储

### 2. 问答链路

`/qa/ask` 进入 `TrustedQAWorkflow.ask()`，依次完成：

1. 加载/创建 session
2. 意图识别，判断这是事实问答、表格问答、总结、对比还是引用定位
3. 槽位补全，检查 metric / period / compare_targets 等关键信息是否缺失
4. 查询扩展
5. 并行混合检索（dense + BM25 + table route）
6. 两阶段 rerank
7. 规则证据门控 + 语义证据审计
8. 不满足就 clarify / retry / refuse
9. 满足后生成答案、引用、证据
10. 持久化消息、检索 trace、评估结果

---

## 三、按模块拆解

## 模块1：API 入口层与应用启动

### 1. 是什么，起到了什么作用

这一层就是系统的“对外接口”和“应用装配层”。它负责把 FastAPI、路由、中间件、异常处理统一挂起来，然后把 `/documents/index`、`/qa/ask`、`/qa/sessions/{session_id}` 这些能力暴露出去。

### 2. 为什么要用它

因为面试时一定要说明：项目不是一个脚本，而是一个可服务化部署的 QA 系统。入口层把“HTTP 请求”稳定地转换成“工作流调用”，也方便后续接前端、灰度发布、监控与测试。

### 3. 作用体现在哪

代码位置：`main.py`、`controller/apis/qa_controller.py`

```python
# main.py
config = get_app_config()
app = FastAPI(title=config.get("app", {}).get("name", "trusted-pdf-qa"))
app.add_middleware(RequestLogMiddleware)
app.add_exception_handler(Exception, app_exception_handler)
app.include_router(router)
```

```python
# controller/apis/qa_controller.py
@router.post("/qa/ask")
async def ask(request: QARequest):
    return await get_trusted_qa_workflow().ask(
        question=request.question,
        session_id=request.session_id or None,
        collection_name=request.collection_name,
        top_k=request.top_k,
        expand_query_num=request.expand_query_num,
        enable_cache=request.enable_cache,
    )
```

### 4. 带来的效果是什么

项目具备标准化 API 形态，能够直接对接业务系统；同时入口和业务工作流解耦，后面想换工作流实现或增加新接口都比较容易。

---

## 模块2：TrustedQAWorkflow 编排层

### 1. 是什么，起到了什么作用

这是整个项目最核心的模块，相当于“总调度器”。它把 session、意图理解、槽位补全、查询扩展、检索、证据门控、答案生成、落库全串起来，决定最终返回 `answer / clarify / refuse` 哪一种结果。

### 2. 为什么要用它

如果没有这一层，项目只会变成一堆离散组件：检索是一个服务，生成又是一个服务，session 又是另一个服务，无法形成完整链路。用工作流编排层，可以把复杂流程稳定地固化下来，也便于后续插入新的 agent 节点。

### 3. 作用体现在哪

代码位置：`service/agent/trusted_qa_workflow.py:69-282`

```python
session = self.session_service.load_session(session_id, collection_name=collection_name)
intent_trace = await self.intent_agent.classify(question)
query_type = str(intent_trace.get("query_type") or "fact_lookup")
selected_skill = self.skill_registry.select_skill(query_type)
slots = await self.slot_agent.fill(question, query_type)
required_slots = required_slots_for_query_type(query_type)
missing_slots = find_missing_slots(slots, required_slots)
```

```python
llm_expanded = await self.llm_service.expand_queries(question, query_type, expand_query_num)
expanded = llm_expanded or expand_queries(question, query_type, expand_query_num)

retrieval_result = await self.retriever.retrieve(
    question=question,
    collection_name=collection_name,
    top_k=max(1, int(top_k)),
    query_type=query_type,
    expand_query_num=max(1, int(expand_query_num)),
    enable_cache=enable_cache,
    expanded_queries=expanded,
)
```

```python
evidence_audit = await self.evidence_audit_agent.audit(...)
rule_gate = self.evidence_gate.evaluate(...)
gate = merge_audit_and_rule_gate(rule_gate, evidence_audit)

answer_payload = self.answer_generator.generate(...)
if decision == "answer":
    llm_answer = await self.llm_service.generate_grounded_answer(...)

self._save(sid, question, response)
```

### 4. 带来的效果是什么

把整个系统从“单点问答能力”升级成“可解释、可追踪、可扩展的 Agent 工作流”。面试时可以强调，这个项目的核心不是某一个模型，而是这条可信问答编排链路。

---

## 模块3：意图识别 + 槽位补全 + 语义审计 Agent

### 1. 是什么，起到了什么作用

这一层是“轻量 Agent 层”。它包含三个角色：

1. `IntentUnderstandingAgent`：识别问题类型
2. `SlotFillingAgent`：补齐关键槽位
3. `EvidenceAuditAgent`：判断证据是否真的覆盖了问题语义

### 2. 为什么要用它

企业文档问答里，真正难的不是“生成一句话”，而是判断用户到底在问什么、缺了什么上下文、检索回来的内容是不是答到了点上。单靠关键词规则不够鲁棒，单靠 LLM 又容易漂，所以这里采用“LLM + 规则 fallback”的混合方案。

### 3. 作用体现在哪

代码位置：`service/agent/controlled_agents.py:183-487`

```python
class IntentUnderstandingAgent:
    async def classify(self, question: str) -> Dict[str, Any]:
        llm_result = await self._classify_with_llm(question)
        if llm_result is not None:
            return llm_result
        query_type = self._fallback_query_type(question)
        return self._build_result(query_type, source="rule_fallback", reason="Matched the closest fixed intent group.")
```

```python
class SlotFillingAgent:
    async def fill(self, question: str, query_type: str) -> Dict[str, Any]:
        rule_slots = self._rule_slots(question, query_type)
        llm_slots = await self._fill_with_llm(question, query_type, rule_slots)
        return _merge_slots(rule_slots, llm_slots)
```

```python
class EvidenceAuditAgent:
    async def audit(...):
        fallback = self._rule_audit(question, query_type, slots, selected_skill, evidence, rerank_trace)
        llm_audit = await self._audit_with_llm(question, query_type, slots, selected_skill, evidence, rerank_trace)
        if llm_audit is None:
            return fallback
        return self._normalize_audit(llm_audit, fallback=fallback)

def merge_audit_and_rule_gate(rule_gate, audit):
    if DECISION_RANK[audit_decision] > DECISION_RANK[rule_decision]:
        final_decision = audit_decision
```

### 4. 带来的效果是什么

系统不再是“检索到就答”，而是先判断问题类型、参数是否完整、证据是否语义覆盖。这样可以显著减少表格题答错、对比题答偏、总结题证据不足还硬答的问题。

---

## 模块4：文档索引链路（解析、切块、Embedding、入库）

### 1. 是什么，起到了什么作用

这是项目的“知识准备链路”。它负责把 PDF 从原始文件变成可检索的结构化 chunk，并把向量和元数据一起写进向量库。

### 2. 为什么要用它

PDF 文档天然不适合直接问答，尤其是带表格、标题层级、分页信息的企业资料。如果不先做结构化解析，检索质量会很差，引用也没法定位到页码/标题路径。

### 3. 作用体现在哪

代码位置：`service/pdf/document_indexer.py:95-430`

```python
pdf_documents = collect_pdf_documents(pdf_path)

payload = self.mineru_client.parse_pdf_to_mineru_json(
    pdf_doc.path,
    use_cache=True,
    force_rebuild=force_rebuild,
)

chunks = self.chunker.chunk_mineru_payload(
    mineru_payload=payload,
    doc_id=doc_id,
    collection_name=collection,
    doc_source=doc_source_value,
)
```

```python
vectors = await self.embedding_service.embed_texts(
    texts,
    use_cache=True,
    chunk_text=True,
    max_concurrency=6,
)

normalized = _normalize_chunk_for_retrieval(base, vectors[index])
all_chunks.extend(normalized_chunks)
```

```python
if force_rebuild:
    indexed_count = replace_collection_chunks(collection, all_chunks)
else:
    indexed_count = upsert_runtime_chunks(all_chunks)
session_result = self.session_service.upsert_collection_chunks(
    collection,
    all_chunks,
    force_rebuild=force_rebuild,
)
```

### 4. 带来的效果是什么

把原始 PDF 变成“可检索、可引用、可追踪”的知识底座，后续问答时能准确拿到页码、标题路径、表格头等结构信息，这是可信回答的前提。

---

## 模块5：并行混合检索

### 1. 是什么，起到了什么作用

这是问答系统的“找证据”模块。它不是只跑向量检索，而是同时跑 dense、BM25、table route，并支持 query expansion、缓存、并发控制和任务级 trace。

### 2. 为什么要用它

企业 PDF 场景里，单一路由很容易失效：

1. 纯 dense 对精确指标、页码、专有名词有时不稳
2. 纯 BM25 对语义改写、同义表达支持差
3. 表格问题需要专门提高 table chunk 的召回率

所以必须做 hybrid retrieval。

### 3. 作用体现在哪

代码位置：`service/retrieval/parallel_query_executor.py:116-242`、`service/retrieval/hybrid_retriever.py`

```python
cache_key = self.retrieval_cache.build_key(
    collection_name=collection_name,
    question_hash=cache_hash,
    query_type=effective_query_type,
    top_k=effective_top_k,
)

if enable_cache:
    cached = self.retrieval_cache.get(cache_key)
```

```python
for variant in query_variants:
    tasks.append(asyncio.create_task(self._run_route_task(route="dense", ...)))
    tasks.append(asyncio.create_task(self._run_route_task(route="bm25", ...)))
    if should_run_table:
        tasks.append(asyncio.create_task(self._run_route_task(route="table", ...)))

raw_task_results = await asyncio.gather(*tasks)
merged_candidates = self._merge_candidates(raw_task_results)
```

```python
# service/retrieval/hybrid_retriever.py
stage1 = await self.parallel_executor.execute(...)
reranked, rerank_trace = self.reranker.rerank(
    query=question,
    candidates=candidates,
    top_k=top_k,
    query_type=query_type,
    table_evidence_quota=self.table_evidence_quota,
)
```

### 4. 带来的效果是什么

召回率和稳定性都更高，尤其是对“表格指标问答、带年份的精确查询、同义表达改写”的鲁棒性明显更好；同时 trace 能记录每条 route 的耗时、命中数和缓存情况，便于排障。

---

## 模块6：两阶段重排（Rerank）

### 1. 是什么，起到了什么作用

这个模块负责把第一阶段召回的一大批候选 chunk 做精排。它综合 dense score、BM25 score、metadata boost、table boost，再做去重、邻居补充和表格配额控制。

### 2. 为什么要用它

因为混合召回只解决“找得到”，但不保证“排得准”。企业文档里常见问题是：召回了一堆相似段落、重复 chunk、同页邻近内容缺失、表格证据排不进前列，所以必须有一层精排逻辑。

### 3. 作用体现在哪

代码位置：`service/retrieval/two_stage_hybrid_reranker.py:145-307`

```python
payload["dense_score"] = _clip01(dense_norm[index])
payload["bm25_score"] = _clip01(bm25_norm[index])
payload["metadata_boost"] = _metadata_boost(query_tokens, page_hint, payload)
payload["table_boost"] = _table_boost(query_tokens, payload, query_type)

payload["final_score"] = _clip01(
    self.dense_weight * payload["dense_score"]
    + self.bm25_weight * payload["bm25_score"]
    + self.metadata_boost_weight * payload["metadata_boost"]
    + self.table_boost_weight * payload["table_boost"]
)
```

```python
if normalized_text and normalized_text in seen_exact:
    continue
if _near_duplicate_score(token_set, existing) >= self.near_duplicate_threshold:
    near_duplicate = True
```

```python
if str(query_type or "") != "table_qa" or quota <= 0:
    return merged[:limit]

table_rows = [row for row in merged if str(row.get("chunk_type") or "") == "table"]
required_tables = min(quota, limit, len(table_rows))
```

### 4. 带来的效果是什么

最终返回的证据更相关、更少重复，而且表格类问题能稳定拿到足够的 table evidence。也就是说，它把“召回结果”变成了“可用于回答的证据集”。

---

## 模块7：证据门控（Guardrails）

### 1. 是什么，起到了什么作用

这是系统的“安全阀”。它根据证据数量、top score、平均分、多文档覆盖、表格证据占比等条件决定是直接回答、重试检索、追问补槽，还是拒答。

### 2. 为什么要用它

可信问答最怕“明明证据不够还硬答”。这个模块的核心意义就是把 hallucination 风险前置拦住，而不是等生成之后再修补。

### 3. 作用体现在哪

代码位置：`service/agent/evidence_gate.py:43-157`

```python
if not rows:
    decision = "retry" if retry_count < self.retry_limit else "refuse"
    return {
        "decision": decision,
        "reason": "no_evidence" if decision == "retry" else "no_evidence_after_retry",
        "confidence": 0.0,
    }
```

```python
if query_type == "table_qa" and table_count < max(1, int(table_evidence_quota)):
    if retry_count < self.retry_limit:
        return {"decision": "retry", "reason": "missing_table_evidence", ...}
    if not slots.get("metric") or not slots.get("period"):
        return {"decision": "clarify", "reason": "table_slots_missing", ...}
```

```python
if top_score < self.evidence_min_top_score or avg_score < self.evidence_min_avg_score:
    if retry_count < self.retry_limit:
        return {"decision": "retry", "reason": "low_score_retry", ...}
    return {"decision": "refuse" if self.refuse_on_low_evidence else "answer", ...}
```

### 4. 带来的效果是什么

系统会更谨慎，宁可 clarify / retry / refuse，也不轻易给出没有依据的答案。面试时可以把它概括成“基于证据阈值的可信问答闸门”。

---

## 模块8：答案生成与引用生成

### 1. 是什么，起到了什么作用

这个模块负责把最终证据转换成用户可读结果，同时生成标准化的 `evidence` 和 `citations`。它本质上是“回答结构化层”。

### 2. 为什么要用它

因为即使检索对了，如果不把证据结构化，就很难展示引用、页码、chunk_id，也不方便下游评估和前端渲染。这个模块把“检索结果”变成“可解释答案载荷”。

### 3. 作用体现在哪

代码位置：`service/agent/answer_generator.py:18-103`

```python
def build_evidence_payload(rows):
    item = {
        "evidence_id": f"E{index}",
        "chunk_id": str(row.get("chunk_id") or ""),
        "doc_source": str(row.get("doc_source") or ""),
        "chunk_type": str(row.get("chunk_type") or "text"),
        "content": content,
        "score": float(row.get("final_score") or row.get("score") or 0.0),
    }
```

```python
def build_citations(evidence):
    citations.append(
        {
            "citation_id": f"C{index}",
            "page_idx": meta.get("page_idx"),
            "page_range": meta.get("page_range", ""),
            "heading_path": meta.get("heading_path", ""),
            "quote": _clip(item.get("content", ""), 260),
        }
    )
```

```python
evidence_payload = build_evidence_payload(evidence)
citations = build_citations(evidence_payload)

if decision == "clarify":
    ...
if decision == "refuse":
    ...
if query_type == "table_qa":
    answer = self._table_answer(evidence_payload)
```

### 4. 带来的效果是什么

答案不再只是自然语言文本，而是带证据、带引用、带置信度的结构化输出，更符合企业级问答系统对可解释性的要求。

---

## 模块9：LLM 服务层

### 1. 是什么，起到了什么作用

这一层封装了真实大模型调用，承担三类职责：

1. 查询扩展
2. 语义分类/槽位/审计等 Agent 调用的底层 `complete`
3. 基于证据的 grounded answer 生成

### 2. 为什么要用它

因为项目里 LLM 不是直接裸调，而是需要统一处理 provider、model、Responses API / Chat Completions 切换、代理、超时、可用性检测、错误追踪等问题。

### 3. 作用体现在哪

代码位置：`service/llm/llm_client.py:139-363`

```python
self.model = (
    os.getenv("TRUSTED_QA_LLM_MODEL")
    or str(model_cfg.get("model") or llm_cfg.get("model") or selector or "gpt-4o-mini")
)
self.enabled = _env_truthy("TRUSTED_QA_ENABLE_REAL_LLM", default_enabled)
self.use_responses_api = bool(llm_cfg.get("use_responses_api", provider_cfg.get("use_responses_api", False)))
```

```python
if self.use_responses_api and hasattr(client, "responses"):
    response = await client.responses.create(...)

response = await client.chat.completions.create(...)
```

```python
async def expand_queries(...):
    system_prompt = "You rewrite enterprise PDF QA questions for hybrid retrieval."

async def generate_grounded_answer(...):
    system_prompt = (
        "You are a trusted enterprise PDF QA agent. Answer only from the provided evidence. "
        "Every key claim must cite citation ids like [C1]."
    )
```

### 4. 带来的效果是什么

LLM 能力被约束在“增强检索”和“基于证据生成”两个可信边界内，而不是让模型直接自由发挥。这能兼顾能力和可靠性。

---

## 模块10：Session / Trace 持久化

### 1. 是什么，起到了什么作用

这是项目的“可追踪记忆层”。它把用户消息、助手回复、检索 trace、评估结果一起写进 PostgreSQL，支持后续查看 session、复盘检索链路、做离线评估。

### 2. 为什么要用它

因为企业场景里，能回答还不够，还要能解释“为什么这么答”“当时检索到了什么”“用的是什么证据”。没有 trace 持久化，系统就不可审计、不可回溯。

### 3. 作用体现在哪

代码位置：`service/session/session_service.py:38-197`

```python
def save_session(...):
    user_message_id = str(uuid.uuid4())
    assistant_message_id = str(uuid.uuid4())
    trace_id = _clean_str(trace.get("trace_id")) or str(uuid.uuid4())
```

```sql
INSERT INTO qa_messages (
    message_id, session_id, role, query_type, question, answer, decision,
    confidence, citations_json, evidence_json, metadata_json,
    retrieval_trace_id, created_at
) VALUES (...)
```

```sql
INSERT INTO retrieval_traces (
    trace_id, session_id, message_id, collection_name, question,
    expanded_queries_json, retrieval_trace_json, rerank_trace_json,
    selected_candidates_json, created_at
) VALUES (...)
```

### 4. 带来的效果是什么

每次问答都能被复盘，便于做线上问题排查、效果评估、用户会话回放，这一点在面试里非常加分，因为它体现了“工程化闭环”。

---

## 模块11：运行时存储与 pgvector 仓库

### 1. 是什么，起到了什么作用

这是项目的“向量存储抽象层”。`runtime.py` 负责构造运行时 repository，`PgvectorRepository` 负责 dense_search、keyword_search、table_search、chunk upsert 等核心存储能力。

### 2. 为什么要用它

因为上层工作流不应该直接依赖 SQL 细节。仓储层把“怎么存”“怎么查”统一收口，工作流只关心“我要取证据”。

### 3. 作用体现在哪

代码位置：`service/retrieval/runtime.py`、`service/retrieval/pgvector_repository.py`

```python
def _build_runtime_repository() -> PgvectorRepository:
    config = get_app_config()
    backend = get_storage_backend(config).strip().lower() or "pgvector"
    if backend != "pgvector":
        raise RuntimeError("Runtime repository must use pgvector.")

    return PgvectorRepository(
        backend="pgvector",
        database_url=database_url,
        embedding_dim=_configured_embedding_dim(config),
    )
```

```python
def dense_search(...):
    return self._dense_search_pgvector(...)

def keyword_search(...):
    return self._keyword_search_pgvector(...)

def table_search(...):
    return self.keyword_search(..., chunk_type="table", table_only=True)
```

```python
chunk_sql = text("""
    INSERT INTO pdf_chunks (..., metadata_json, embedding, created_at, updated_at)
    VALUES (..., CAST(:metadata_json AS jsonb), CAST(:embedding AS vector), NOW(), NOW())
    ON CONFLICT (chunk_id) DO UPDATE SET ...
""")
```

### 4. 带来的效果是什么

整个系统的向量检索底座稳定落在 pgvector 上，既支持真实数据库运行，也保留了相对清晰的抽象边界，方便后续替换或扩展。

---

## 模块12：Embedding 与配置中心

### 1. 是什么，起到了什么作用

这两个模块分别负责：

1. `EmbeddingService`：把文本转成向量，支持真实 embedding provider 和 deterministic fallback
2. `config_loader`：统一加载项目默认配置与 YAML 配置

### 2. 为什么要用它

Embedding 是检索的基础；配置中心则是项目工程化的基础。没有 embedding，dense retrieval 不成立；没有统一配置，模型、检索参数、阈值、缓存策略都会失控。

### 3. 作用体现在哪

代码位置：`service/embedding/embedding_service.py`、`utils/config_loader.py`

```python
class EmbeddingService:
    async def embed_text(self, text: str, use_cache: bool = True, chunk_text: bool = False) -> List[float]:
        normalized = normalize_whitespace(text, preserve_newlines=False)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        vector = await self._embed_with_provider(normalized)
```

```python
async def embed_texts(...):
    coroutines = [self.embed_text(text, use_cache=use_cache, chunk_text=chunk_text) for text in text_list]
    results = await bounded_gather(...)
```

```python
DEFAULT_CONFIG: dict[str, Any] = {
    "agent": {"orchestration": "trusted_qa_workflow"},
    "embedding": {"provider": "qwen", "dimension": 1024},
    "storage": {"backend": "pgvector"},
    "retrieval": {"strategy": "hybrid", "top_k": 5, "expand_query_num": 3},
    "guardrails": {"evidence_min_top_score": 0.45, "retry_limit": 2},
}
```

### 4. 带来的效果是什么

检索能力可配置、可切换、可缓存；系统运行参数集中统一，不需要把模型名、阈值、存储地址硬编码在业务逻辑里，维护成本明显更低。

---

## 四、这个项目的核心亮点总结

如果面试官问“你觉得这个项目最核心的地方是什么”，可以总结成下面 4 句话：

1. 这不是普通 RAG，而是“可信问答工作流”，核心在于证据优先，而不是模型优先。
2. 它把文档解析、结构化切块、混合检索、证据门控、答案生成、trace 持久化串成了完整闭环。
3. 它对企业 PDF 场景做了针对性增强，特别是表格问答、多文档对比、引用定位这几类难题。
4. 它的工程化做得比较完整，既能服务化提供 API，也能回溯每次问答的证据和决策过程。

## 五、面试时建议的讲法

可以按下面顺序讲：

1. 先用一句话定义项目：企业 PDF 的可信问答 Agent。
2. 再讲两条链路：文档入库链路、问答链路。
3. 然后重点展开三个核心：混合检索、证据门控、可追踪工作流。
4. 最后补充工程化亮点：session/trace、配置中心、pgvector 持久化。

这样讲会比按文件名背代码更像一个真正做过项目的人。
