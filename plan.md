# 企业 PDF 文档可信问答 Agent 重构计划

## Summary

根据你补充的 6 条要求，本项目重新定位为：

**企业 PDF 非结构化文档可信问答 Agent**

场景只处理 PDF，不再兼容旧聊天机器人逻辑，也不保留 `/askLLM`、`/team-leader-task`。项目主线是：面向制度 PDF、技术手册 PDF、经营分析/管理报表 PDF，构建一个支持结构化解析、并行检索、缓存加速、混合排序、证据门控和引用溯源的可信问答 Agent。

完整链路为：

`PDF 导入 -> MinerU 解析 -> 标题/表格结构化分块 -> 1024 维 Embedding -> pgvector 存储 -> 并行混合检索 -> 缓存命中 -> 多路候选合并 -> 新排序算法 -> 证据门控 -> 引用约束回答/澄清/拒答 -> 会话与评估记录`

核心设计原则：

- 只做 PDF 场景。
- 只保留新 Agent API。
- 以 ReAct Agent + Agent Skills 作为项目主线，把 PDF 业务问答能力封装成可路由、可执行、可追踪的技能。
- 默认使用 PostgreSQL + pgvector。
- YAML 支持数据库/模型/并发/缓存/排序参数切换。
- 检索策略固定为混合检索，不再暴露 `faiss/pgvector/hybrid` 三种索引模式。
- Embedding 维度统一为 1024。
- 查询扩展后的多个 query 并发执行，并加入缓存和并发控制；这些属于支撑 Agent Skills 稳定执行的工程能力。

## Key Changes

### 1. 删除旧接口，重建 API

- 删除或停止暴露旧接口：
  - `/askLLM`
  - `/team-leader-task`
  - 旧的通用聊天入口
- 新项目只保留以下 API：
  - `GET /health`
  - `POST /documents/index`
  - `POST /qa/ask`
  - `GET /qa/sessions/{session_id}`
- `POST /documents/index` 只接受 PDF：
  - `pdf_path`: 单个 PDF 文件路径，或 PDF 目录。
  - `force_rebuild`: 是否强制重建。
  - `collection_name`: 文档集合名称。
- `POST /qa/ask`：
  - `question`
  - `session_id`
  - `collection_name`
  - `top_k`
  - `expand_query_num`
  - `enable_cache`
- 响应必须包含：
  - `answer`
  - `decision`: `answer | clarify | refuse`
  - `query_type`
  - `confidence`
  - `citations`
  - `evidence`
  - `retrieval_trace`
  - `rerank_trace`
  - `session_id`

### 2. PDF Only 文档处理链路

- 文档输入只支持 PDF，不再规划 TXT/DOCX。
- PDF 解析统一走 MinerU：
  - 上传 PDF。
  - 获取 MinerU JSON。
  - 按 `page_idx + para_blocks.index` 恢复阅读顺序。
  - 提取标题、正文、列表、表格。
- 标题恢复：
  - `第X节` -> L1。
  - `一、` -> L2。
  - `（一）` -> L3。
  - `1、2、3` 作为普通枚举，不进入标题层级。
- chunk 策略：
  - 文本按标题路径聚合后，用 token 递归切分。
  - 表格单独成 chunk，不并入正文。
  - 超长表格按行拆分，每个子表重复表头。
  - 所有 chunk 的 embedding 维度固定为 1024。
- chunk 元数据：
  - `chunk_id`
  - `doc_id`
  - `collection_name`
  - `doc_source`
  - `page_idx`
  - `page_range`
  - `chunk_type`: `text | table`
  - `chunk_index`
  - `heading_path`
  - `level1_title`
  - `level2_title`
  - `level3_title`
  - `table_id`
  - `sub_table_id`
  - `table_header_text`
  - `table_context_text`

### 3. pgvector 优先，YAML 可切换

- 默认数据库使用 PostgreSQL + pgvector。
- 不再把 FAISS、pgvector、hybrid 作为三种“索引模式”暴露给业务层。
- 业务层只有一种检索策略：`hybrid_retrieval`。
- YAML 中可切换的是底层存储与运行配置：
  - `storage.backend: pgvector | local_dev`
  - 默认 `pgvector`
  - `local_dev` 只作为无 PostgreSQL 环境下的开发 fallback，不作为简历主能力描述。
- pgvector schema 使用 1024 维：
  - `embedding vector(1024)`
- PostgreSQL 表设计：
  - `pdf_documents`
  - `pdf_chunks`
  - `qa_sessions`
  - `qa_messages`
  - `retrieval_traces`
  - `evaluation_records`
- `pdf_chunks` 保存：
  - 原文内容。
  - 1024 维 embedding。
  - JSON 元数据。
  - 可检索文本字段。
  - 可选 `tsvector` 字段，用于关键词/稀疏检索。

### 4. 并行查询、并发控制与缓存

- Query Expansion 生成多个检索 query 后并行执行。
- 并行检索包含：
  - 原始问题 dense retrieval。
  - 扩展问题 dense retrieval。
  - 原始问题 keyword/BM25 retrieval。
  - 扩展问题 keyword/BM25 retrieval。
  - 表格意图下的 table-prioritized retrieval。
- 并发控制：
  - 使用 `asyncio.gather` 并发执行查询任务。
  - 使用 `asyncio.Semaphore` 控制最大并发数。
  - YAML 配置：
    - `retrieval.max_concurrency`
    - `retrieval.query_timeout_seconds`
    - `retrieval.expand_query_num`
- 缓存设计：
  - Query embedding cache：避免重复 embedding。
  - Retrieval result cache：相同问题、集合、top_k、query_type 下短期复用。
  - Document parse cache：相同 PDF hash 不重复 MinerU 解析。
  - Chunk embedding cache：相同 chunk 内容不重复向量化。
- 缓存 key：
  - `collection_name`
  - `question_hash`
  - `query_type`
  - `top_k`
  - `chunk_hash`
  - `pdf_file_hash`
- YAML 配置：
  - `cache.enabled`
  - `cache.ttl_seconds`
  - `cache.max_items`
  - `cache.embedding_cache_enabled`
  - `cache.retrieval_cache_enabled`
  - `cache.document_parse_cache_enabled`

### 5. 固定混合检索，采用工程可落地的两阶段排序

检索策略固定为 Hybrid Retrieval，不再提供三选一模式。

候选来源：

- Dense retrieval：
  - pgvector cosine similarity。
- Sparse retrieval：
  - 只使用 BM25。
  - BM25 负责召回制度条款、财报指标、型号参数、表头字段等精确关键词。
- Metadata retrieval：
  - 标题路径命中。
  - 文档名、页码、章节名命中。
- Table-aware retrieval：
  - 表格类问题优先召回 `chunk_type=table`。
  - 使用 `table_header_text`、`table_context_text` 辅助匹配。

排序算法采用更轻量、可解释、易实现的 **Two-Stage Hybrid Reranker**。

第一阶段：候选召回。

- 对原始 query 和扩展 query 并行执行 pgvector dense retrieval。
- 对原始 query 和扩展 query 并行执行 BM25 retrieval。
- 每路召回保留 `top_n` 候选，例如 `top_n = top_k * 4`。
- 合并候选后按 `chunk_id` 去重，保留每个 chunk 在不同召回路中的最高分。

第二阶段：归一化融合排序。

- 将 dense score 和 BM25 score 分别归一化到 `0-1`。
- 使用简化公式计算最终分数：

```text
final_score =
  0.50 * dense_score
+ 0.35 * bm25_score
+ 0.10 * metadata_boost
+ 0.05 * table_boost
```

各项含义：

- `dense_score`：pgvector 语义相似度。
- `bm25_score`：BM25 关键词分数，解决财报指标、制度条款、型号参数等精确词命中问题。
- `metadata_boost`：标题路径、章节名、文档名、页码命中时加分。
- `table_boost`：仅在 `query_type=table_qa` 时启用，表头、表格上下文、指标名命中时加分。

排序后执行：

- chunk 去重和近重复文本过滤。
- 对 top 候选做同页邻近 chunk 补充，用于防止单个 chunk 截断上下文。
- 表格问题保证至少返回 `table_evidence_quota` 条 table evidence；如果 table evidence 不足，进入证据门控重试。
- 多文档对比问题保证每个目标 PDF 至少有候选证据；不足时触发澄清或拒答。
- 最终输出 top-k evidence。
### 6. Agent 工作流

主工作流：

```text
START
-> load_session
-> classify_query_type
-> clarify_gate
-> select_skill_from_registry
-> execute_skill_tool_chain
-> parallel_hybrid_retrieval
-> two_stage_hybrid_rerank
-> evidence_gate
-> answer_with_citations / clarify / refuse
-> evaluate
-> save_session
-> END
```

问题类型：

- `fact_lookup`
- `table_qa`
- `summarization`
- `citation_locate`
- `report_generation`
- `multi_doc_compare`
- `ambiguous_query`

证据门控：

- 缺少文档范围、年份、指标、对象时触发澄清。
- 检索结果少于阈值时触发重试。
- top score 或 avg score 低于阈值时触发重试。
- 表格问题没有命中 table evidence 时触发重试。
- 关键结论无法绑定 citation 时拒答或说明不确定。
- 多文档证据冲突时列出差异，不强行生成单一结论。

答案生成要求：

- 每个关键结论必须绑定 citation。
- 不允许使用证据外事实。
- 表格问题输出：
  - 指标
  - 数值
  - 单位
  - 期间
  - 来源
- 原文定位问题优先返回：
  - 原文片段
  - 页码
  - 标题路径
  - chunk_id

### 7. Agent Skills 设计：用业务能力包装 ReAct Agent

本项目的核心亮点不只是 RAG 检索，而是把 **业务场景 + ReAct 编排 + Skills 能力封装** 组合成一个 Agent 开发项目。ReAct Agent 不直接面对一堆底层函数，而是通过 Skill Registry 选择合适的业务技能；每个 skill 内部再调用检索、重排、证据校验和生成工具。

Skill 抽象字段：

- `skill_name`：技能名称。
- `query_types`：该技能支持的问题类型。
- `required_slots`：执行前必须具备的信息，例如文档范围、年份、指标、对比对象。
- `input_schema`：技能输入结构。
- `tool_chain`：技能内部调用的工具链，例如 query expansion、dense retrieval、BM25 retrieval、rerank、evidence gate。
- `output_schema`：技能输出结构，必须包含 answer/citations/evidence/decision。
- `guardrails`：技能级证据门控策略。
- `trace_fields`：写入 retrieval_trace 和 skill_trace 的关键过程字段。

v1 设计 6 个业务 Skills：

- `FactLookupSkill`：事实查询技能，用于制度条款、技术手册步骤、经营报告事实类问题；强调短答案和强引用。
- `TableQASkill`：表格问答技能，用于收入、毛利率、预算、成本、型号参数等表格指标；优先检索 table chunk，并校验指标、数值、单位、期间、来源。
- `CitationLocateSkill`：原文定位技能，用于“出处在哪里”“原文怎么说”；重点返回页码、标题路径、chunk_id 和原文片段。
- `SummarizationSkill`：总结归纳技能，用于按章节或文档范围总结 PDF 内容；先召回多段证据，再按主题合成。
- `ReportGenerationSkill`：报告生成技能，用于基于 PDF 生成结构化摘要、风险点、要点清单；先生成提纲，再分节检索和合成。
- `MultiDocCompareSkill`：多 PDF 对比技能，用于制度版本、手册版本、经营报告之间的差异比较；按文档分别检索证据，再做字段对齐和差异输出。

ReAct 与 Skills 的关系：

```text
用户问题
-> Query Type Classifier
-> Clarify Gate
-> Skill Registry 选择业务 Skill
-> ReAct Agent 执行 skill.tool_chain
-> Observation 记录每一步检索、重排、门控结果
-> Evidence Gate 判断 answer / retry / clarify / refuse
-> Answer Generator 输出带 citation 的结构化答案
```

这里的“skills”不是简单函数列表，而是业务能力单元。它能支撑简历表达为“基于 ReAct 的多技能 Agent 编排”，比单纯写 RAG 更贴近 Agent 开发岗位。v1 不强行声称已经实现多 Agent；如果后续要扩展，可以把 `TableQASkill`、`ReportGenerationSkill`、`MultiDocCompareSkill` 演进成独立 specialist agents，但当前计划只承诺可落地的 single ReAct Agent + multi-skill architecture。
## Project Architecture

```markdown
Enterprise-Unstructured-Document-Trusted-Question-Answering-Agent
├── config
│   └── yaml
│       ├── agent
│       │   └── agent.yaml
│       ├── db
│       │   └── database.yaml
│       └── environment
│           └── app.yaml
├── constant
│   ├── routes.py
│   ├── query_types.py
│   ├── retrieval.py
│   └── error_codes.py
├── domain
│   ├── document.py
│   ├── chunk.py
│   ├── qa.py
│   ├── citation.py
│   ├── retrieval.py
│   └── agent_state.py
├── controller
│   └── apis
│       ├── health_controller.py
│       ├── document_controller.py
│       ├── qa_controller.py
│       └── session_controller.py
├── middlewares
│   ├── request_log.py
│   ├── trace_context.py
│   └── exception_handler.py
├── exception
│   ├── base_exception.py
│   └── business_exception.py
├── mapper
│   ├── document_mapper.py
│   ├── chunk_mapper.py
│   ├── citation_mapper.py
│   ├── retrieval_trace_mapper.py
│   └── session_mapper.py
├── service
│   ├── pdf
│   │   ├── pdf_loader.py
│   │   ├── mineru_client.py
│   │   ├── mineru_parser.py
│   │   ├── heading_recovery.py
│   │   └── structured_chunker.py
│   ├── embedding
│   │   ├── embedding_service.py
│   │   └── embedding_cache.py
│   ├── retrieval
│   │   ├── hybrid_retriever.py
│   │   ├── parallel_query_executor.py
│   │   ├── pgvector_repository.py
│   │   ├── sparse_retriever.py
│   │   ├── two_stage_hybrid_reranker.py
│   │   └── retrieval_cache.py
│   ├── agent
│   │   ├── trusted_qa_workflow.py
│   │   ├── query_classifier.py
│   │   ├── query_expander.py
│   │   ├── clarify_gate.py
│   │   ├── evidence_gate.py
│   │   └── answer_generator.py
│   ├── evaluation
│   │   └── ragas_evaluator.py
│   └── session
│       └── session_service.py
├── utils
│   ├── config_loader.py
│   ├── content_normalizer.py
│   ├── token_counter.py
│   ├── hash_utils.py
│   └── async_utils.py
├── database
│   ├── init_db.py
│   ├── pgvector_schema.sql
│   └── connection.py
├── log
├── scripts
│   ├── workflow_acceptance.py
│   ├── e2e_acceptance.py
│   └── pgvector_smoke.py
├── tests
│   ├── test_pdf_chunking.py
│   ├── test_parallel_retrieval.py
│   ├── test_reranker.py
│   ├── test_evidence_gate.py
│   └── test_qa_api.py
├── Dockerfile
├── README.md
├── .gitignore
├── requirements.txt
└── main.py
```

## Architecture Diagram

```mermaid
flowchart TD
    A["User / API Client"] --> B["FastAPI Controllers"]

    B --> C["POST /documents/index"]
    B --> D["POST /qa/ask"]
    B --> E["GET /qa/sessions/{session_id}"]

    C --> F["PDF Loader"]
    F --> G["MinerU Client"]
    G --> H["MinerU JSON Parser"]
    H --> I["Heading Recovery"]
    I --> J["Structured PDF Chunker"]

    J --> K["Text Chunks"]
    J --> L["Table Chunks"]

    K --> M["Embedding Service 1024D"]
    L --> M

    M --> N["PostgreSQL + pgvector"]
    N --> O["pdf_documents"]
    N --> P["pdf_chunks vector(1024)"]
    N --> Q["qa_sessions / qa_messages"]

    D --> R["Trusted QA Agent Workflow"]
    R --> S["ReAct Agent Core"]
    S --> T["Query Type Classifier"]
    T --> U["Clarify Gate"]
    U --> V0["Skill Registry / Skill Router"]
    V0 --> V1["FactLookupSkill"]
    V0 --> V2["TableQASkill"]
    V0 --> V3["CitationLocateSkill"]
    V0 --> V4["Summarization / Report / Compare Skills"]
    V1 --> V["Parallel Query Executor"]
    V2 --> V
    V3 --> V
    V4 --> V
    V --> W["Dense pgvector Retrieval"]
    V --> X["BM25 Retrieval"]
    V --> Y["Metadata / Table Retrieval"]

    W --> Z["Candidate Merge"]
    X --> Z
    Y --> Z

    Z --> AA["Two-Stage Hybrid Reranker"]
    AA --> AB["Evidence Gate"]

    AB --> AC["Retry Retrieval"]
    AB --> AD["Clarify / Refuse"]
    AB --> AE["Citation-Grounded Answer Generator"]

    AC --> V
    AE --> AF["RAGAS-compatible Evaluation"]
    AD --> AF

    AF --> AG["Session + Trace Store"]
    AG --> Q

    AE --> B
    AD --> B
```

## YAML Defaults

```yaml
app:
  name: Enterprise-Unstructured-Document-Trusted-Question-Answering-Agent

llm:
  current_model: anyrouter-gpt-5.3-codex

agent:
  orchestration: react_with_skills
  max_iterations: 6
  skill_trace_enabled: true
  default_skill: fact_lookup

skills:
  enabled:
    - fact_lookup
    - table_qa
    - citation_locate
    - summarization
    - report_generation
    - multi_doc_compare
  clarify_before_skill: true
  fallback_skill: fact_lookup
embedding:
  provider: qwen
  model: text-embedding-v4
  dimension: 1024

storage:
  backend: pgvector
  pgvector:
    database_url: postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/trusted_qa
    embedding_dim: 1024
  local_dev:
    enabled: false

pdf:
  parser: mineru
  max_pages_per_task: 200

chunking:
  chunk_size_tokens: 1024
  chunk_overlap_tokens: 200
  max_chunk_size: 7000
  heading_max_level: 3

retrieval:
  strategy: hybrid
  top_k: 5
  expand_query_num: 3
  max_concurrency: 6
  query_timeout_seconds: 20
  table_evidence_quota: 2

reranker:
  dense_weight: 0.50
  bm25_weight: 0.35
  metadata_boost_weight: 0.10
  table_boost_weight: 0.05
  top_n_factor: 4
  near_duplicate_threshold: 0.90

cache:
  enabled: true
  ttl_seconds: 3600
  max_items: 5000
  embedding_cache_enabled: true
  retrieval_cache_enabled: true
  document_parse_cache_enabled: true

guardrails:
  evidence_min_docs: 2
  evidence_min_top_score: 0.45
  evidence_min_avg_score: 0.30
  retry_limit: 2
  refuse_on_low_evidence: true
```

## Test Plan

- PDF 处理测试：
  - 单 PDF 可完成 MinerU 解析。
  - 多 PDF 目录可批量解析。
  - 标题层级恢复正确。
  - 表格 chunk 保留表头、上下文、页码、table_id。
  - 所有 embedding 维度为 1024。
- pgvector 测试：
  - 自动建表。
  - `pdf_chunks.embedding` 为 `vector(1024)`。
  - chunk 可 upsert。
  - 同一 PDF 重新索引可覆盖旧 chunk。
- 并发与缓存测试：
  - 多个 expanded query 并行执行。
  - 并发数不超过 YAML 配置。
  - 相同 query 第二次命中 retrieval cache。
  - 相同 chunk 不重复 embedding。
- 混合检索与排序测试：
  - dense-only 命中弱但关键词强的查询，可通过 sparse score 排上来。
  - 表格问题必须优先返回 table chunk。
  - 重复 chunk 会被降权或去重。
  - 多文档对比问题能召回不同 PDF 的证据。
- Skills/ReAct 编排测试：
  - 不同 query_type 能路由到正确 skill。
  - 每个 skill 输出统一的 answer/citations/evidence/decision schema。
  - ReAct trace 能记录 selected_skill、tool_chain、observation、gate_decision。
  - 表格、原文定位、多文档对比问题不会退化成通用 RAG 生成。
- Agent 测试：
  - 事实查询返回答案和引用。
  - 表格查询返回指标、数值、单位、期间、来源。
  - 信息不足问题触发澄清。
  - 低证据问题触发重试。
  - 重试仍失败时拒答。
- 验收脚本：
  - `scripts/workflow_acceptance.py`
  - `scripts/e2e_acceptance.py`
  - `scripts/pgvector_smoke.py`

## Assumptions

- 项目 v1 只支持 PDF。
- 旧聊天机器人 API 不保留。
- pgvector 是默认和简历主线能力。
- YAML 可以切换底层存储，但业务检索策略始终是 hybrid retrieval。
- Embedding 维度统一为 1024。
- 排序算法采用 Two-Stage Hybrid Reranker，在 dense + BM25 召回基础上做少量元数据和表格加权，避免 v1 设计过重。
- 项目核心简历亮点是“PDF 业务场景 + ReAct Agent + Agent Skills + 可信证据门控”；并行查询、缓存、BM25 和 pgvector 是支撑 Agent Skills 稳定执行的工程能力。
- v1 明确实现 single ReAct Agent + multi-skill architecture，不夸大为完整 multi-agent；后续可将高价值 skills 演进为 specialist agents。




