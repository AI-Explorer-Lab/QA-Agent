项目背景
这个项目现在的定位是：企业 PDF 非结构化文档可信问答 Agent。它不是普通聊天机器人，也不是简单 RAG demo，而是针对制度 PDF、技术手册 PDF、经营报表 PDF 这类企业文档，做“可检索、可排序、可引用、可拒答、可追踪”的问答系统。新入口只挂载新 Agent API，旧 /askLLM、/team-leader-task 已在测试中验证为 404。

1. API 与配置模块
方法：用 FastAPI 只暴露 /health、/documents/index、/qa/ask、/qa/sessions/{session_id}，入口在 main.py (line 36)，路由在 controller/apis。

为什么：简历项目需要主线清晰，不能让旧聊天接口和新 PDF Agent 混在一起。API 缩窄后，项目目标非常明确：导入 PDF，然后基于证据回答。

效果：调用链更干净，E2E 脚本可以完整验证 PDF index、问答、session、trace；旧接口不会误导面试官。

2. PDF 解析与结构化分块
方法：PDF 只走 MinerU 风格解析，按 page_idx + block_index 恢复阅读顺序；再做标题恢复、文本 chunk、表格 chunk 独立切分。核心文件是 service/pdf/mineru_client.py (line 36)、service/pdf/heading_recovery.py (line 68)、service/pdf/structured_chunker.py (line 33)。

为什么：企业 PDF 的难点不是只有“文字很多”，而是标题层级、页码、表格、指标、上下文关系都很重要。直接把 PDF 切成普通文本块，会丢掉章节路径和表格语义。

效果：每个 chunk 都携带 heading_path、page_idx、chunk_type、table_header_text 等元数据，后续检索和 citation 能回到具体页码、章节和表格来源。

3. Embedding 与存储模块
方法：Embedding 维度统一为 1024，pgvector schema 使用 embedding vector(1024)，本地开发环境提供 local_dev fallback。核心在 service/embedding/embedding_service.py (line 20)、database/pgvector_schema.sql (line 24)、service/retrieval/pgvector_repository.py。

为什么：简历项目要能落地，不能只停留在内存 demo。pgvector 适合展示企业级 RAG 的持久化、索引、向量检索能力；local fallback 则保证没有 PostgreSQL 时也能测试主链路。

效果：pgvector_smoke 已验证 schema，E2E 可以在本地 fallback 下跑通，部署时也能切回 PostgreSQL + pgvector。

4. 检索排序模块，重点
方法：固定采用 Hybrid Retrieval，不再暴露 FAISS/pgvector/hybrid 三种业务模式。查询先扩展，再由 ParallelQueryExecutor (line 107) 并发执行 dense、BM25、table-prioritized 多路召回；候选进入 TwoStageHybridReranker (line 124) 做二阶段融合排序：

final_score =
  0.50 * dense_score
+ 0.35 * bm25_score
+ 0.10 * metadata_boost
+ 0.05 * table_boost
为什么：企业 PDF 问答里，单纯向量检索不够。制度条款、财务指标、型号参数、年份、表头字段经常依赖精确词；而用户问题又可能是语义化表达。所以 dense 负责语义召回，BM25 负责关键词召回，metadata 负责章节/标题/文档名加权，table boost 负责表格问答。

效果：排序结果可解释，rerank_trace 会输出权重、候选数、表格证据数、top score。它比黑盒 reranker 更适合简历项目展示，因为你能讲清楚每个分数来自哪里、为什么能提升可信度。

5. 并发、缓存与可观测检索
方法：Query expansion 后的多个 query 并发跑 dense/BM25/table 路线，使用 semaphore 控制并发，timeout 控制慢查询，retrieval cache 避免重复请求。实现集中在 parallel_query_executor.py (line 143) 和 service/retrieval/retrieval_cache.py。

为什么：真实企业文档检索不是一次 query 一个结果，通常需要多个改写、多路召回、多文档候选。没有并发会慢，没有缓存会重复算，没有 trace 就无法排查为什么答错。

效果：E2E 输出里能看到 task_trace、cache_hit、query_variants、max_concurrency、每路 returned 数量；这对面试时解释工程稳定性很有帮助。

6. ReAct + Skill 模块，重点
方法：主工作流在 TrustedQAWorkflow (line 26)。流程是：加载 session -> 问题分类 -> 选择 skill -> query expansion -> hybrid retrieval -> rerank -> evidence gate -> answer/clarify/refuse -> evaluation -> 保存 session。Skill 注册在 service/agent/skill_registry.py (line 8)，业务技能定义在 service/agent/skills.py (line 33)。

为什么：这个项目的亮点不只是 RAG，而是“PDF 业务能力 Agent”。FactLookup、TableQA、CitationLocate、Summarization、ReportGeneration、MultiDocCompare 这些不是随便放几个函数，而是按业务问题类型封装成 Skill，每个 Skill 有自己的 tool_chain、guardrails 和 trace 字段。

效果：最终响应里会有 skill_trace 和 react_observations，能清楚看到 Agent 选择了哪个 Skill、执行了哪些工具、证据门控如何判断。这让项目更像 Agent 开发项目，而不是普通知识库问答。

7. 证据门控与可信回答
方法：EvidenceGate (line 28) 根据证据数量、top score、avg score、表格证据、文档覆盖等条件决定 answer/retry/clarify/refuse。答案生成要求每个关键结论绑定 citation；真实 LLM 可用时走 generate_grounded_answer (line 164)，不可用时走确定性 fallback。

为什么：企业场景里“编一个答案”比“不回答”更危险。可信问答系统必须知道什么时候证据不足，什么时候需要澄清，什么时候拒答。

效果：输出不仅有 answer，还有 citations、evidence、retrieval_trace、rerank_trace、confidence。这使回答可以追溯，也能解释为什么系统敢回答或不敢回答。

8. 测试与简历表达
方法：测试覆盖 PDF chunking、embedding cache、并行检索、缓存、reranker、evidence gate、API、LLM YAML 配置优先级。新增测试在 tests/test_llm_yaml_config.py。

为什么：简历项目不能只说“实现了”，要能证明链路可运行、配置可切换、核心算法有断言。

效果：现在可以在简历里写成：

重构企业 PDF 可信问答 Agent，基于 MinerU 结构化解析、pgvector 1024D 向量存储、并行 Hybrid Retrieval、Two-Stage Hybrid Reranker、Evidence Gate 与 ReAct + Skill 编排，实现带 citation、trace、session 和评估记录的可信 PDF QA 系统。全链路通过单元测试、E2E 验收和 pgvector schema smoke test。