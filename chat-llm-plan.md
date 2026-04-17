# Chat-llm 技术方案（当前执行版）

## 1. 目标与边界
- 保持原有业务逻辑不变：`chat`、`rag`、`react`、`tools(chat/query/rag)` 路由与职责不变。
- 本轮重点做架构与工程化增强，不改产品行为语义。
- 统一配置入口为 `config/app.yaml`，环境变量作为覆盖层。

## 2. 架构改造总览
- 配置层：新增 `provider + model` 组合选择（如 `anyrouter-gpt-5.3-codex`）。
- 模型层：全项目 `ChatOpenAI/OpenAI` 调用统一走 `core/config_loader.py`。
- 检索层：保留 `faiss`，新增 `pgvector`，支持 `faiss/pgvector/hybrid` 路由。
- 分块层：文档先恢复 `L1/L2/L3` 标题树，再按标题路径做语义分段与 token 递归切分；表格独立按行拆分。
- 评估层：新增 `ragas` 兼容评估入口（无依赖时自动回退到可解释指标）。

## 3. 分块策略（重点）

### 3.1 参数
- `chunk_size_tokens = 1024`
- `chunk_overlap_tokens = 200`
- `max_chunk_size = 7000`

### 3.2 JSON 处理（MinerU 输出）
- 解析顺序：按 `pdf_info.page_idx + para_blocks.index` 顺序遍历段落与表格。
- 标题恢复：
  - 优先读取 MinerU `json` 的 `type=title` 版式块。
  - 用“`第X节` / `一、` / `（一）`”三级规则恢复 `L1/L2/L3`。
  - `1、2、3...` 视为普通枚举项，不参与主标题层级，避免污染二级结构。
- 段落：
  - 先按 `L1 -> L2 -> L3` 标题路径做语义边界切分。
  - 每个语义段内再用 `RecursiveCharacterTextSplitter.from_tiktoken_encoder()` 切分。
  - 当标题块缺失时，再退回段落正则恢复标题。
- 表格：
  - 表格不进入正文递归切分。
  - 若表格 token 超限，按“行级”拆分子表，不在行中间截断。
  - 每个子表重复注入表头。
  - 每个子表 token 必须 `< max_chunk_size`。
  - 表格继承当前 `L1/L2/L3` 标题元数据，但不并入正文 chunk。

### 3.3 元数据
- 文本 chunk：
  - `chunk_id, doc_id, doc_source, chunk_type, chunk_index`
  - `level1_title, level2_title, level3_title, heading_path`
- 表格 chunk：在文本字段基础上再加
  - `table_id`
  - `sub_table_id`
  - `sub_table_index`
  - `table_id_subtable_count`
  - 可选：`table_anchor_text, table_anchor_confidence`

## 4. 向量与数据库方案

### 4.1 向量后端
- 保留 `faiss`。
- 新增 `pgvector`。
- 路由支持：
  - `faiss`：仅 FAISS 检索。
  - `pgvector`：仅 PG 检索（失败时回退 FAISS）。
  - `hybrid`：双路召回 + 去重 + 相似度排序。

### 4.2 PostgreSQL 设计
- 建表：`rag_chunks`
- 字段：
  - `id`（主键）
  - `chunk_id`（唯一索引）
  - `doc_id`（索引）
  - `doc_source`
  - `content`
  - `metadata_json`（保存 `level1_title/level2_title/level3_title/heading_path` 与表格元数据）
  - `embedding vector(<dim>)`
  - `created_at`
- 机制：
  - 启动时按配置初始化扩展与 schema（`CREATE EXTENSION IF NOT EXISTS vector`）。
  - 写入采用 upsert，保证重复构建可覆盖更新。
  - 物理表结构保持稳定，章节与表格扩展字段统一放入 `metadata_json`，减少后续迁移成本。

## 5. 配置规范
- `llm.current_model`：`<provider>-<model_key>`
- `llm.providers.<provider>.models.<model_key>`：定义具体模型与参数
- `vector.backend`：`faiss | pgvector | hybrid`
- `vector.pgvector_database_url`
- `vector.pgvector_embedding_dim`
- `chunking.chunk_size_tokens / chunk_overlap_tokens / max_chunk_size`

## 6. 你的 7 条新增需求对应落实
1. **不改原逻辑**：已落实。仅重构实现与配置读取方式，工具职责与流程保持。
2. **RAGAS + token 切分**：已落实。新增 `ragas` 评估入口；切分使用 `.from_tiktoken_encoder()`，`1024/200`。
3. **max_chunk_size=7000**：已落实（配置与代码一致）。
4. **MinerU JSON 段落/表格策略**：已落实（段落递归切分、表格行级拆分+表头复用+上限约束）。
5. **chunk 元数据**：已落实（文本/表格字段齐全，含子表计数）。
   - 新版文本元数据为 `level1_title, level2_title, level3_title, heading_path`。
6. **faiss + pgvector 共存 + PG 设计**：已落实（后端共存、路由、表结构与 upsert 机制）。
7. **参考 Q&A速记.md**：已吸收为“业务逻辑不变 + 架构增强 + 可验收脚本”原则。

## 7. 验收标准
- 文本与表格 chunk 都带齐规定元数据。
- 目录/单文件两种输入路径都可构建索引。
- `workflow_acceptance.py` 与 `e2e_acceptance.py` 通过。
- `pgvector` 在数据库可连通时可完成建表、写入与检索。

## 8. 变更清单（落点）
- `core/config_loader.py`
- `config/app.yaml`
- `chunking_service/structured_chunking.py`
- `chunking_service/heading_recovery.py`
- `chunking_service/document_processor.py`
- `db_service/faiss_store.py`
- `db_service/pgvector_store.py`
- `db_service/vector_store_router.py`
- `workflow/rag_workflow.py`
- `workflow/react_workflow.py`
- `workflow/team_leader_workflow.py`
- `api/routes.py`
- `scripts/workflow_acceptance.py`
- `scripts/e2e_acceptance.py`
- `test/docx_body_xml_dump.py`
