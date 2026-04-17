# Chat-llm 重构后使用指引

## 1. 先看这 4 个文件
1. `config/app.yaml`：所有运行配置入口（LLM、向量后端、chunk 参数）。
2. `core/config_loader.py`：把 YAML 映射到运行时环境变量。
3. `db_service/faiss_store.py`：从文档切分->embedding->向量库落盘的主入口。
4. `api/routes.py`：HTTP 调用入口（含构建索引 API）。

## 2. 当前架构（不改业务逻辑）
- 保留原逻辑：`chat`、`rag`、`react`、`tools(chat/query/rag)`。
- 统一模型配置：按 `provider + model` 选择具体模型。
- 向量后端：`faiss`、`pgvector`、`hybrid` 三种。
- 分块：按“第X节”聚合正文后 token 切分；表格按行拆分并带 lineage 元数据。
- 评估：RAG 结果返回 `ragas_evaluation` 字段。

## 3. YAML 配置方法（推荐）

编辑 `config/app.yaml`：

- 模型选择：
  - `llm.current_model: anyrouter-gpt-5.3-codex`
  - 代表调用 `llm.providers.anyrouter.models.gpt-5.3-codex`
- 向量后端：
  - `vector.backend: faiss | pgvector | hybrid`
- pgvector 连接：
  - `vector.pgvector_database_url: postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/chat_llm`
- 分块参数：
  - `chunking.chunk_size_tokens: 1024`
  - `chunking.chunk_overlap_tokens: 200`
  - `chunking.max_chunk_size: 7000`

## 4. 如何把文档从 chunking -> embedding -> 存向量库

### 4.1 命令行（推荐）

1. 目录输入（自动识别 pdf/txt）：
```bash
python db_service/faiss_store.py --document-path ./docs/pdf_docs --type auto
```

2. 单文件输入（兼容）：
```bash
python db_service/faiss_store.py --document-path ./docs/test_docs/test.txt --type auto
```

3. 指定后端（覆盖 YAML 当次运行）：
```bash
python db_service/faiss_store.py --document-path ./docs/pdf_docs --type auto --backend faiss
python db_service/faiss_store.py --document-path ./docs/pdf_docs --type auto --backend pgvector
python db_service/faiss_store.py --document-path ./docs/pdf_docs --type auto --backend hybrid
```

### 4.2 API 方式

接口：`POST /vector/build-index`

可传参数：
- `document_path`（目录或单文件）
- `file_type`（`pdf|txt`）
- `backend`（可选，`faiss|pgvector|hybrid|both`）

示例：
```bash
curl -X POST "http://127.0.0.1:8000/vector/build-index?document_path=./docs/pdf_docs&file_type=pdf&backend=pgvector"
```

## 5. 检索到底走 FAISS 还是 PG？

由 `vector.backend` 决定：
- `faiss`：只查 FAISS。
- `pgvector`：优先查 PG；PG 不可用时自动回退 FAISS。
- `hybrid`：FAISS + PG 双路召回，去重后按相似度排序。

## 6. pgvector 使用前提

你需要先有可连接的 PostgreSQL（安装 `pgvector` 扩展），并保证：
- 数据库可连：`127.0.0.1:5432`
- URL 正确写在 `vector.pgvector_database_url`

当前代码会自动执行：
- `CREATE EXTENSION IF NOT EXISTS vector`
- 自动建表 `rag_chunks`
- upsert 写入 chunk 与 embedding

如果你本机没有 PostgreSQL，可先用 Docker 起一个（示例）：
```bash
docker run --name chat-llm-pgvector -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=chat_llm -p 5432:5432 -d pgvector/pgvector:pg16
```

可用一条命令做 PG 烟测：
```bash
python scripts/pgvector_smoke.py --document-path ./docs/test_docs/test.txt --type auto --query 测试检索 --top-k 3
```

## 7. 验收命令

```bash
python -m compileall chunking_service db_service core api workflow scripts
python scripts/workflow_acceptance.py
python scripts/e2e_acceptance.py --port 8015 --http-timeout 120
```

## 8. 常见问题

1. **路径不存在**
- 现已兼容“目录”和“单文件”两种输入。
- 相对路径会优先按当前工作目录解析，找不到再按项目根目录解析。

2. **卡在 `Processing PDF`**
- 这是 MinerU 上传与异步抽取过程；大文档会等待。
- 本次已切换为 MinerU JSON 版式解析，避免 docx 依赖并提升结构一致性。

3. **pgvector 连接失败**
- 不是代码异常，通常是 PostgreSQL 未启动或 URL 配置错误。
- 失败时系统会保留 FAISS 成果，不会整链路中断。
