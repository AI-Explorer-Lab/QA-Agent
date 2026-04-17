from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.init_db import init_local_dev_schema


def main() -> None:
    schema = Path("database/pgvector_schema.sql").read_text(encoding="utf-8")
    assert "pdf_chunks" in schema
    assert "vector(1024)" in schema
    sqlite_path = init_local_dev_schema("sqlite:///database/local_dev_smoke.db")
    print({"pgvector_schema": "ok", "local_dev_schema": str(sqlite_path)})


if __name__ == "__main__":
    main()
