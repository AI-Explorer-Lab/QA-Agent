from __future__ import annotations

import os
import sys
import asyncio
from pathlib import Path


def _print(msg: str) -> None:
    # Avoid encoding surprises on some Windows consoles.
    try:
        sys.stdout.write(msg + "\n")
    except Exception:
        print(msg)


async def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from dotenv import load_dotenv
    except Exception as exc:
        _print(f"[PG-CHECK] missing python-dotenv ({exc}). Try: python -m pip install python-dotenv")
        return 2

    load_dotenv(project_root / ".env", encoding="utf-8-sig")

    try:
        from core.config_loader import load_runtime_env

        load_runtime_env()
    except Exception as exc:
        _print(f"[PG-CHECK] failed to load config ({exc})")
        return 2

    database_url = (os.getenv("PGVECTOR_DATABASE_URL") or "").strip()
    if not database_url:
        _print("[PG-CHECK] PGVECTOR_DATABASE_URL is empty.")
        _print("[PG-CHECK] Set it in .env or config/app.yaml (storage.pgvector.database_url).")
        return 2

    try:
        from sqlalchemy import text
        from database import get_async_session
    except Exception as exc:
        _print(f"[PG-CHECK] missing SQLAlchemy ({exc}). Try: python -m pip install sqlalchemy asyncpg")
        return 2

    try:
        async with get_async_session(backend="pgvector", database_url=database_url) as conn:
            info = (await conn.execute(text("select current_database(), current_user"))).fetchone()
            _print(f"[PG-CHECK] connected db={info[0]} user={info[1]}")

            exists = (
                await conn.execute(
                text(
                    """
                    select exists (
                      select 1
                      from information_schema.tables
                      where table_schema='public' and table_name='pdf_chunks'
                    )
                    """
                )
                )
            ).scalar()
            _print(f"[PG-CHECK] pdf_chunks_exists={bool(exists)}")
            if not exists:
                _print("[PG-CHECK] pdf_chunks table not found. Run schema init / build-index first.")
                return 3

            count = (await conn.execute(text("select count(*) from pdf_chunks"))).scalar()
            _print(f"[PG-CHECK] pdf_chunks_count={int(count)}")

            rows = (
                await conn.execute(
                text(
                    """
                    select id, chunk_id, doc_id, left(content, 80) as preview, created_at
                    from pdf_chunks
                    order by id desc
                    limit 5
                    """
                )
                )
            ).fetchall()
            _print(f"[PG-CHECK] latest_rows={len(rows)}")
            for row in rows:
                mapped = dict(row._mapping)
                _print(
                    f"  - id={mapped.get('id')} chunk_id={mapped.get('chunk_id')} "
                    f"doc_id={mapped.get('doc_id')} created_at={mapped.get('created_at')}"
                )
        return 0
    except ModuleNotFoundError as exc:
        _print(f"[PG-CHECK] driver missing: {exc}")
        _print("[PG-CHECK] Fix: python -m pip install asyncpg pgvector")
        _print("[PG-CHECK] Then restart your API / rerun build-index.")
        return 2
    except Exception as exc:
        _print(f"[PG-CHECK] DB error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

