from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _resolve_path(env_name: str, fallback: Path) -> Path:
    raw = (os.getenv(env_name) or "").strip()
    if not raw:
        return fallback
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


DATA_DIR = _resolve_path("APP_DATA_DIR", ROOT / "data")
DB_PATH = _resolve_path("APP_DB_PATH", DATA_DIR / "app.db")
CHECKPOINT_DB_PATH = _resolve_path("APP_CHECKPOINT_DB_PATH", DATA_DIR / "langgraph_checkpoints.db")
UPLOAD_DIR = _resolve_path("APP_UPLOADS_DIR", ROOT / "uploads" / "pdfs")


def ensure_dirs() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_one(sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(sql, params).fetchone()


def fetch_all(sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(sql, params).fetchall()


def execute(sql: str, params: tuple[Any, ...] = ()) -> dict[str, int]:
    with connect() as conn:
        cursor = conn.execute(sql, params)
        return {"lastrowid": cursor.lastrowid or 0, "rowcount": cursor.rowcount}


def executescript(sql: str) -> None:
    with connect() as conn:
        conn.executescript(sql)
