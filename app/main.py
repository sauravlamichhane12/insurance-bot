from __future__ import annotations

import csv
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

import bcrypt
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader
from starlette.middleware.sessions import SessionMiddleware

from .db import UPLOAD_DIR, execute, executescript, fetch_all, fetch_one, ensure_dirs

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT / "public"

load_dotenv(ROOT / ".env")


def parse_csv_env(name: str, fallback: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or fallback


def parse_positive_int_env(name: str, fallback: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    try:
        value = int(raw)
        return value if value > 0 else fallback
    except ValueError:
        return fallback


PORT = int(os.getenv("PORT", "8000"))
APP_ENV = (os.getenv("APP_ENV") or "development").strip().lower()
IS_PRODUCTION = APP_ENV == "production"
ALLOWED_ORIGINS = parse_csv_env("ALLOWED_ORIGINS", ["http://localhost:8000", "http://127.0.0.1:8000"])
MAX_UPLOAD_BYTES = parse_positive_int_env("MAX_UPLOAD_BYTES", 25 * 1024 * 1024)
AUTH_USERNAME = (os.getenv("AUTH_USERNAME") or "").strip()
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD") or ""
SESSION_SECRET = os.getenv("SESSION_SECRET") or ""
PINECONE_API_KEY = (os.getenv("PINECONE_API_KEY") or "").strip()
PINECONE_INDEX = (os.getenv("PINECONE_INDEX") or os.getenv("PINECONE_INDEX_NAME") or "").strip()
PINECONE_NAMESPACE = (os.getenv("PINECONE_NAMESPACE") or "medilink-insurance").strip()
AWS_ACCESS_KEY_ID = (os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") or ""
AWS_REGION = (
    os.getenv("AWS_REGION")
    or os.getenv("aws_region")
    or os.getenv("BEDROCK_AWS_REGION")
    or ""
).strip()
BEDROCK_EVALUATOR_MODEL_ID = (
    os.getenv("BEDROCK_EVALUATOR_MODEL_ID") or "global.anthropic.claude-sonnet-4-6"
).strip()
BEDROCK_EMBEDDING_MODEL_ID = (
    os.getenv("BEDROCK_EMBEDDING_MODEL_ID") or "amazon.titan-embed-text-v2:0"
).strip()
SEMANTIC_MAX_CHARS = 1200

if not SESSION_SECRET:
    raise RuntimeError("Missing required env vars: SESSION_SECRET")
if "*" in ALLOWED_ORIGINS:
    raise RuntimeError("ALLOWED_ORIGINS must list explicit origins; wildcard is not allowed with credentialed sessions")

rag_service: Any | None = None
rag_service_error: str | None = None

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax", https_only=IS_PRODUCTION, max_age=3600)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_error_message(value: Any) -> str:
    return str(value or "unknown error")[:1000]


def _extract_text_blocks(message: Any) -> str:
    if isinstance(message, str):
        return message

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        if parts:
            return "".join(parts)

    blocks = getattr(message, "content_blocks", None) or []
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict):
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        else:
            block_type = getattr(block, "type", None)
            text = getattr(block, "text", None)
            if block_type == "text" and isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def missing_rag_env_vars() -> list[str]:
    missing: list[str] = []
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if not PINECONE_INDEX:
        missing.append("PINECONE_INDEX or PINECONE_INDEX_NAME")
    if not AWS_ACCESS_KEY_ID:
        missing.append("AWS_ACCESS_KEY_ID")
    if not AWS_SECRET_ACCESS_KEY:
        missing.append("AWS_SECRET_ACCESS_KEY")
    if not AWS_REGION:
        missing.append("AWS_REGION")
    return missing


def get_rag_service():
    global rag_service, rag_service_error

    if rag_service is not None:
        return rag_service

    missing = missing_rag_env_vars()
    if missing:
        rag_service_error = f"missing required env vars: {', '.join(missing)}"
        raise HTTPException(status_code=503, detail=f"RAG service unavailable: {rag_service_error}")

    try:
        from .rag_service import RAGConfig, RAGService

        rag_service = RAGService(
            RAGConfig(
                pinecone_api_key=PINECONE_API_KEY,
                pinecone_index_name=PINECONE_INDEX,
                pinecone_namespace=PINECONE_NAMESPACE,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                aws_region=AWS_REGION,
                bedrock_evaluator_model_id=BEDROCK_EVALUATOR_MODEL_ID,
                bedrock_embedding_model_id=BEDROCK_EMBEDDING_MODEL_ID,
                semantic_max_chars=SEMANTIC_MAX_CHARS,
                top_k=4,
            )
        )
        rag_service_error = None
        return rag_service
    except Exception as exc:
        rag_service_error = sanitize_error_message(exc)
        print("RAG service initialization failed:", rag_service_error)
        raise HTTPException(status_code=503, detail=f"RAG service unavailable: {rag_service_error}")


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", name or "document.pdf")
    return cleaned[:180] or "document.pdf"


def sanitize_user(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "authProvider": row["auth_provider"],
        "isActive": bool(row["is_active"]),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "lastLoginAt": row["last_login_at"],
    }


def normalize_boolean(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return fallback


def parse_role(value: Any) -> str | None:
    if value == "admin":
        return "admin"
    if value == "user":
        return "user"
    return None


def active_admin_count() -> int:
    row = fetch_one("SELECT COUNT(*) AS count FROM users WHERE role = ? AND is_active = 1", ("admin",))
    return int(row["count"]) if row else 0


def unusable_password_hash() -> str:
    token = secrets.token_urlsafe(24).encode("utf-8")
    return bcrypt.hashpw(token, bcrypt.gensalt(rounds=12)).decode("utf-8")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    if not password or not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def normalize_login_target(target: str | None, fallback: str = "/chat") -> str:
    if target == "/admin":
        return "/admin"
    if target in {"/chat.html", "/chat", "/"}:
        return "/chat"
    return fallback


def redirect_with_error(path: str, code: str) -> RedirectResponse:
    return RedirectResponse(f"{path}?error={quote(code)}", status_code=302)


def login_user(request: Request, user: dict[str, Any]) -> None:
    request.session["user"] = {"id": user["id"], "username": user["username"], "role": user["role"]}
    execute("UPDATE users SET last_login_at = ?, updated_at = ? WHERE id = ?", (now_iso(), now_iso(), user["id"]))


def create_user_record(
    username: str,
    password: str | None,
    role: str,
    is_active: bool,
    auth_provider: str = "local",
):
    ts = now_iso()
    password_hash = hash_password(password) if password else unusable_password_hash()
    result = execute(
        """
        INSERT INTO users (
          username, password_hash, role, is_active, auth_provider, azure_oid, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (username, password_hash, role, 1 if is_active else 0, auth_provider, None, ts, ts),
    )
    return fetch_one("SELECT * FROM users WHERE id = ?", (result["lastrowid"],))


def authenticate_credentials(username: str, password: str) -> dict[str, Any]:
    row = fetch_one("SELECT * FROM users WHERE username = ?", (username,))
    if not row:
        raise HTTPException(status_code=401, detail="invalid username or password")
    user = dict(row)
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="account is inactive")
    if not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="invalid username or password")
    return user


def bootstrap_db() -> None:
    ensure_dirs()
    executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT NOT NULL UNIQUE,
          password_hash TEXT NOT NULL,
          role TEXT NOT NULL CHECK(role IN ('admin','user')),
          is_active INTEGER NOT NULL DEFAULT 1,
          auth_provider TEXT NOT NULL DEFAULT 'local',
          azure_oid TEXT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          last_login_at TEXT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          original_name TEXT NOT NULL,
          stored_name TEXT NOT NULL,
          stored_path TEXT NOT NULL,
          mime_type TEXT NOT NULL,
          size_bytes INTEGER NOT NULL,
          uploaded_by_user_id INTEGER NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('uploaded','ingesting','ingested','failed')),
          vector_namespace TEXT NULL,
          chunk_count INTEGER NOT NULL DEFAULT 0,
          indexed_at TEXT NULL,
          index_error TEXT NULL,
          flowise_reference TEXT NULL,
          flowise_response TEXT NULL,
          error_message TEXT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        """
    )

    user_columns = {row["name"] for row in fetch_all("PRAGMA table_info(users)")}
    for name, ddl in [
        ("auth_provider", "TEXT NOT NULL DEFAULT 'local'"),
        ("azure_oid", "TEXT NULL"),
    ]:
        if name not in user_columns:
            execute(f"ALTER TABLE users ADD COLUMN {name} {ddl}")
    execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_azure_oid ON users(azure_oid)")

    existing_columns = {row["name"] for row in fetch_all("PRAGMA table_info(documents)")}
    for name, ddl in [
        ("vector_namespace", "TEXT NULL"),
        ("chunk_count", "INTEGER NOT NULL DEFAULT 0"),
        ("indexed_at", "TEXT NULL"),
        ("index_error", "TEXT NULL"),
    ]:
        if name not in existing_columns:
            execute(f"ALTER TABLE documents ADD COLUMN {name} {ddl}")

    admin_count = fetch_one("SELECT COUNT(*) AS count FROM users WHERE role = ? AND is_active = 1", ("admin",))
    if AUTH_USERNAME and AUTH_PASSWORD and (not admin_count or int(admin_count["count"]) == 0):
        try:
            create_user_record(username=AUTH_USERNAME, password=AUTH_PASSWORD, role="admin", is_active=True)
            print("Bootstrap local admin created from env AUTH_USERNAME.")
        except sqlite3.IntegrityError:
            row = fetch_one("SELECT * FROM users WHERE username = ?", (AUTH_USERNAME,))
            if row:
                execute(
                    "UPDATE users SET role = ?, is_active = 1, auth_provider = ?, password_hash = ?, updated_at = ? WHERE id = ?",
                    ("admin", "local", hash_password(AUTH_PASSWORD), now_iso(), row["id"]),
                )
                print("Existing user promoted as bootstrap local admin from env AUTH_USERNAME.")


def require_user(request: Request) -> dict[str, Any]:
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="unauthorized")
    return user


def require_admin(request: Request) -> dict[str, Any]:
    user = require_user(request)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="forbidden")
    return user


def redirect_if_unauthenticated(request: Request, target: str):
    if not request.session.get("user"):
        return RedirectResponse(target, status_code=302)
    return None


ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".csv"}
ALLOWED_UPLOAD_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
}

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def load_csv_rows(path: Path) -> list[list[str]]:
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.reader(handle))
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV encoding is not supported")


def validate_document_upload_metadata(upload: UploadFile) -> None:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="file is required")
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF and CSV files are allowed")

    content_type = (upload.content_type or "").lower().strip()
    if content_type and content_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only PDF and CSV files are allowed")


def save_upload_file(upload: UploadFile, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with destination.open("wb") as buffer:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="Uploaded file is too large")
            buffer.write(chunk)
    return total


def validate_saved_document(path: Path) -> None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            if len(reader.pages) <= 0:
                raise ValueError("PDF has no pages")
            return
        if suffix == ".csv":
            rows = load_csv_rows(path)
            has_content = any(any(str(cell).strip() for cell in row) for row in rows)
            if not has_content:
                raise ValueError("CSV has no content")
            return
        raise ValueError("Unsupported file type")
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF or CSV document") from exc


def index_document_with_pinecone(document_row: dict[str, Any]) -> dict[str, Any]:
    execute(
        "UPDATE documents SET status = ?, vector_namespace = ?, index_error = NULL, error_message = NULL, updated_at = ? WHERE id = ?",
        ("ingesting", PINECONE_NAMESPACE, now_iso(), document_row["id"]),
    )
    try:
        service = get_rag_service()
        result = service.index_document(document_row)
        execute(
            "UPDATE documents SET status = ?, chunk_count = ?, indexed_at = ?, index_error = NULL, error_message = NULL, updated_at = ? WHERE id = ?",
            ("ingested", result["chunk_count"], now_iso(), now_iso(), document_row["id"]),
        )
        return result
    except Exception as exc:
        message = sanitize_error_message(exc)
        execute(
            "UPDATE documents SET status = ?, index_error = ?, error_message = ?, updated_at = ? WHERE id = ?",
            ("failed", message, message, now_iso(), document_row["id"]),
        )
        raise


def index_document_by_id(document_id: int) -> None:
    row = fetch_one("SELECT * FROM documents WHERE id = ?", (document_id,))
    if not row:
        print(f"Skipping ingestion for missing document {document_id}")
        return
    try:
        index_document_with_pinecone(dict(row))
    except Exception as exc:
        print(f"Document {document_id} indexing failed:", sanitize_error_message(exc))


def remove_document_vectors(document_id: int | str) -> None:
    try:
        service = get_rag_service()
        service.delete_document_vectors(document_id)
    except Exception as exc:
        print(f"Failed to delete Pinecone vectors for document {document_id}: {sanitize_error_message(exc)}")


@app.on_event("startup")
def on_startup() -> None:
    bootstrap_db()


def render_file(name: str):
    return FileResponse(PUBLIC_DIR / name)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/login")
def get_login(request: Request):
    if request.session.get("user"):
        return RedirectResponse("/", status_code=302)
    return render_file("login.html")


@app.post("/login")
async def post_login(request: Request, username: str = Form(...), password: str = Form(...), next: str = Form("/chat")):
    try:
        user = authenticate_credentials(username.strip(), password)
        login_user(request, user)
        return RedirectResponse(normalize_login_target(next, "/chat"), status_code=302)
    except HTTPException as exc:
        code = "inactive" if exc.status_code == 403 else "invalid"
        return redirect_with_error("/login", code)
    except Exception as exc:
        print("User login failed:", sanitize_error_message(exc))
        return redirect_with_error("/login", "invalid")


@app.get("/admin/login")
def get_admin_login(request: Request):
    user = request.session.get("user")
    if user:
        return RedirectResponse("/admin" if user.get("role") == "admin" else "/", status_code=302)
    return render_file("admin-login.html")


@app.post("/admin/login")
async def post_admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
    try:
        user = authenticate_credentials(username.strip(), password)
        if user["role"] != "admin":
            return redirect_with_error("/admin/login", "admin")
        login_user(request, user)
        return RedirectResponse("/admin", status_code=302)
    except HTTPException as exc:
        code = "inactive" if exc.status_code == 403 else "invalid"
        return redirect_with_error("/admin/login", code)
    except Exception as exc:
        print("Admin login failed:", sanitize_error_message(exc))
        return redirect_with_error("/admin/login", "invalid")


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@app.get("/")
def home_redirect():
    return RedirectResponse("/chat", status_code=302)


@app.get("/chat")
def chat_page(request: Request):
    redirect_response = redirect_if_unauthenticated(request, "/login")
    if redirect_response:
        return redirect_response
    return render_file("chat.html")


@app.get("/chat.html")
def chat_html_redirect():
    return RedirectResponse("/chat", status_code=301)


@app.get("/admin")
def admin_page(request: Request):
    redirect_response = redirect_if_unauthenticated(request, "/admin/login")
    if redirect_response:
        return redirect_response
    user = require_user(request)
    if user.get("role") != "admin":
        return RedirectResponse("/admin/login?error=admin", status_code=302)
    return render_file("admin.html")


@app.get("/admin.html")
def admin_html_404():
    return render_file("404.html")


@app.get("/api/me")
def api_me(request: Request):
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return user


@app.get("/api/admin/users")
def list_users(request: Request):
    require_admin(request)
    users = [sanitize_user(row) for row in fetch_all("SELECT * FROM users ORDER BY created_at DESC")]
    return {"users": users}


@app.post("/api/admin/users")
def create_user(request: Request, payload: dict[str, Any] = Body(...)):
    require_admin(request)
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    role = parse_role(payload.get("role"))
    is_active = normalize_boolean(payload.get("isActive"), True)

    if not username or len(username) > 120:
        raise HTTPException(status_code=400, detail="username is required and must be <= 120 characters")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="password is required and must be at least 8 characters")
    if not role:
        raise HTTPException(status_code=400, detail="role must be admin or user")

    try:
        created = create_user_record(username=username, password=password, role=role, is_active=is_active, auth_provider="local")
        return JSONResponse({"user": sanitize_user(created)}, status_code=201)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="username already exists")


@app.put("/api/admin/users/{user_id}")
def update_user(request: Request, user_id: int, payload: dict[str, Any] = Body(...)):
    require_admin(request)
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="invalid user id")

    existing = fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if not existing:
        raise HTTPException(status_code=404, detail="user not found")

    username = existing["username"] if "username" not in payload else str(payload.get("username") or "").strip()
    password = None if "password" not in payload else str(payload.get("password") or "")
    role = existing["role"] if "role" not in payload else parse_role(payload.get("role"))
    is_active = bool(existing["is_active"]) if "isActive" not in payload else normalize_boolean(payload.get("isActive"), True)

    if not username or len(username) > 120:
        raise HTTPException(status_code=400, detail="username is required and must be <= 120 characters")
    if password is not None and password and len(password) < 8:
        raise HTTPException(status_code=400, detail="password must be at least 8 characters")
    if not role:
        raise HTTPException(status_code=400, detail="role must be admin or user")

    currently_active_admin = existing["role"] == "admin" and int(existing["is_active"]) == 1
    will_remain_active_admin = role == "admin" and is_active
    if currently_active_admin and not will_remain_active_admin and active_admin_count() <= 1:
        raise HTTPException(status_code=400, detail="cannot demote/deactivate the last active admin")

    try:
        password_hash = existing["password_hash"] if not password else hash_password(password)
        execute(
            "UPDATE users SET username = ?, password_hash = ?, role = ?, is_active = ?, auth_provider = ?, updated_at = ? WHERE id = ?",
            (username, password_hash, role, 1 if is_active else 0, "local", now_iso(), user_id),
        )
        updated = fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
        return {"user": sanitize_user(updated)}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="username already exists")


@app.delete("/api/admin/users/{user_id}")
def delete_user(request: Request, user_id: int):
    require_admin(request)
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="invalid user id")

    target = fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if not target:
        raise HTTPException(status_code=404, detail="user not found")
    if target["role"] == "admin" and int(target["is_active"]) == 1 and active_admin_count() <= 1:
        raise HTTPException(status_code=400, detail="cannot delete the last active admin")

    execute("DELETE FROM users WHERE id = ?", (user_id,))
    return Response(status_code=204)


@app.post("/api/admin/documents/upload")
def upload_document(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    user = require_admin(request)
    validate_document_upload_metadata(file)

    file_name = f"{int(datetime.now().timestamp() * 1000)}-{safe_filename(file.filename)}"
    stored_path = UPLOAD_DIR / file_name
    file_size = save_upload_file(file, stored_path)
    validate_saved_document(stored_path)

    ts = now_iso()
    try:
        result = execute(
            """
            INSERT INTO documents (
              original_name, stored_name, stored_path, mime_type, size_bytes, uploaded_by_user_id,
              status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file.filename,
                file_name,
                str(stored_path),
                file.content_type or "application/octet-stream",
                file_size,
                int(user["id"]),
                "uploaded",
                ts,
                ts,
            ),
        )
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise

    document = fetch_one("SELECT * FROM documents WHERE id = ?", (result["lastrowid"],))
    background_tasks.add_task(index_document_by_id, int(result["lastrowid"]))
    return JSONResponse({"document": dict(document)}, status_code=201)


@app.get("/api/admin/documents")
def list_documents(request: Request):
    require_admin(request)
    documents = [dict(row) for row in fetch_all("SELECT * FROM documents ORDER BY created_at DESC")]
    return {"documents": documents}


@app.post("/api/admin/documents/{document_id}/reingest")
def reingest_document(request: Request, document_id: int):
    require_admin(request)
    if document_id <= 0:
        raise HTTPException(status_code=400, detail="invalid document id")
    row = fetch_one("SELECT * FROM documents WHERE id = ?", (document_id,))
    if not row:
        raise HTTPException(status_code=404, detail="document not found")
    try:
        remove_document_vectors(document_id)
        index_document_with_pinecone(dict(row))
    except Exception as exc:
        print("Re-index failed:", sanitize_error_message(exc))
    updated = fetch_one("SELECT * FROM documents WHERE id = ?", (document_id,))
    return {"document": dict(updated)}


@app.delete("/api/admin/documents/{document_id}")
def delete_document(request: Request, document_id: int):
    require_admin(request)
    if document_id <= 0:
        raise HTTPException(status_code=400, detail="invalid document id")
    row = fetch_one("SELECT * FROM documents WHERE id = ?", (document_id,))
    if not row:
        raise HTTPException(status_code=404, detail="document not found")
    remove_document_vectors(document_id)
    execute("DELETE FROM documents WHERE id = ?", (document_id,))
    try:
        Path(row["stored_path"]).unlink(missing_ok=True)
    except Exception:
        pass
    return Response(status_code=204)


@app.post("/api/chat/stream")
def chat_stream(request: Request, payload: dict[str, Any] = Body(...)):
    require_user(request)
    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    chat_id = str(payload.get("chatId") or "").strip()
    effective_chat_id = chat_id or f"chat-{secrets.token_urlsafe(12)}"
    history = payload.get("history") or []

    def event_stream() -> Iterator[bytes]:
        answer = ""
        try:
            service = get_rag_service()
            for token, _metadata in service.stream_answer(question, history, thread_id=effective_chat_id):
                text = _extract_text_blocks(token)
                if not text:
                    continue
                answer += text
                yield f"data: {json.dumps({'event': 'token', 'data': text}, ensure_ascii=False)}\n\n".encode("utf-8")
            yield (
                f"data: {json.dumps({'event': 'done', 'data': {'ok': True, 'chatId': effective_chat_id, 'answer': answer}}, ensure_ascii=False)}\n\n"
            ).encode("utf-8")
        except Exception as exc:
            message = sanitize_error_message(exc)
            print("Chat stream failed:", message)
            yield f"data: {json.dumps({'event': 'error', 'data': message}, ensure_ascii=False)}\n\n".encode("utf-8")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/{path:path}")
def api_404(path: str):
    raise HTTPException(status_code=404, detail="not found")


@app.get("/{path:path}")
def static_fallback(path: str):
    if path in {"", "/"}:
        return render_file("404.html")
    candidate = PUBLIC_DIR / path
    if candidate.exists() and candidate.is_file():
        return FileResponse(candidate)
    return render_file("404.html")
