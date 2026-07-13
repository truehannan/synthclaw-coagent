from __future__ import annotations
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, date
from cryptography.fernet import Fernet
from config import DB_PATH, BASE_DIR

logger = logging.getLogger(__name__)

KEY_FILE = BASE_DIR / ".fernet_key"
CONTEXT_DIR = BASE_DIR / "context"

# ── D1 backend routing ───────────────────────────────────────────────────────
# When D1 is configured, all DB operations go through d1_storage module.
# Local SQLite remains as offline fallback.

_use_d1: bool = False
_d1 = None  # lazy import


def _init_d1_backend():
    """Try to initialize D1 backend from environment/config."""
    global _use_d1, _d1
    import os
    account_id = os.getenv("CF_ACCOUNT_ID", "").strip()
    database_id = os.getenv("CF_D1_DATABASE_ID", "").strip()
    api_token = os.getenv("CF_API_TOKEN", "").strip()
    storage_mode = os.getenv("STORAGE_MODE", "local").strip().lower()

    if storage_mode == "cloudflare" and account_id and database_id and api_token:
        try:
            import d1_storage
            d1_storage.configure(account_id, database_id, api_token)
            if d1_storage.init_d1():
                _d1 = d1_storage
                _use_d1 = True
                logger.info("D1 storage backend active")
                return
        except Exception as e:
            logger.warning(f"D1 init failed, falling back to local SQLite: {e}")
    _use_d1 = False


# Initialize D1 on module load (non-blocking — falls back silently)
try:
    _init_d1_backend()
except Exception:
    _use_d1 = False


def _get_conn():
    """Get a SQLite connection (convenience helper for api_server)."""
    return sqlite3.connect(DB_PATH)


def get_fernet() -> Fernet:
    if KEY_FILE.exists():
        key = KEY_FILE.read_bytes()
    else:
        key = Fernet.generate_key()
        KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        KEY_FILE.write_bytes(key)
        KEY_FILE.chmod(0o600)
    return Fernet(key)


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY,
            chat_id   INTEGER,
            role      TEXT,
            content   TEXT,
            ts        DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS credentials (
            id          INTEGER PRIMARY KEY,
            name        TEXT UNIQUE,
            enc_value   BLOB,
            description TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS memory (
            key        TEXT PRIMARY KEY,
            value      TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS config (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            id        INTEGER PRIMARY KEY,
            chat_id   INTEGER,
            summary   TEXT,
            msg_count INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS model_usage (
            id            INTEGER PRIMARY KEY,
            model         TEXT NOT NULL,
            input_tokens  INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            ts            DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS task_state (
            chat_id    INTEGER PRIMARY KEY,
            messages   TEXT NOT NULL,
            media      TEXT NOT NULL,
            model      TEXT NOT NULL,
            step_count INTEGER NOT NULL DEFAULT 0,
            attempt_step INTEGER NOT NULL DEFAULT 0,
            stall_count INTEGER NOT NULL DEFAULT 0,
            last_sig   TEXT,
            last_error TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS long_term_facts (
            id INTEGER PRIMARY KEY,
            chat_id INTEGER NOT NULL,
            fact TEXT NOT NULL,
            importance INTEGER NOT NULL DEFAULT 1,
            source TEXT DEFAULT 'auto',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(chat_id, fact)
        );
    """)

    # Backward-compatible migration for existing DBs
    c.execute("PRAGMA table_info(task_state)")
    existing_cols = {row[1] for row in c.fetchall()}
    if "attempt_step" not in existing_cols:
        c.execute("ALTER TABLE task_state ADD COLUMN attempt_step INTEGER NOT NULL DEFAULT 0")
    if "stall_count" not in existing_cols:
        c.execute("ALTER TABLE task_state ADD COLUMN stall_count INTEGER NOT NULL DEFAULT 0")
    if "last_sig" not in existing_cols:
        c.execute("ALTER TABLE task_state ADD COLUMN last_sig TEXT")
    if "last_error" not in existing_cols:
        c.execute("ALTER TABLE task_state ADD COLUMN last_error TEXT")

    # Dynamic API tools table
    c.execute("""
        CREATE TABLE IF NOT EXISTS dynamic_tools (
            id          INTEGER PRIMARY KEY,
            name        TEXT UNIQUE NOT NULL,
            base_url    TEXT NOT NULL,
            auth_type   TEXT NOT NULL DEFAULT 'bearer',
            auth_header TEXT DEFAULT 'Authorization',
            auth_prefix TEXT DEFAULT 'Bearer ',
            auth_cred   TEXT NOT NULL,
            endpoints   TEXT NOT NULL DEFAULT '[]',
            description TEXT DEFAULT '',
            docs_url    TEXT DEFAULT '',
            enabled     INTEGER DEFAULT 1,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Skill sources table — tracks where skills come from for reinstall
    c.execute("""
        CREATE TABLE IF NOT EXISTS skill_sources (
            id          INTEGER PRIMARY KEY,
            name        TEXT UNIQUE NOT NULL,
            source_type TEXT NOT NULL,
            source_uri  TEXT NOT NULL,
            version     TEXT DEFAULT '',
            auto_update INTEGER DEFAULT 1,
            installed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def add_long_term_fact(chat_id: int, fact: str, importance: int = 1, source: str = "auto"):
    """Insert/update a durable memory fact with dedupe per chat."""
    if _use_d1:
        return _d1.add_long_term_fact(chat_id, fact, importance, source)
    clean = (fact or "").strip()
    if not clean:
        return
    imp = max(1, min(int(importance or 1), 5))
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO long_term_facts (chat_id, fact, importance, source) VALUES (?,?,?,?)",
        (chat_id, clean, imp, source),
    )
    c.execute(
        "UPDATE long_term_facts SET importance = MAX(importance, ?) WHERE chat_id=? AND fact=?",
        (imp, chat_id, clean),
    )
    conn.commit()
    conn.close()


def get_long_term_facts(chat_id: int, limit: int = 40) -> list[dict]:
    """Return durable memory facts ordered by importance then recency."""
    if _use_d1:
        return _d1.get_long_term_facts(chat_id, limit)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT fact, importance, source, created_at FROM long_term_facts "
        "WHERE chat_id=? ORDER BY importance DESC, created_at DESC LIMIT ?",
        (chat_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    return [
        {"fact": r[0], "importance": r[1], "source": r[2], "created_at": r[3]}
        for r in rows
    ]


# ── Conversation history ────────────────────────────────────────────────────

def get_history(chat_id: int, limit: int = 20) -> list[dict]:
    if _use_d1:
        return _d1.get_history(chat_id, limit)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE chat_id=? ORDER BY ts DESC LIMIT ?",
        (chat_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


def save_message(chat_id: int, role: str, content: str):
    if _use_d1:
        return _d1.save_message(chat_id, role, content)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?,?,?)",
        (chat_id, role, content),
    )
    conn.commit()
    conn.close()


def clear_history(chat_id: int):
    if _use_d1:
        return _d1.clear_history(chat_id)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    conn.commit()
    conn.close()


# ── Task state (checkpoint / resume) ──────────────────────────────────────

def save_task_state(
    chat_id: int,
    messages: list,
    media: list,
    model: str,
    step_count: int,
    attempt_step: int = 0,
    stall_count: int = 0,
    last_sig: str | None = None,
    last_error: str | None = None,
):
    """Persist an in-progress agent loop so it can be resumed after a checkpoint."""
    if _use_d1:
        return _d1.save_task_state(chat_id, messages, media, model, step_count, attempt_step, stall_count, last_sig, last_error)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO task_state (chat_id, messages, media, model, step_count, attempt_step, stall_count, last_sig, last_error)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (
            chat_id,
            json.dumps(messages),
            json.dumps(media),
            model,
            step_count,
            attempt_step,
            stall_count,
            last_sig,
            last_error,
        ),
    )
    conn.commit()
    conn.close()


def load_task_state(chat_id: int) -> dict | None:
    """Return saved task state or None if nothing is saved for this chat."""
    if _use_d1:
        return _d1.load_task_state(chat_id)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT messages, media, model, step_count, attempt_step, stall_count, last_sig, last_error "
        "FROM task_state WHERE chat_id=?",
        (chat_id,),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "messages":   json.loads(row[0]),
        "media":      json.loads(row[1]),
        "model":      row[2],
        "step_count": row[3],
        "attempt_step": row[4] if row[4] is not None else row[3],
        "stall_count": row[5] if row[5] is not None else 0,
        "last_sig": row[6],
        "last_error": row[7],
    }


def clear_task_state(chat_id: int):
    """Remove any saved checkpoint for this chat (called on fresh start or successful finish)."""
    if _use_d1:
        return _d1.clear_task_state(chat_id)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM task_state WHERE chat_id=?", (chat_id,))
    conn.commit()
    conn.close()


# ── Credentials (encrypted) ─────────────────────────────────────────────────

def store_credential(name: str, value: str, description: str = ""):
    f = get_fernet()
    enc = f.encrypt(value.encode())
    if _use_d1:
        # Store base64 string in D1 (can't store raw bytes)
        import base64
        _d1.store_credential(name, base64.b64encode(enc).decode(), description)
        # Also store locally as fallback
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO credentials (name, enc_value, description) VALUES (?,?,?)", (name, enc, description))
            conn.commit()
            conn.close()
        except Exception:
            pass
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO credentials (name, enc_value, description) VALUES (?,?,?)",
        (name, enc, description),
    )
    conn.commit()
    conn.close()


def get_credential(name: str) -> str | None:
    f = get_fernet()
    if _use_d1:
        enc_b64 = _d1.get_credential(name)
        if enc_b64:
            import base64
            try:
                enc = base64.b64decode(enc_b64)
                return f.decrypt(enc).decode()
            except Exception:
                pass
        # Fallback to local
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT enc_value FROM credentials WHERE name=?", (name,))
    row = c.fetchone()
    conn.close()
    return f.decrypt(row[0]).decode() if row else None


def list_credentials() -> list[dict]:
    if _use_d1:
        return _d1.list_credentials()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, description, created_at FROM credentials")
    rows = c.fetchall()
    conn.close()
    return [{"name": r[0], "description": r[1], "created_at": r[2]} for r in rows]


# ── Memory (key-value) ───────────────────────────────────────────────────────

def set_memory(key: str, value: str):
    if _use_d1:
        _d1.set_memory(key, value)
        # Also store locally
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO memory (key, value) VALUES (?,?)", (key, value))
            conn.commit()
            conn.close()
        except Exception:
            pass
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO memory (key, value) VALUES (?,?)", (key, value)
    )
    conn.commit()
    conn.close()


def get_memory(key: str) -> str | None:
    if _use_d1:
        val = _d1.get_memory(key)
        if val is not None:
            return val
        # Fallback to local
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value FROM memory WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def get_all_memory() -> dict:
    if _use_d1:
        return _d1.get_all_memory()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT key, value FROM memory")
    rows = c.fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


# ── App config ───────────────────────────────────────────────────────────────

def get_config(key: str, default=None) -> str | None:
    if _use_d1:
        val = _d1.get_config(key, default)
        if val is not None:
            return val
        # Fallback to local
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value FROM config WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else default


def set_config(key: str, value: str):
    if _use_d1:
        _d1.set_config(key, value)
        # Also store locally
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?,?)", (key, value))
            conn.commit()
            conn.close()
        except Exception:
            pass
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?,?)", (key, value))
    conn.commit()
    conn.close()


# ── Conversation summaries ───────────────────────────────────────────────────

def count_messages(chat_id: int) -> int:
    if _use_d1:
        return _d1.count_messages(chat_id)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM messages WHERE chat_id=?", (chat_id,))
    count = c.fetchone()[0]
    conn.close()
    return count


def get_oldest_messages(chat_id: int, limit: int) -> list[dict]:
    """Get the oldest N messages for a chat (for summarization)."""
    if _use_d1:
        return _d1.get_oldest_messages(chat_id, limit)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, role, content FROM messages WHERE chat_id=? ORDER BY ts ASC LIMIT ?",
        (chat_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "role": r[1], "content": r[2]} for r in rows]


def delete_messages_by_ids(msg_ids: list[int]):
    """Delete messages by their IDs (after summarization)."""
    if not msg_ids:
        return
    if _use_d1:
        return _d1.delete_messages_by_ids(msg_ids)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    placeholders = ",".join("?" for _ in msg_ids)
    c.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", msg_ids)
    conn.commit()
    conn.close()


def save_summary(chat_id: int, summary: str, msg_count: int):
    """Save a conversation summary (cumulative — replaces previous)."""
    if _use_d1:
        return _d1.save_summary(chat_id, summary, msg_count)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Keep only one summary per chat — replace on update
    c.execute("DELETE FROM conversation_summaries WHERE chat_id=?", (chat_id,))
    c.execute(
        "INSERT INTO conversation_summaries (chat_id, summary, msg_count) VALUES (?,?,?)",
        (chat_id, summary, msg_count),
    )
    conn.commit()
    conn.close()


def get_summary(chat_id: int) -> str | None:
    """Get the latest conversation summary for a chat."""
    if _use_d1:
        return _d1.get_summary(chat_id)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT summary FROM conversation_summaries WHERE chat_id=? ORDER BY created_at DESC LIMIT 1",
        (chat_id,),
    )
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


# ── Markdown context files ───────────────────────────────────────────────────
# Persistent long-term context stored as .md files on disk.
#
#   context/<chat_id>/
#     profile.md      — owner facts, preferences, identity
#     summary.md      — running conversation summary (auto-updated)
#     sessions/
#       YYYY-MM-DD.md — per-day session log with key events
#

def _chat_dir(chat_id: int) -> Path:
    d = CONTEXT_DIR / str(chat_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sessions_dir(chat_id: int) -> Path:
    d = _chat_dir(chat_id) / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Profile ──

def get_profile(chat_id: int) -> str:
    """Read the owner profile markdown."""
    p = _chat_dir(chat_id) / "profile.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def save_profile(chat_id: int, content: str):
    """Overwrite the owner profile markdown."""
    p = _chat_dir(chat_id) / "profile.md"
    p.write_text(content, encoding="utf-8")


def append_to_profile(chat_id: int, line: str):
    """Add a line to the profile if it's not already there."""
    p = _chat_dir(chat_id) / "profile.md"
    existing = p.read_text(encoding="utf-8") if p.exists() else ""
    if line.strip() not in existing:
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"\n- {line.strip()}\n")


# ── Summary ──

def get_md_summary(chat_id: int) -> str:
    """Read the running conversation summary."""
    p = _chat_dir(chat_id) / "summary.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def save_md_summary(chat_id: int, content: str):
    """Overwrite the running summary markdown."""
    p = _chat_dir(chat_id) / "summary.md"
    p.write_text(content, encoding="utf-8")


# ── Session logs ──

def get_today_session(chat_id: int) -> str:
    """Read today's session log."""
    p = _sessions_dir(chat_id) / f"{date.today().isoformat()}.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def append_to_session(chat_id: int, entry: str):
    """Append an entry to today's session log."""
    p = _sessions_dir(chat_id) / f"{date.today().isoformat()}.md"
    now = datetime.now().strftime("%H:%M")
    with open(p, "a", encoding="utf-8") as f:
        if not p.exists() or p.stat().st_size == 0:
            f.write(f"# Session {date.today().isoformat()}\n\n")
        f.write(f"- **{now}** {entry}\n")


def get_recent_sessions(chat_id: int, days: int = 3) -> str:
    """Read the last N days of session logs concatenated."""
    sdir = _sessions_dir(chat_id)
    files = sorted(sdir.glob("*.md"), reverse=True)[:days]
    parts = []
    for f in reversed(files):  # chronological order
        parts.append(f.read_text(encoding="utf-8"))
    return "\n---\n".join(parts)


# ── Model usage tracking ──

def record_model_usage(model: str, input_tokens: int, output_tokens: int):
    """Persist token usage for one LLM call."""
    if _use_d1:
        return _d1.record_model_usage(model, input_tokens, output_tokens)
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO model_usage (model, input_tokens, output_tokens) VALUES (?,?,?)",
            (model, input_tokens or 0, output_tokens or 0),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"record_model_usage failed: {e}")


def get_model_usage_summary() -> dict[str, dict]:
    """Return {model: {input_tokens, output_tokens, calls}} for all recorded usage."""
    if _use_d1:
        return _d1.get_model_usage_summary()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT model, SUM(input_tokens), SUM(output_tokens), COUNT(*) "
            "FROM model_usage GROUP BY model ORDER BY model"
        )
        rows = c.fetchall()
        conn.close()
        return {
            row[0]: {"input_tokens": row[1] or 0, "output_tokens": row[2] or 0, "calls": row[3] or 0}
            for row in rows
        }
    except Exception as e:
        logger.error(f"get_model_usage_summary failed: {e}")
        return {}


# ── Full context bundle ──

def get_full_context_md(chat_id: int) -> str:
    """Build a single context string from all MD files for injection into system prompt."""
    sections = []

    profile = get_profile(chat_id)
    if profile:
        sections.append(f"## Owner Profile\n{profile}")

    md_summary = get_md_summary(chat_id)
    if md_summary:
        sections.append(f"## Conversation Summary\n{md_summary}")

    sessions = get_recent_sessions(chat_id, days=3)
    if sessions:
        sections.append(f"## Recent Sessions\n{sessions}")

    return "\n\n".join(sections)



# ── Dynamic API tools ────────────────────────────────────────────────────────

def register_dynamic_tool(
    name: str,
    base_url: str,
    auth_cred: str,
    auth_type: str = "bearer",
    auth_header: str = "Authorization",
    auth_prefix: str = "Bearer ",
    endpoints: list | None = None,
    description: str = "",
    docs_url: str = "",
) -> bool:
    """Register or update a dynamic API tool."""
    if _use_d1:
        return _d1.register_dynamic_tool(name, base_url, auth_cred, auth_type, auth_header, auth_prefix, endpoints, description, docs_url)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT OR REPLACE INTO dynamic_tools
           (name, base_url, auth_type, auth_header, auth_prefix, auth_cred, endpoints, description, docs_url, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
        (
            name.lower().strip(),
            base_url.rstrip("/"),
            auth_type,
            auth_header,
            auth_prefix,
            auth_cred,
            json.dumps(endpoints or []),
            description,
            docs_url,
        ),
    )
    conn.commit()
    conn.close()
    return True


def get_dynamic_tool(name: str) -> dict | None:
    """Get a single dynamic tool by name."""
    if _use_d1:
        return _d1.get_dynamic_tool(name)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT name, base_url, auth_type, auth_header, auth_prefix, auth_cred, endpoints, description, docs_url, enabled "
        "FROM dynamic_tools WHERE name = ?",
        (name.lower().strip(),),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "name": row[0],
        "base_url": row[1],
        "auth_type": row[2],
        "auth_header": row[3],
        "auth_prefix": row[4],
        "auth_cred": row[5],
        "endpoints": json.loads(row[6]) if row[6] else [],
        "description": row[7],
        "docs_url": row[8],
        "enabled": bool(row[9]),
    }


def list_dynamic_tools() -> list[dict]:
    """List all registered dynamic tools."""
    if _use_d1:
        return _d1.list_dynamic_tools()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT name, base_url, auth_cred, description, enabled FROM dynamic_tools ORDER BY name"
    )
    rows = c.fetchall()
    conn.close()
    return [
        {"name": r[0], "base_url": r[1], "auth_cred": r[2], "description": r[3], "enabled": bool(r[4])}
        for r in rows
    ]


def update_dynamic_tool_endpoints(name: str, endpoints: list):
    """Update the endpoints list for a dynamic tool."""
    if _use_d1:
        return _d1.update_dynamic_tool_endpoints(name, endpoints)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE dynamic_tools SET endpoints = ?, updated_at = CURRENT_TIMESTAMP WHERE name = ?",
        (json.dumps(endpoints), name.lower().strip()),
    )
    conn.commit()
    conn.close()


def remove_dynamic_tool(name: str) -> bool:
    """Delete a dynamic tool."""
    if _use_d1:
        return _d1.remove_dynamic_tool(name)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM dynamic_tools WHERE name = ?", (name.lower().strip(),))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted



# ── Skill sources ─────────────────────────────────────────────────────────────
# Track where skills are installed from so they can be auto-reinstalled on fresh
# installs when D1 is configured.

def add_skill_source(name: str, source_type: str, source_uri: str, version: str = "", auto_update: bool = True):
    """Register a skill source.
    source_type: 'clawhub', 'url', or 'github'
    source_uri: full URI like 'clawhub:@owner/name', 'url:https://...', 'github:owner/repo'
    """
    if _use_d1:
        return _d1.add_skill_source(name, source_type, source_uri, version, auto_update)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO skill_sources (name, source_type, source_uri, version, auto_update, updated_at) "
        "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (name, source_type, source_uri, version, 1 if auto_update else 0),
    )
    conn.commit()
    conn.close()


def remove_skill_source(name: str) -> bool:
    """Remove a skill source record."""
    if _use_d1:
        return _d1.remove_skill_source(name)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM skill_sources WHERE name = ?", (name,))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def list_skill_sources() -> list[dict]:
    """List all registered skill sources."""
    if _use_d1:
        return _d1.list_skill_sources()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, source_type, source_uri, version, auto_update, installed_at, updated_at FROM skill_sources ORDER BY name")
    rows = c.fetchall()
    conn.close()
    return [
        {
            "name": r[0],
            "source_type": r[1],
            "source_uri": r[2],
            "version": r[3] or "",
            "auto_update": bool(r[4]),
            "installed_at": r[5] or "",
            "updated_at": r[6] or "",
        }
        for r in rows
    ]


def get_skill_source(name: str) -> dict | None:
    """Get a specific skill source by name."""
    if _use_d1:
        return _d1.get_skill_source(name)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, source_type, source_uri, version, auto_update FROM skill_sources WHERE name = ?", (name,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "name": row[0],
        "source_type": row[1],
        "source_uri": row[2],
        "version": row[3] or "",
        "auto_update": bool(row[4]),
    }



# ── Aliases for api_server compatibility ──────────────────────────────────────

def get_messages(chat_id: int, limit: int = 50) -> list[dict]:
    """Alias for get_history — used by api_server."""
    return get_history(chat_id, limit)


def clear_messages(chat_id: int):
    """Alias for clear_history — used by api_server."""
    return clear_history(chat_id)


def get_usage_stats() -> list[dict]:
    """Return usage stats as a list (used by api_server /api/models/usage)."""
    summary = get_model_usage_summary()
    result = []
    for model, data in summary.items():
        result.append({
            "model": model,
            "input_tokens": data["input_tokens"],
            "output_tokens": data["output_tokens"],
            "calls": data["calls"],
            "total_tokens": data["input_tokens"] + data["output_tokens"],
        })
    return result
