"""
SynthClaw-CoAgent — Persistent Memory Layer
SQLite database + Fernet encryption for credentials + markdown file context.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, date
from cryptography.fernet import Fernet
from config import DB_PATH, BASE_DIR

CONTEXT_DIR = BASE_DIR / "context"

KEY_FILE = BASE_DIR / ".fernet_key"


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
    """)
    conn.commit()
    conn.close()


# ── Conversation history ────────────────────────────────────────────────────

def get_history(chat_id: int, limit: int = 20) -> list[dict]:
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?,?,?)",
        (chat_id, role, content),
    )
    conn.commit()
    conn.close()


def clear_history(chat_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    conn.commit()
    conn.close()


# ── Credentials (encrypted) ─────────────────────────────────────────────────

def store_credential(name: str, value: str, description: str = ""):
    f = get_fernet()
    enc = f.encrypt(value.encode())
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT enc_value FROM credentials WHERE name=?", (name,))
    row = c.fetchone()
    conn.close()
    return f.decrypt(row[0]).decode() if row else None


def list_credentials() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, description, created_at FROM credentials")
    rows = c.fetchall()
    conn.close()
    return [{"name": r[0], "description": r[1], "created_at": r[2]} for r in rows]


# ── Memory (key-value) ───────────────────────────────────────────────────────

def set_memory(key: str, value: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO memory (key, value) VALUES (?,?)", (key, value)
    )
    conn.commit()
    conn.close()


def get_memory(key: str) -> str | None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value FROM memory WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def get_all_memory() -> dict:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT key, value FROM memory")
    rows = c.fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


# ── App config ───────────────────────────────────────────────────────────────

def get_config(key: str, default=None) -> str | None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value FROM config WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else default


def set_config(key: str, value: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?,?)", (key, value))
    conn.commit()
    conn.close()


# ── Conversation summaries ───────────────────────────────────────────────────

def count_messages(chat_id: int) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM messages WHERE chat_id=?", (chat_id,))
    count = c.fetchone()[0]
    conn.close()
    return count


def get_oldest_messages(chat_id: int, limit: int) -> list[dict]:
    """Get the oldest N messages for a chat (for summarization)."""
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    placeholders = ",".join("?" for _ in msg_ids)
    c.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", msg_ids)
    conn.commit()
    conn.close()


def save_summary(chat_id: int, summary: str, msg_count: int):
    """Save a conversation summary (cumulative — replaces previous)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM conversation_summaries WHERE chat_id=?", (chat_id,))
    c.execute(
        "INSERT INTO conversation_summaries (chat_id, summary, msg_count) VALUES (?,?,?)",
        (chat_id, summary, msg_count),
    )
    conn.commit()
    conn.close()


def get_summary(chat_id: int) -> str | None:
    """Get the latest conversation summary for a chat."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT summary FROM conversation_summaries WHERE chat_id=? ORDER BY created_at DESC LIMIT 1",
        (chat_id,),
    )
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


# ── Markdown file context system ─────────────────────────────────────────────

def _chat_dir(chat_id: int) -> Path:
    d = CONTEXT_DIR / str(chat_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sessions_dir(chat_id: int) -> Path:
    d = _chat_dir(chat_id) / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


# -- profile.md: persistent facts about the user --

def get_profile(chat_id: int) -> str:
    p = _chat_dir(chat_id) / "profile.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def save_profile(chat_id: int, content: str):
    p = _chat_dir(chat_id) / "profile.md"
    p.write_text(content, encoding="utf-8")


def append_to_profile(chat_id: int, line: str):
    """Add a line if not already present."""
    profile = get_profile(chat_id)
    if line.strip() and line.strip() not in profile:
        new = profile.rstrip() + "\n" + line.strip() + "\n" if profile else line.strip() + "\n"
        save_profile(chat_id, new)


# -- summary.md: rolling conversation summary --

def get_md_summary(chat_id: int) -> str:
    p = _chat_dir(chat_id) / "summary.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def save_md_summary(chat_id: int, content: str):
    p = _chat_dir(chat_id) / "summary.md"
    p.write_text(content, encoding="utf-8")


# -- sessions/YYYY-MM-DD.md: daily session logs --

def get_today_session(chat_id: int) -> str:
    p = _sessions_dir(chat_id) / f"{date.today().isoformat()}.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def append_to_session(chat_id: int, line: str):
    p = _sessions_dir(chat_id) / f"{date.today().isoformat()}.md"
    ts = datetime.now().strftime("%H:%M")
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {line}\n")


def get_recent_sessions(chat_id: int, days: int = 3) -> str:
    sdir = _sessions_dir(chat_id)
    parts = []
    for i in range(days):
        from datetime import timedelta
        d = date.today() - timedelta(days=i)
        p = sdir / f"{d.isoformat()}.md"
        if p.exists():
            parts.append(f"### {d.isoformat()}\n{p.read_text(encoding='utf-8')}")
    return "\n".join(reversed(parts))


def get_full_context_md(chat_id: int) -> str:
    """Bundle profile + summary + recent sessions for system prompt injection."""
    sections = []
    profile = get_profile(chat_id)
    if profile:
        sections.append(f"## User Profile\n{profile}")
    summary = get_md_summary(chat_id)
    if summary:
        sections.append(f"## Conversation Summary\n{summary}")
    sessions = get_recent_sessions(chat_id, 3)
    if sessions:
        sections.append(f"## Recent Sessions\n{sessions}")
    return "\n\n".join(sections)
