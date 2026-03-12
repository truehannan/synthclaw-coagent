"""
SynthClaw-CoAgent — Persistent Memory Layer
SQLite database + Fernet encryption for credentials.
"""
import sqlite3
from pathlib import Path
from cryptography.fernet import Fernet
from config import DB_PATH, BASE_DIR

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
