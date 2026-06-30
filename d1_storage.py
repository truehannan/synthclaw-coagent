"""
Cloudflare D1 storage backend — replaces local SQLite when configured.

Uses Cloudflare D1 HTTP API:
  POST /client/v4/accounts/{account_id}/d1/database/{database_id}/query

All functions mirror memory.py's interface so they can be swapped transparently.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# ── Connection state ─────────────────────────────────────────────────────────

_d1_config: dict = {}
_initialized: bool = False


def configure(account_id: str, database_id: str, api_token: str):
    """Set D1 connection parameters. Called once at startup."""
    global _d1_config, _initialized
    _d1_config = {
        "account_id": account_id,
        "database_id": database_id,
        "api_token": api_token,
        "base_url": f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}",
    }
    _initialized = False


def is_configured() -> bool:
    """Return True if D1 credentials are set."""
    return bool(
        _d1_config.get("account_id")
        and _d1_config.get("database_id")
        and _d1_config.get("api_token")
    )


# ── Low-level D1 HTTP API ─────────────────────────────────────────────────────

def _query(sql: str, params: list | None = None) -> list[dict]:
    """Execute a SQL query against D1 and return rows as list of dicts.
    For INSERT/UPDATE/DELETE, returns [{"changes": N, "last_row_id": N}].
    """
    if not is_configured():
        raise RuntimeError("D1 not configured")

    url = f"{_d1_config['base_url']}/query"
    headers = {
        "Authorization": f"Bearer {_d1_config['api_token']}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {"sql": sql}
    if params:
        payload["params"] = params

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        data = resp.json()

        if not data.get("success"):
            errors = data.get("errors", [])
            err_msg = errors[0].get("message", "Unknown D1 error") if errors else "Unknown D1 error"
            raise RuntimeError(f"D1 query failed: {err_msg}")

        # D1 returns results as array of result objects
        results = data.get("result", [])
        if not results:
            return []

        # Each result has "results" (rows) and "meta"
        first_result = results[0] if isinstance(results, list) else results
        rows = first_result.get("results", [])
        meta = first_result.get("meta", {})

        # For DML statements, return meta info
        if meta.get("changes", 0) > 0 or not rows:
            if not rows:
                return [{"changes": meta.get("changes", 0), "last_row_id": meta.get("last_row_id", 0)}]

        return rows

    except requests.RequestException as e:
        logger.error(f"D1 HTTP error: {e}")
        raise RuntimeError(f"D1 connection failed: {e}")


def _batch_query(statements: list[dict]) -> list[list[dict]]:
    """Execute multiple SQL statements in a batch.
    Each statement: {"sql": "...", "params": [...]}
    """
    if not is_configured():
        raise RuntimeError("D1 not configured")

    url = f"{_d1_config['base_url']}/query"
    headers = {
        "Authorization": f"Bearer {_d1_config['api_token']}",
        "Content-Type": "application/json",
    }

    # D1 batch: send array of statements
    # Actually D1 raw endpoint doesn't support batch natively via /query
    # We use /raw for batch or just loop
    results = []
    for stmt in statements:
        try:
            rows = _query(stmt["sql"], stmt.get("params"))
            results.append(rows)
        except Exception as e:
            logger.error(f"Batch statement failed: {stmt['sql'][:80]} — {e}")
            results.append([])
    return results


# ── Schema initialization ─────────────────────────────────────────────────────

D1_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER,
    role TEXT,
    content TEXT,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS credentials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    enc_value TEXT,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER,
    summary TEXT,
    msg_count INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS task_state (
    chat_id INTEGER PRIMARY KEY,
    messages TEXT NOT NULL,
    media TEXT NOT NULL,
    model TEXT NOT NULL,
    step_count INTEGER NOT NULL DEFAULT 0,
    attempt_step INTEGER NOT NULL DEFAULT 0,
    stall_count INTEGER NOT NULL DEFAULT 0,
    last_sig TEXT,
    last_error TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS long_term_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    fact TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 1,
    source TEXT DEFAULT 'auto',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chat_id, fact)
);

CREATE TABLE IF NOT EXISTS dynamic_tools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    base_url TEXT NOT NULL,
    auth_type TEXT NOT NULL DEFAULT 'bearer',
    auth_header TEXT DEFAULT 'Authorization',
    auth_prefix TEXT DEFAULT 'Bearer ',
    auth_cred TEXT NOT NULL,
    endpoints TEXT NOT NULL DEFAULT '[]',
    description TEXT DEFAULT '',
    docs_url TEXT DEFAULT '',
    enabled INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS skill_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    source_type TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    version TEXT DEFAULT '',
    auto_update INTEGER DEFAULT 1,
    installed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, ts);
CREATE INDEX IF NOT EXISTS idx_facts_chat ON long_term_facts(chat_id, importance);
CREATE INDEX IF NOT EXISTS idx_model_usage_model ON model_usage(model);
"""


def init_d1():
    """Initialize D1 database schema. Called once at startup."""
    global _initialized
    if _initialized:
        return True
    if not is_configured():
        return False

    # Execute each CREATE statement separately (D1 doesn't support multi-statement)
    statements = [s.strip() for s in D1_SCHEMA.split(";") if s.strip()]
    errors = []
    for stmt in statements:
        try:
            _query(stmt)
        except Exception as e:
            # Ignore "already exists" type errors
            err_str = str(e)
            if "already exists" not in err_str.lower():
                errors.append(f"{stmt[:60]}... → {e}")

    if errors:
        logger.warning(f"D1 schema init had {len(errors)} issues: {errors[:3]}")
    else:
        logger.info("D1 schema initialized successfully")

    _initialized = True
    return True


# ── Conversation history ──────────────────────────────────────────────────────

def get_history(chat_id: int, limit: int = 20) -> list[dict]:
    rows = _query(
        "SELECT role, content FROM messages WHERE chat_id = ?1 ORDER BY ts DESC LIMIT ?2",
        [chat_id, limit],
    )
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def save_message(chat_id: int, role: str, content: str):
    _query(
        "INSERT INTO messages (chat_id, role, content) VALUES (?1, ?2, ?3)",
        [chat_id, role, content],
    )


def clear_history(chat_id: int):
    _query("DELETE FROM messages WHERE chat_id = ?1", [chat_id])


def count_messages(chat_id: int) -> int:
    rows = _query("SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ?1", [chat_id])
    return rows[0]["cnt"] if rows else 0


def get_oldest_messages(chat_id: int, limit: int) -> list[dict]:
    rows = _query(
        "SELECT id, role, content FROM messages WHERE chat_id = ?1 ORDER BY ts ASC LIMIT ?2",
        [chat_id, limit],
    )
    return [{"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]


def delete_messages_by_ids(msg_ids: list[int]):
    if not msg_ids:
        return
    placeholders = ",".join(f"?{i+1}" for i in range(len(msg_ids)))
    _query(f"DELETE FROM messages WHERE id IN ({placeholders})", msg_ids)


# ── Credentials (stored encrypted — encryption happens in memory.py) ─────────

def store_credential(name: str, enc_value: str, description: str = ""):
    """Store encrypted credential value (already encrypted by caller)."""
    _query(
        "INSERT OR REPLACE INTO credentials (name, enc_value, description) VALUES (?1, ?2, ?3)",
        [name, enc_value, description],
    )


def get_credential(name: str) -> str | None:
    """Get encrypted credential value (caller decrypts)."""
    rows = _query("SELECT enc_value FROM credentials WHERE name = ?1", [name])
    return rows[0]["enc_value"] if rows else None


def list_credentials() -> list[dict]:
    rows = _query("SELECT name, description, created_at FROM credentials")
    return [{"name": r["name"], "description": r["description"], "created_at": r["created_at"]} for r in rows]


# ── Memory (key-value) ────────────────────────────────────────────────────────

def set_memory(key: str, value: str):
    _query("INSERT OR REPLACE INTO memory (key, value) VALUES (?1, ?2)", [key, value])


def get_memory(key: str) -> str | None:
    rows = _query("SELECT value FROM memory WHERE key = ?1", [key])
    return rows[0]["value"] if rows else None


def get_all_memory() -> dict:
    rows = _query("SELECT key, value FROM memory")
    return {r["key"]: r["value"] for r in rows}


# ── App config ────────────────────────────────────────────────────────────────

def get_config(key: str, default=None) -> str | None:
    rows = _query("SELECT value FROM config WHERE key = ?1", [key])
    return rows[0]["value"] if rows else default


def set_config(key: str, value: str):
    _query("INSERT OR REPLACE INTO config (key, value) VALUES (?1, ?2)", [key, value])


# ── Task state ────────────────────────────────────────────────────────────────

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
    _query(
        "INSERT OR REPLACE INTO task_state (chat_id, messages, media, model, step_count, attempt_step, stall_count, last_sig, last_error) "
        "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        [chat_id, json.dumps(messages), json.dumps(media), model, step_count, attempt_step, stall_count, last_sig, last_error],
    )


def load_task_state(chat_id: int) -> dict | None:
    rows = _query(
        "SELECT messages, media, model, step_count, attempt_step, stall_count, last_sig, last_error FROM task_state WHERE chat_id = ?1",
        [chat_id],
    )
    if not rows:
        return None
    r = rows[0]
    return {
        "messages": json.loads(r["messages"]),
        "media": json.loads(r["media"]),
        "model": r["model"],
        "step_count": r["step_count"],
        "attempt_step": r.get("attempt_step", r["step_count"]),
        "stall_count": r.get("stall_count", 0),
        "last_sig": r.get("last_sig"),
        "last_error": r.get("last_error"),
    }


def clear_task_state(chat_id: int):
    _query("DELETE FROM task_state WHERE chat_id = ?1", [chat_id])


# ── Conversation summaries ────────────────────────────────────────────────────

def save_summary(chat_id: int, summary: str, msg_count: int):
    _query("DELETE FROM conversation_summaries WHERE chat_id = ?1", [chat_id])
    _query(
        "INSERT INTO conversation_summaries (chat_id, summary, msg_count) VALUES (?1, ?2, ?3)",
        [chat_id, summary, msg_count],
    )


def get_summary(chat_id: int) -> str | None:
    rows = _query(
        "SELECT summary FROM conversation_summaries WHERE chat_id = ?1 ORDER BY created_at DESC LIMIT 1",
        [chat_id],
    )
    return rows[0]["summary"] if rows else None


# ── Long-term facts ───────────────────────────────────────────────────────────

def add_long_term_fact(chat_id: int, fact: str, importance: int = 1, source: str = "auto"):
    clean = (fact or "").strip()
    if not clean:
        return
    imp = max(1, min(int(importance or 1), 5))
    _query(
        "INSERT OR IGNORE INTO long_term_facts (chat_id, fact, importance, source) VALUES (?1, ?2, ?3, ?4)",
        [chat_id, clean, imp, source],
    )
    _query(
        "UPDATE long_term_facts SET importance = MAX(importance, ?1) WHERE chat_id = ?2 AND fact = ?3",
        [imp, chat_id, clean],
    )


def get_long_term_facts(chat_id: int, limit: int = 40) -> list[dict]:
    rows = _query(
        "SELECT fact, importance, source, created_at FROM long_term_facts "
        "WHERE chat_id = ?1 ORDER BY importance DESC, created_at DESC LIMIT ?2",
        [chat_id, limit],
    )
    return [{"fact": r["fact"], "importance": r["importance"], "source": r["source"], "created_at": r["created_at"]} for r in rows]


# ── Model usage tracking ──────────────────────────────────────────────────────

def record_model_usage(model: str, input_tokens: int, output_tokens: int):
    try:
        _query(
            "INSERT INTO model_usage (model, input_tokens, output_tokens) VALUES (?1, ?2, ?3)",
            [model, input_tokens or 0, output_tokens or 0],
        )
    except Exception as e:
        logger.error(f"D1 record_model_usage failed: {e}")


def get_model_usage_summary() -> dict[str, dict]:
    try:
        rows = _query(
            "SELECT model, SUM(input_tokens) as sum_in, SUM(output_tokens) as sum_out, COUNT(*) as cnt "
            "FROM model_usage GROUP BY model ORDER BY model"
        )
        return {
            r["model"]: {"input_tokens": r["sum_in"] or 0, "output_tokens": r["sum_out"] or 0, "calls": r["cnt"] or 0}
            for r in rows
        }
    except Exception as e:
        logger.error(f"D1 get_model_usage_summary failed: {e}")
        return {}


# ── Dynamic tools ─────────────────────────────────────────────────────────────

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
    _query(
        "INSERT OR REPLACE INTO dynamic_tools "
        "(name, base_url, auth_type, auth_header, auth_prefix, auth_cred, endpoints, description, docs_url, updated_at) "
        "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, CURRENT_TIMESTAMP)",
        [
            name.lower().strip(),
            base_url.rstrip("/"),
            auth_type,
            auth_header,
            auth_prefix,
            auth_cred,
            json.dumps(endpoints or []),
            description,
            docs_url,
        ],
    )
    return True


def get_dynamic_tool(name: str) -> dict | None:
    rows = _query(
        "SELECT name, base_url, auth_type, auth_header, auth_prefix, auth_cred, endpoints, description, docs_url, enabled "
        "FROM dynamic_tools WHERE name = ?1",
        [name.lower().strip()],
    )
    if not rows:
        return None
    r = rows[0]
    return {
        "name": r["name"],
        "base_url": r["base_url"],
        "auth_type": r["auth_type"],
        "auth_header": r["auth_header"],
        "auth_prefix": r["auth_prefix"],
        "auth_cred": r["auth_cred"],
        "endpoints": json.loads(r["endpoints"]) if r["endpoints"] else [],
        "description": r["description"],
        "docs_url": r["docs_url"],
        "enabled": bool(r["enabled"]),
    }


def list_dynamic_tools() -> list[dict]:
    rows = _query("SELECT name, base_url, auth_cred, description, enabled FROM dynamic_tools ORDER BY name")
    return [
        {"name": r["name"], "base_url": r["base_url"], "auth_cred": r["auth_cred"], "description": r["description"], "enabled": bool(r["enabled"])}
        for r in rows
    ]


def update_dynamic_tool_endpoints(name: str, endpoints: list):
    _query(
        "UPDATE dynamic_tools SET endpoints = ?1, updated_at = CURRENT_TIMESTAMP WHERE name = ?2",
        [json.dumps(endpoints), name.lower().strip()],
    )


def remove_dynamic_tool(name: str) -> bool:
    result = _query("DELETE FROM dynamic_tools WHERE name = ?1", [name.lower().strip()])
    return bool(result and result[0].get("changes", 0) > 0)


# ── Skill sources ─────────────────────────────────────────────────────────────

def add_skill_source(name: str, source_type: str, source_uri: str, version: str = "", auto_update: bool = True):
    """Register a skill source in D1.
    source_type: 'clawhub' or 'url'
    source_uri: 'clawhub:@owner/name' or 'url:https://...'
    """
    _query(
        "INSERT OR REPLACE INTO skill_sources (name, source_type, source_uri, version, auto_update, updated_at) "
        "VALUES (?1, ?2, ?3, ?4, ?5, CURRENT_TIMESTAMP)",
        [name, source_type, source_uri, version, 1 if auto_update else 0],
    )


def remove_skill_source(name: str) -> bool:
    result = _query("DELETE FROM skill_sources WHERE name = ?1", [name])
    return bool(result and result[0].get("changes", 0) > 0)


def list_skill_sources() -> list[dict]:
    rows = _query("SELECT name, source_type, source_uri, version, auto_update, installed_at, updated_at FROM skill_sources ORDER BY name")
    return [
        {
            "name": r["name"],
            "source_type": r["source_type"],
            "source_uri": r["source_uri"],
            "version": r.get("version", ""),
            "auto_update": bool(r.get("auto_update", 1)),
            "installed_at": r.get("installed_at", ""),
            "updated_at": r.get("updated_at", ""),
        }
        for r in rows
    ]


def get_skill_source(name: str) -> dict | None:
    rows = _query("SELECT name, source_type, source_uri, version, auto_update FROM skill_sources WHERE name = ?1", [name])
    if not rows:
        return None
    r = rows[0]
    return {
        "name": r["name"],
        "source_type": r["source_type"],
        "source_uri": r["source_uri"],
        "version": r.get("version", ""),
        "auto_update": bool(r.get("auto_update", 1)),
    }


# ── Bulk config sync (for wizard DB-first flow) ──────────────────────────────

def get_all_config() -> dict:
    """Get all config key-value pairs from D1."""
    rows = _query("SELECT key, value FROM config")
    return {r["key"]: r["value"] for r in rows}


def set_all_config(config_dict: dict):
    """Bulk set config values in D1."""
    for key, value in config_dict.items():
        _query("INSERT OR REPLACE INTO config (key, value) VALUES (?1, ?2)", [key, str(value)])


def sync_config_to_d1(local_config: dict):
    """Push local config to D1 (used during setup)."""
    set_all_config(local_config)


def sync_config_from_d1() -> dict:
    """Pull all config from D1 (used during wizard DB-first flow)."""
    return get_all_config()
