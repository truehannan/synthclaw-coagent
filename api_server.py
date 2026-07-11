"""
SynthClaw API Server — REST API for the frontend chat interface.

Provides endpoints for:
- Chat (streaming via SSE)
- Providers & Models
- Memory & Credentials
- Skills management
- System status
- Agent Society
- Session management

Runs alongside Telegram/WhatsApp on a configurable port (default 8000).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
import hashlib
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

import config as cfg
import memory as mem

logger = logging.getLogger(__name__)

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SynthClaw API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────

API_TOKEN = os.getenv("SYNTHCLAW_API_TOKEN", "")
# If no token set, generate one on first run and save it
if not API_TOKEN:
    token_file = cfg.BASE_DIR / ".api_token"
    if token_file.exists():
        API_TOKEN = token_file.read_text().strip()
    else:
        API_TOKEN = uuid.uuid4().hex
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(API_TOKEN)
        token_file.chmod(0o600)
        logger.info(f"Generated API token: {API_TOKEN[:8]}...")


async def verify_token(request: Request):
    """Simple token-based auth. Token passed as Bearer or X-API-Token header."""
    auth = request.headers.get("Authorization", "")
    token = request.headers.get("X-API-Token", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    if not token or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")
    return True



# ── Request/Response Models ───────────────────────────────────────────────────

class ChatMessage(BaseModel):
    message: str
    model: Optional[str] = None

class MemoryItem(BaseModel):
    key: str
    value: str

class CredentialItem(BaseModel):
    name: str
    value: str
    description: Optional[str] = ""

class ProviderKeyItem(BaseModel):
    key: str

class ModelSwitchItem(BaseModel):
    model: str

class SkillInstallItem(BaseModel):
    source: str  # @user/skill or URL

class ConfigUpdateItem(BaseModel):
    key: str
    value: str


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/auth/status")
async def auth_status(request: Request):
    """Check if the provided token is valid."""
    auth = request.headers.get("Authorization", "")
    token = request.headers.get("X-API-Token", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    valid = bool(token and token == API_TOKEN)
    return {"authenticated": valid}


@app.post("/api/auth/login")
async def auth_login(request: Request):
    """Login with password — returns token if password matches hash."""
    body = await request.json()
    password = body.get("password", "")
    # Simple: if password matches stored hash or equals the raw token
    if password == API_TOKEN:
        return {"token": API_TOKEN, "success": True}
    # Check against stored password hash
    stored_hash = mem.get_memory("api_password_hash")
    if stored_hash and hashlib.sha256(password.encode()).hexdigest() == stored_hash:
        return {"token": API_TOKEN, "success": True}
    raise HTTPException(status_code=401, detail="Invalid password")



# ══════════════════════════════════════════════════════════════════════════════
#  CHAT ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# Chat session for web frontend (separate from Telegram chat_id)
WEB_CHAT_ID = 999999  # Fixed chat_id for web interface
_chat_lock = asyncio.Lock()
_pending_approval = {"active": False, "description": "", "resolved": None}


@app.post("/api/chat/send", dependencies=[Depends(verify_token)])
async def chat_send(msg: ChatMessage):
    """Send a message and get streaming response via SSE."""
    from openai import OpenAI

    model = msg.model or mem.get_memory("default_model") or cfg.DEFAULT_MODEL
    user_message = msg.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message")

    # Save user message
    mem.save_message(WEB_CHAT_ID, "user", user_message)

    # Get conversation history
    history = mem.get_messages(WEB_CHAT_ID, limit=cfg.MAX_HISTORY_MESSAGES)

    # Resolve client
    try:
        from agent import _resolve_client_and_model
        client, api_model, provider = _resolve_client_and_model(model)
    except Exception:
        # Fallback to default client
        client = OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_API_BASE)
        api_model = model
        provider = "DigitalOcean"

    # Build messages for LLM
    system_prompt = (
        "You are SynthClaw, a personal AI agent running on the user's server. "
        "Be helpful, concise, and direct. You have access to tools but in this "
        "web interface you respond conversationally."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})

    async def generate():
        """Stream response tokens via SSE."""
        try:
            response = client.chat.completions.create(
                model=api_model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                stream=True,
            )
            full_reply = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_reply += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

            # Save assistant message
            # Strip think tags
            import re
            clean = re.sub(r"<think>[\s\S]*?</think>", "", full_reply).strip()
            mem.save_message(WEB_CHAT_ID, "assistant", clean)
            yield f"data: {json.dumps({'done': True, 'full': clean})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/chat/history", dependencies=[Depends(verify_token)])
async def chat_history(limit: int = 50):
    """Get conversation history for web interface."""
    messages = mem.get_messages(WEB_CHAT_ID, limit=limit)
    return {"messages": messages}


@app.post("/api/chat/clear", dependencies=[Depends(verify_token)])
async def chat_clear():
    """Clear web chat history."""
    mem.clear_messages(WEB_CHAT_ID)
    return {"success": True}


@app.post("/api/chat/stop", dependencies=[Depends(verify_token)])
async def chat_stop():
    """Stop currently running task."""
    try:
        from agent import ACTIVE_TASKS
        ACTIVE_TASKS[WEB_CHAT_ID] = False
    except Exception:
        pass
    return {"success": True}


@app.post("/api/chat/approve", dependencies=[Depends(verify_token)])
async def chat_approve():
    """Approve pending dangerous operation."""
    _pending_approval["resolved"] = "approved"
    return {"success": True}


@app.post("/api/chat/deny", dependencies=[Depends(verify_token)])
async def chat_deny():
    """Deny pending dangerous operation."""
    _pending_approval["resolved"] = "denied"
    return {"success": True}


@app.get("/api/chat/task-status", dependencies=[Depends(verify_token)])
async def chat_task_status():
    """Get current task status."""
    try:
        from agent import TASK_RUNTIME, ACTIVE_TASKS
        rt = TASK_RUNTIME.get(WEB_CHAT_ID, {})
        running = ACTIVE_TASKS.get(WEB_CHAT_ID, False)
        return {"running": running, "runtime": rt, "pending_approval": _pending_approval}
    except Exception:
        return {"running": False, "runtime": {}, "pending_approval": _pending_approval}



# ══════════════════════════════════════════════════════════════════════════════
#  AGENT SOCIETY ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/society/status", dependencies=[Depends(verify_token)])
async def society_status():
    """Get current agent society state."""
    from agents import get_society_status
    return get_society_status()


@app.post("/api/society/reset", dependencies=[Depends(verify_token)])
async def society_reset():
    """Reset all agents."""
    from agents import reset_society
    reset_society()
    return {"success": True}


# ══════════════════════════════════════════════════════════════════════════════
#  PROVIDERS & MODELS ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/providers", dependencies=[Depends(verify_token)])
async def list_providers():
    """List all providers with configuration status."""
    from agent import PROVIDER_META
    providers = []
    for name, meta in PROVIDER_META.items():
        # Check if provider has a stored key
        from agent import _provider_key_name
        key_name = _provider_key_name(name)
        has_key = bool(mem.get_credential(key_name))
        providers.append({
            "name": name,
            "slug": meta["slug"],
            "emoji": meta["emoji"],
            "configured": has_key,
            "key_name": key_name,
        })
    return {"providers": providers}


@app.get("/api/providers/{name}/models", dependencies=[Depends(verify_token)])
async def provider_models(name: str):
    """Fetch models for a specific provider (live from API)."""
    from model_fetcher import fetch_provider_models
    models = fetch_provider_models(name, force=True)
    if not models:
        # Fallback to catalog
        models = list(cfg.MODEL_CATALOG.get(name, []))
    return {"provider": name, "models": models}


@app.post("/api/providers/{name}/key", dependencies=[Depends(verify_token)])
async def store_provider_key(name: str, item: ProviderKeyItem):
    """Store API key for a provider."""
    from agent import _provider_key_name
    key_name = _provider_key_name(name)
    mem.store_credential(key_name, item.key)
    return {"success": True, "key_name": key_name}


@app.delete("/api/providers/{name}/key", dependencies=[Depends(verify_token)])
async def delete_provider_key(name: str):
    """Remove stored API key for a provider."""
    from agent import _provider_key_name
    key_name = _provider_key_name(name)
    # Delete credential
    try:
        conn = mem._get_conn()
        conn.execute("DELETE FROM credentials WHERE name = ?", (key_name,))
        conn.commit()
        conn.close()
    except Exception:
        pass
    return {"success": True}


@app.get("/api/models", dependencies=[Depends(verify_token)])
async def list_all_models():
    """Get all available models across providers."""
    from model_fetcher import get_all_available_models
    models = get_all_available_models()
    return {"models": models}


@app.get("/api/models/current", dependencies=[Depends(verify_token)])
async def current_model():
    """Get current active model."""
    model = mem.get_memory("default_model") or cfg.DEFAULT_MODEL
    return {"model": model}


@app.post("/api/models/switch", dependencies=[Depends(verify_token)])
async def switch_model(item: ModelSwitchItem):
    """Switch active model."""
    mem.set_memory("default_model", item.model)
    return {"success": True, "model": item.model}


@app.get("/api/models/usage", dependencies=[Depends(verify_token)])
async def model_usage():
    """Get token usage statistics per model."""
    try:
        usage = mem.get_usage_stats()
        return {"usage": usage}
    except Exception:
        return {"usage": []}



# ══════════════════════════════════════════════════════════════════════════════
#  MEMORY ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/memory", dependencies=[Depends(verify_token)])
async def get_memory_all():
    """Get all memory facts."""
    facts = mem.get_all_memory()
    return {"facts": facts}


@app.post("/api/memory", dependencies=[Depends(verify_token)])
async def set_memory_item(item: MemoryItem):
    """Set a memory fact."""
    mem.set_memory(item.key, item.value)
    return {"success": True}


@app.delete("/api/memory/{key}", dependencies=[Depends(verify_token)])
async def delete_memory_item(key: str):
    """Delete a memory fact."""
    try:
        conn = mem._get_conn()
        conn.execute("DELETE FROM memory WHERE key = ?", (key,))
        conn.commit()
        conn.close()
    except Exception:
        pass
    return {"success": True}


# ══════════════════════════════════════════════════════════════════════════════
#  CREDENTIALS ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/credentials", dependencies=[Depends(verify_token)])
async def list_credentials():
    """List credentials (names only, values masked)."""
    creds = mem.list_credentials()
    return {"credentials": creds}


@app.post("/api/credentials", dependencies=[Depends(verify_token)])
async def store_credential(item: CredentialItem):
    """Store a credential (encrypted)."""
    mem.store_credential(item.name, item.value, item.description)
    return {"success": True}


@app.delete("/api/credentials/{name}", dependencies=[Depends(verify_token)])
async def delete_credential(name: str):
    """Delete a credential."""
    try:
        conn = mem._get_conn()
        conn.execute("DELETE FROM credentials WHERE name = ?", (name,))
        conn.commit()
        conn.close()
    except Exception:
        pass
    return {"success": True}


# ══════════════════════════════════════════════════════════════════════════════
#  SKILLS ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/skills", dependencies=[Depends(verify_token)])
async def list_skills():
    """List installed skills."""
    try:
        from tools import list_skills_with_sources
        result = list_skills_with_sources({})
        return json.loads(result) if isinstance(result, str) else result
    except Exception:
        return {"skills": [], "sources": []}


@app.post("/api/skills/install", dependencies=[Depends(verify_token)])
async def install_skill(item: SkillInstallItem):
    """Install a skill from source."""
    try:
        from tools import install_skill
        result = install_skill({"source": item.source})
        return json.loads(result) if isinstance(result, str) else {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/skills/{name}", dependencies=[Depends(verify_token)])
async def uninstall_skill(name: str):
    """Uninstall a skill."""
    try:
        from tools import uninstall_skill
        result = uninstall_skill({"name": name})
        return json.loads(result) if isinstance(result, str) else {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/skills/reinstall", dependencies=[Depends(verify_token)])
async def reinstall_skills():
    """Reinstall all skills from sources."""
    try:
        from tools import reinstall_all_skills
        result = reinstall_all_skills({})
        return json.loads(result) if isinstance(result, str) else {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/system/status", dependencies=[Depends(verify_token)])
async def system_status():
    """System information — CPU, memory, uptime, disk."""
    import platform
    import shutil

    info = {
        "hostname": platform.node(),
        "platform": platform.system(),
        "python": platform.python_version(),
        "uptime": 0,
        "cpu_percent": 0,
        "memory": {"total": 0, "used": 0, "percent": 0},
        "disk": {"total": 0, "used": 0, "percent": 0},
    }

    try:
        import psutil
        info["cpu_percent"] = psutil.cpu_percent(interval=0.5)
        vm = psutil.virtual_memory()
        info["memory"] = {"total": vm.total, "used": vm.used, "percent": vm.percent}
        du = shutil.disk_usage("/")
        info["disk"] = {"total": du.total, "used": du.used, "percent": round(du.used / du.total * 100, 1)}
        info["uptime"] = int(time.time() - psutil.boot_time())
    except ImportError:
        # psutil not available — basic fallback
        try:
            import os
            info["uptime"] = int(os.popen("cat /proc/uptime").read().split()[0])
        except Exception:
            pass
        try:
            du = shutil.disk_usage("/")
            info["disk"] = {"total": du.total, "used": du.used, "percent": round(du.used / du.total * 100, 1)}
        except Exception:
            pass

    return info


@app.get("/api/system/config", dependencies=[Depends(verify_token)])
async def system_config():
    """Get current non-secret configuration."""
    return {
        "interface_mode": cfg.INTERFACE_MODE,
        "storage_mode": cfg.STORAGE_MODE,
        "default_model": cfg.DEFAULT_MODEL,
        "max_tool_iterations": cfg.MAX_TOOL_ITERATIONS,
        "max_history_messages": cfg.MAX_HISTORY_MESSAGES,
        "checkpoint_every": cfg.CHECKPOINT_EVERY,
        "max_rpm": cfg.MAX_RPM,
        "base_dir": str(cfg.BASE_DIR),
        "has_composio": bool(cfg.COMPOSIO_API_KEY),
        "has_d1": bool(cfg.CF_D1_DATABASE_ID and cfg.CF_API_TOKEN),
    }


@app.post("/api/system/config", dependencies=[Depends(verify_token)])
async def update_system_config(item: ConfigUpdateItem):
    """Update a configuration value."""
    allowed_keys = {"default_model", "max_tool_iterations", "max_history_messages", "max_rpm"}
    if item.key not in allowed_keys:
        raise HTTPException(status_code=400, detail=f"Cannot update key: {item.key}")
    mem.set_memory(f"config_{item.key}", item.value)
    return {"success": True}


@app.get("/api/system/health")
async def health_check():
    """Health check — no auth needed."""
    return {"status": "ok", "timestamp": time.time(), "version": "1.0.0"}


@app.get("/api/system/logs", dependencies=[Depends(verify_token)])
async def system_logs(lines: int = 50):
    """Get recent log lines."""
    try:
        log_path = cfg.LOG_PATH
        if log_path.exists():
            with open(log_path, "r") as f:
                all_lines = f.readlines()
                recent = all_lines[-lines:]
                return {"logs": [l.rstrip() for l in recent]}
    except Exception:
        pass
    return {"logs": []}



# ══════════════════════════════════════════════════════════════════════════════
#  SESSIONS ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# Sessions are stored as memory entries with prefix "session_"
# Each session has: id, name, created_at, last_active
_sessions_cache: list[dict] = []


@app.get("/api/sessions", dependencies=[Depends(verify_token)])
async def list_sessions():
    """List all chat sessions."""
    sessions_raw = mem.get_memory("web_sessions")
    if sessions_raw:
        try:
            sessions = json.loads(sessions_raw)
        except Exception:
            sessions = []
    else:
        # Create default session
        sessions = [{
            "id": "default",
            "name": "Main",
            "created_at": time.time(),
            "last_active": time.time(),
            "chat_id": WEB_CHAT_ID,
        }]
        mem.set_memory("web_sessions", json.dumps(sessions))
    return {"sessions": sessions, "active": mem.get_memory("active_session") or "default"}


@app.post("/api/sessions", dependencies=[Depends(verify_token)])
async def create_session(request: Request):
    """Create a new session."""
    body = await request.json()
    name = body.get("name", f"Session {int(time.time())}")

    sessions_raw = mem.get_memory("web_sessions")
    sessions = json.loads(sessions_raw) if sessions_raw else []

    new_id = uuid.uuid4().hex[:8]
    new_chat_id = WEB_CHAT_ID + len(sessions) + 1
    sessions.append({
        "id": new_id,
        "name": name,
        "created_at": time.time(),
        "last_active": time.time(),
        "chat_id": new_chat_id,
    })
    mem.set_memory("web_sessions", json.dumps(sessions))
    return {"session": sessions[-1]}


@app.delete("/api/sessions/{session_id}", dependencies=[Depends(verify_token)])
async def delete_session(session_id: str):
    """Delete a session."""
    sessions_raw = mem.get_memory("web_sessions")
    sessions = json.loads(sessions_raw) if sessions_raw else []

    # Find and remove
    session = next((s for s in sessions if s["id"] == session_id), None)
    if session:
        # Clear messages for this session
        try:
            mem.clear_messages(session.get("chat_id", 0))
        except Exception:
            pass
        sessions = [s for s in sessions if s["id"] != session_id]
        mem.set_memory("web_sessions", json.dumps(sessions))

    return {"success": True}


@app.post("/api/sessions/{session_id}/switch", dependencies=[Depends(verify_token)])
async def switch_session(session_id: str):
    """Switch active session."""
    mem.set_memory("active_session", session_id)
    return {"success": True, "active": session_id}


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATIONS ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/apis", dependencies=[Depends(verify_token)])
async def list_apis():
    """List registered dynamic APIs."""
    try:
        from tools import list_apis
        result = list_apis({})
        return json.loads(result) if isinstance(result, str) else {"apis": []}
    except Exception:
        return {"apis": []}


@app.get("/api/composio/connections", dependencies=[Depends(verify_token)])
async def composio_connections():
    """List Composio connections."""
    if not cfg.COMPOSIO_API_KEY:
        return {"connections": [], "available": False}
    try:
        from tools import composio_list_connections
        result = composio_list_connections({})
        return json.loads(result) if isinstance(result, str) else {"connections": []}
    except Exception:
        return {"connections": [], "available": True}


# ══════════════════════════════════════════════════════════════════════════════
#  SERVER STARTUP
# ══════════════════════════════════════════════════════════════════════════════

def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server (called from main.py)."""
    import uvicorn
    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"API Token: {API_TOKEN[:8]}... (full token in {cfg.BASE_DIR}/.api_token)")
    uvicorn.run(app, host=host, port=port, log_level="info")


def start_api_server_background(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server in a background thread."""
    import threading
    import uvicorn

    config_uv = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config_uv)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    logger.info(f"API server started on {host}:{port} (background)")
    logger.info(f"API Token: {API_TOKEN[:8]}...")
    return server


if __name__ == "__main__":
    mem.init_db()
    start_api_server()
