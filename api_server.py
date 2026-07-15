"""
Conclave API Server — REST API for the frontend chat interface.

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
    title="Conclave API",
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

# Initialize database tables on module load — prevents 500 on first request
mem.init_db()

# ── Auth ──────────────────────────────────────────────────────────────────────

API_TOKEN = os.getenv("CONCLAVE_API_TOKEN", "")
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


# ── Provider metadata (standalone, no heavy agent.py import) ──────────────────

PROVIDER_META = {
    "Qwen": {"slug": "qw", "emoji": "🟠"},
    "DigitalOcean": {"slug": "do", "emoji": "🌊"},
    "Anthropic": {"slug": "an", "emoji": "🟣"},
    "OpenAI": {"slug": "oa", "emoji": "🟢"},
    "OpenRouter": {"slug": "or", "emoji": "🧭"},
    "GitHub": {"slug": "gh", "emoji": "🐙"},
    "NVIDIA": {"slug": "nv", "emoji": "🟩"},
    "HuggingFace": {"slug": "hf", "emoji": "🤗"},
    "Google": {"slug": "gg", "emoji": "🔵"},
    "Cloudflare": {"slug": "cf", "emoji": "🔶"},
}


def _provider_key_name(provider: str) -> str:
    """Map provider name to credential store key name."""
    _map = {
        "OpenRouter": "OPENROUTER_API_KEY",
        "GitHub": "GITHUB_MODELS_API_KEY",
        "OpenAI": "OPENAI_PROVIDER_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "NVIDIA": "NVIDIA_API_KEY",
        "HuggingFace": "HUGGINGFACE_API_KEY",
        "Google": "GOOGLE_AI_API_KEY",
        "Qwen": "QWEN_API_KEY",
        "Cloudflare": "CLOUDFLARE_API_KEY",
    }
    return _map.get(provider, "OPENAI_API_KEY")


def _provider_base_url(provider: str) -> str:
    """Map provider name to its API base URL."""
    _map = {
        "DigitalOcean": "https://inference.do-ai.run/v1",
        "OpenAI": "https://api.openai.com/v1",
        "Anthropic": "https://inference.do-ai.run/v1",
        "OpenRouter": cfg.OPENROUTER_API_BASE,
        "GitHub": cfg.GITHUB_MODELS_API_BASE,
        "NVIDIA": cfg.NVIDIA_API_BASE,
        "HuggingFace": cfg.HUGGINGFACE_API_BASE,
        "Google": cfg.GOOGLE_AI_API_BASE,
        "Qwen": cfg.QWEN_API_BASE,
        "Cloudflare": "",  # needs account_id, handled separately
    }
    return _map.get(provider, cfg.OPENAI_API_BASE or "https://inference.do-ai.run/v1")


def _resolve_model_routing(model: str) -> tuple:
    """Given a model string (possibly prefixed), return (api_model_name, base_url, provider_name).

    Handles prefixes like 'qwen:', 'google:', 'nvidia:', 'hf:', 'openrouter:', 'github:', 'cloudflare:'.
    Unprefixed models go to DigitalOcean/default.
    """
    _prefix_map = {
        "qwen:": ("Qwen", cfg.QWEN_API_BASE),
        "google:": ("Google", cfg.GOOGLE_AI_API_BASE),
        "nvidia:": ("NVIDIA", cfg.NVIDIA_API_BASE),
        "hf:": ("HuggingFace", cfg.HUGGINGFACE_API_BASE),
        "openrouter:": ("OpenRouter", cfg.OPENROUTER_API_BASE),
        "github:": ("GitHub", cfg.GITHUB_MODELS_API_BASE),
        "cloudflare:": ("Cloudflare", "__cloudflare__"),
    }
    for prefix, (provider, base_url) in _prefix_map.items():
        if model.startswith(prefix):
            stripped = model[len(prefix):]
            # Cloudflare needs account_id to build URL
            if base_url == "__cloudflare__":
                account_id = mem.get_memory("cf_account_id") or cfg.CF_ACCOUNT_ID or ""
                base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1" if account_id else ""
            return (stripped, base_url, provider)

    # No prefix — use default (DigitalOcean or env base)
    return (model, cfg.OPENAI_API_BASE or "https://inference.do-ai.run/v1", "DigitalOcean")


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
    valid = bool(token and (token == API_TOKEN or (token == cfg.OPENAI_API_KEY and cfg.OPENAI_API_KEY)))
    return {"authenticated": valid}


@app.get("/api/auth/exists")
async def auth_exists():
    """Check if a user has been set up (controls signup vs login on frontend)."""
    has_password = bool(mem.get_memory("user_password_hash"))
    return {"exists": has_password}


@app.post("/api/auth/signup")
async def auth_signup(request: Request):
    """First-time user registration. Only works if no password has been set."""
    if mem.get_memory("user_password_hash"):
        raise HTTPException(status_code=409, detail="User already exists. Use login.")
    body = await request.json()
    password = body.get("password", "")
    if not password or len(password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    mem.set_memory("user_password_hash", pw_hash)
    return {"token": API_TOKEN, "success": True}


@app.post("/api/auth/login")
async def auth_login(request: Request):
    """Login with token, API key, or password."""
    body = await request.json()
    password = body.get("password", "")
    if password == API_TOKEN:
        return {"token": API_TOKEN, "success": True}
    if password == cfg.OPENAI_API_KEY and cfg.OPENAI_API_KEY:
        return {"token": password, "success": True}
    stored_hash = mem.get_memory("user_password_hash")
    if stored_hash and hashlib.sha256(password.encode()).hexdigest() == stored_hash:
        return {"token": API_TOKEN, "success": True}
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/api/auth/change-password", dependencies=[Depends(verify_token)])
async def auth_change_password(request: Request):
    """Change user password. Requires current password."""
    body = await request.json()
    current = body.get("current_password", "")
    new_pw = body.get("new_password", "")
    if not new_pw or len(new_pw) < 4:
        raise HTTPException(status_code=400, detail="New password must be at least 4 characters")
    # Verify current password
    stored_hash = mem.get_memory("user_password_hash")
    if stored_hash:
        if hashlib.sha256(current.encode()).hexdigest() != stored_hash:
            # Also accept API token as current password
            if current != API_TOKEN:
                raise HTTPException(status_code=401, detail="Current password is incorrect")
    else:
        # No password set — accept API token
        if current != API_TOKEN:
            raise HTTPException(status_code=401, detail="Current password is incorrect")
    # Set new password
    new_hash = hashlib.sha256(new_pw.encode()).hexdigest()
    mem.set_memory("user_password_hash", new_hash)
    return {"success": True}


@app.get("/api/setup/status")
async def setup_status(request: Request):
    """Return detailed setup status — what's configured vs missing.
    No auth required — called during signup/setup flow before token is available."""
    has_provider = False
    provider_name = ""

    # Check env-level API key first
    if cfg.OPENAI_API_KEY:
        has_provider = True
        provider_name = "DigitalOcean"
    else:
        # Check stored provider keys
        try:
            for name in PROVIDER_META:
                key_name = _provider_key_name(name)
                if mem.get_credential(key_name):
                    has_provider = True
                    provider_name = name
                    break
        except Exception:
            pass

    has_model = bool(mem.get_memory("default_model") or cfg.DEFAULT_MODEL)
    default_model = mem.get_memory("default_model") or cfg.DEFAULT_MODEL

    # Storage mode
    storage_mode = cfg.STORAGE_MODE or "local"
    has_d1 = bool(cfg.CF_D1_DATABASE_ID and cfg.CF_API_TOKEN)

    # Interface mode
    interface_mode = cfg.INTERFACE_MODE or "cli"

    # Composio
    has_composio = bool(cfg.COMPOSIO_API_KEY or mem.get_credential("COMPOSIO_API_KEY"))

    # Overall: configured means provider + model are set (minimum viable)
    configured = has_provider and has_model

    return {
        "configured": configured,
        "has_provider": has_provider,
        "provider_name": provider_name,
        "has_model": has_model,
        "default_model": default_model,
        "storage_mode": storage_mode,
        "has_d1": has_d1,
        "interface_mode": interface_mode,
        "has_composio": has_composio,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# Chat session for web frontend (separate from Telegram chat_id)
WEB_CHAT_ID = 999999  # Default chat_id for web interface
_chat_lock = asyncio.Lock()
_pending_approval = {"active": False, "description": "", "resolved": None}


def _get_active_chat_id() -> int:
    """Resolve the active session's chat_id from DB. Falls back to WEB_CHAT_ID."""
    try:
        active_id = mem.get_memory("active_session") or "default"
        sessions_raw = mem.get_memory("web_sessions")
        if sessions_raw:
            sessions_list = json.loads(sessions_raw)
            for s in sessions_list:
                if s.get("id") == active_id:
                    return s.get("chat_id", WEB_CHAT_ID)
    except Exception:
        pass
    return WEB_CHAT_ID


@app.post("/api/chat/send", dependencies=[Depends(verify_token)])
async def chat_send(msg: ChatMessage):
    """Send a message and get streaming response via SSE."""
    try:
        model = msg.model or mem.get_memory("default_model") or cfg.DEFAULT_MODEL
        user_message = msg.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Empty message")

        # Save user message
        chat_id = _get_active_chat_id()
        mem.save_message(chat_id, "user", user_message)

        # Get conversation history
        history = mem.get_messages(chat_id, limit=cfg.MAX_HISTORY_MESSAGES)

        # Resolve client
        client = None
        api_model = model
        try:
            from agent import _resolve_client_and_model
            client, api_model, provider = _resolve_client_and_model(model)
        except Exception:
            pass

        if client is None:
            # Fallback — resolve model prefix to provider + base_url + stripped model name
            api_model, base_url, provider = _resolve_model_routing(model)
            api_key = ""

            # Get the API key for this provider
            key_name = _provider_key_name(provider)
            try:
                api_key = mem.get_credential(key_name) or ""
            except Exception:
                api_key = ""

            # Also check env-level key
            if not api_key:
                api_key = cfg.OPENAI_API_KEY or ""

            if not api_key:
                async def error_gen():
                    yield f"data: {json.dumps({'error': 'No API key configured for ' + provider + '. Go to Providers to add one.'})}\n\n"
                return StreamingResponse(error_gen(), media_type="text/event-stream")

            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)

        # Build messages for LLM
        system_prompt = (
            "You are Conclave, a personal AI agent running on the user's server. "
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
                import re
                clean = re.sub(r"<think>[\s\S]*?</think>", "", full_reply).strip()
                mem.save_message(chat_id, "assistant", clean or "(empty response)")
                yield f"data: {json.dumps({'done': True, 'full': clean or '(empty response)'})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        # Catch-all: if ANYTHING above crashes, return error as SSE
        logger.error(f"chat_send crashed: {e}", exc_info=True)
        async def crash_gen():
            yield f"data: {json.dumps({'error': f'Server error: {str(e)}'})}\n\n"
        return StreamingResponse(crash_gen(), media_type="text/event-stream")


@app.post("/api/chat/run", dependencies=[Depends(verify_token)])
async def chat_run(msg: ChatMessage):
    """Run the full agentic loop (same as Telegram/CLI) with structured SSE events.

    Event types:
    - thinking: agent is processing
    - tool_call: agent is calling a tool {tool, args}
    - tool_result: tool execution result {tool, output}
    - agent_spawn: sub-agent created {agent}
    - plan: orchestrator plan {steps}
    - text: intermediate text message
    - progress: status update
    - done: final response {full}
    - error: error occurred {message}
    """
    import asyncio as _asyncio

    model = msg.model or mem.get_memory("default_model") or cfg.DEFAULT_MODEL
    user_message = msg.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message")

    chat_id = _get_active_chat_id()
    event_queue: _asyncio.Queue = _asyncio.Queue()

    async def send_fn(text: str):
        """Callback passed to run_agent — intercepts intermediate messages."""
        if not text:
            return
        # Detect event type from text patterns
        if text.startswith("🔄") or text.startswith("📋"):
            await event_queue.put({"type": "progress", "content": text})
        elif text.startswith("  ──•"):
            # Agent delegation step
            await event_queue.put({"type": "agent_step", "content": text})
        else:
            await event_queue.put({"type": "text", "content": text})

    async def run_agent_task():
        """Run the agent in background, push events to queue."""
        try:
            await event_queue.put({"type": "thinking", "content": "Processing..."})

            # Try importing and running the full agent
            from agent import run_agent, ACTIVE_TASKS
            ACTIVE_TASKS[chat_id] = True

            reply, media = await run_agent(
                chat_id=chat_id,
                user_message=user_message,
                model=model,
                send_fn=send_fn,
            )

            if reply:
                await event_queue.put({"type": "done", "full": reply})
            else:
                await event_queue.put({"type": "done", "full": "(No response)"})

        except Exception as e:
            logger.error(f"run_agent failed: {e}", exc_info=True)
            await event_queue.put({"type": "error", "message": str(e)})
        finally:
            await event_queue.put(None)  # Signal stream end

    async def generate():
        """SSE generator — reads from event queue."""
        # Start the agent task in background
        task = _asyncio.create_task(run_agent_task())

        try:
            while True:
                try:
                    event = await _asyncio.wait_for(event_queue.get(), timeout=120.0)
                except _asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Timeout waiting for agent response'})}\n\n"
                    break

                if event is None:
                    break  # Stream end

                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/chat/debug", dependencies=[Depends(verify_token)])
async def chat_debug():
    """Debug endpoint — shows what model/provider/key would be used for chat.
    Hit this from browser: /api/chat/debug (add X-API-Token header or use ?token=xxx)"""
    model = mem.get_memory("default_model") or cfg.DEFAULT_MODEL
    api_model, base_url, provider = _resolve_model_routing(model)
    key_name = _provider_key_name(provider)

    # Check what key we'd use
    has_credential = False
    try:
        cred = mem.get_credential(key_name)
        has_credential = bool(cred)
    except Exception as e:
        has_credential = f"ERROR: {e}"

    has_env_key = bool(cfg.OPENAI_API_KEY)
    chat_id = _get_active_chat_id()
    history_count = len(mem.get_messages(chat_id, limit=100))

    return {
        "stored_model": model,
        "api_model": api_model,
        "base_url": base_url,
        "provider": provider,
        "key_name": key_name,
        "has_credential_in_store": has_credential,
        "has_env_key": has_env_key,
        "chat_id": chat_id,
        "history_count": history_count,
        "base_url_empty": base_url == "" or base_url is None,
    }


@app.get("/api/chat/history", dependencies=[Depends(verify_token)])
async def chat_history(limit: int = 50):
    """Get conversation history for active session."""
    chat_id = _get_active_chat_id()
    messages = mem.get_messages(chat_id, limit=limit)
    return {"messages": messages}


@app.post("/api/chat/clear", dependencies=[Depends(verify_token)])
async def chat_clear():
    """Clear active session chat history."""
    chat_id = _get_active_chat_id()
    mem.clear_messages(chat_id)
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
    try:
        from agents import get_society_status
        return get_society_status()
    except Exception:
        return {"agents": {"active": 0, "by_status": {}, "by_role": {}, "total_completed": 0}, "tree": [], "active": []}


@app.get("/api/society/stream", dependencies=[Depends(verify_token)])
async def society_stream():
    """SSE stream — pushes agent society state every 1s while active."""
    try:
        from agents import get_society_status
    except Exception:
        get_society_status = lambda: {"agents": {"active": 0}, "tree": [], "active": []}

    async def generate():
        while True:
            status = get_society_status()
            yield f"data: {json.dumps(status)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/society/reset", dependencies=[Depends(verify_token)])
async def society_reset():
    """Reset all agents."""
    try:
        from agents import reset_society
        reset_society()
    except Exception:
        pass
    return {"success": True}


# ══════════════════════════════════════════════════════════════════════════════
#  PROVIDERS & MODELS ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/providers", dependencies=[Depends(verify_token)])
async def list_providers():
    """List all providers with configuration status."""
    providers = []
    for name, meta in PROVIDER_META.items():
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
    try:
        from model_fetcher import fetch_provider_models
        models = fetch_provider_models(name, force=True)
    except Exception:
        models = []
    if not models:
        # Fallback to catalog
        models = list(cfg.MODEL_CATALOG.get(name, []))
    return {"provider": name, "models": models}


@app.post("/api/providers/{name}/key", dependencies=[Depends(verify_token)])
async def store_provider_key(name: str, item: ProviderKeyItem):
    """Store API key for a provider."""
    key_name = _provider_key_name(name)
    mem.store_credential(key_name, item.key)
    return {"success": True, "key_name": key_name}


@app.delete("/api/providers/{name}/key", dependencies=[Depends(verify_token)])
async def delete_provider_key(name: str):
    """Remove stored API key for a provider."""
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
    try:
        from model_fetcher import get_all_available_models
        models = get_all_available_models()
    except Exception:
        # Fallback: flatten catalog
        models = [m for provider_models in cfg.MODEL_CATALOG.values() for m in provider_models]
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
async def api_install_skill(item: SkillInstallItem):
    """Install a skill from source."""
    try:
        from tools import install_skill
        result = install_skill(item.source)  # pass string directly, not dict
        return result if isinstance(result, dict) else json.loads(result) if isinstance(result, str) else {"result": str(result)}
    except ImportError:
        raise HTTPException(status_code=400, detail="Skills system not available (agent not fully initialized)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/skills/{name}", dependencies=[Depends(verify_token)])
async def api_uninstall_skill(name: str):
    """Uninstall a skill."""
    try:
        from tools import uninstall_skill
        result = uninstall_skill(name)  # pass string directly
        return result if isinstance(result, dict) else json.loads(result) if isinstance(result, str) else {"result": str(result)}
    except ImportError:
        raise HTTPException(status_code=400, detail="Skills system not available")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/skills/reinstall", dependencies=[Depends(verify_token)])
async def reinstall_skills():
    """Reinstall all skills from sources."""
    try:
        from tools import reinstall_all_skills
        result = reinstall_all_skills({})
        return json.loads(result) if isinstance(result, str) else {"result": result}
    except ImportError:
        raise HTTPException(status_code=400, detail="Skills system not available")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



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
    """Get current non-secret configuration (reads from DB overrides first, then env defaults)."""
    return {
        "interface_mode": mem.get_memory("config_interface_mode") or cfg.INTERFACE_MODE,
        "storage_mode": mem.get_memory("config_storage_mode") or cfg.STORAGE_MODE,
        "default_model": mem.get_memory("default_model") or cfg.DEFAULT_MODEL,
        "max_tool_iterations": int(mem.get_memory("config_max_tool_iterations") or cfg.MAX_TOOL_ITERATIONS),
        "max_history_messages": cfg.MAX_HISTORY_MESSAGES,
        "checkpoint_every": cfg.CHECKPOINT_EVERY,
        "max_rpm": int(mem.get_memory("config_max_rpm") or cfg.MAX_RPM),
        "base_dir": str(cfg.BASE_DIR),
        "has_composio": bool(cfg.COMPOSIO_API_KEY or mem.get_credential("COMPOSIO_API_KEY")),
        "has_d1": bool(cfg.CF_D1_DATABASE_ID and cfg.CF_API_TOKEN) or bool(mem.get_memory("cf_d1_database_id")),
    }


@app.post("/api/system/config", dependencies=[Depends(verify_token)])
async def update_system_config(item: ConfigUpdateItem):
    """Update a configuration value."""
    allowed_keys = {"default_model", "max_tool_iterations", "max_history_messages", "max_rpm", "interface_mode", "storage_mode"}
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


@app.post("/api/system/run", dependencies=[Depends(verify_token)])
async def system_run(request: Request):
    """Execute a shell command on the server (owner-only, dangerous)."""
    import subprocess
    body = await request.json()
    command = body.get("command", "").strip()
    if not command:
        raise HTTPException(status_code=400, detail="No command provided")
    timeout = min(body.get("timeout", 30), 60)  # cap at 60s
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cfg.WORKSPACE_DIR),
        )
        return {
            "stdout": result.stdout[-5000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}



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

@app.get("/api/mcp/servers", dependencies=[Depends(verify_token)])
async def list_mcp_servers():
    """List all configured MCP servers."""
    all_mem = mem.get_all_memory()
    servers = []
    for key, value in all_mem.items():
        if key.startswith("mcp:"):
            name = key[4:]  # strip "mcp:" prefix
            try:
                config_data = json.loads(value)
            except Exception:
                config_data = {}
            servers.append({
                "name": name,
                "config": config_data,
                "transport": config_data.get("transport", "stdio") if "transport" in config_data else ("stdio" if "command" in config_data else "sse"),
            })
    return {"servers": servers}


@app.post("/api/mcp/servers", dependencies=[Depends(verify_token)])
async def add_mcp_server(request: Request):
    """Add MCP server(s) from JSON config."""
    body = await request.json()
    json_config = body.get("config", {})

    # Support both {mcpServers: {...}} and direct {name: {command, args}}
    if isinstance(json_config, str):
        try:
            json_config = json.loads(json_config)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

    servers = json_config.get("mcpServers", json_config)
    if not isinstance(servers, dict) or not servers:
        raise HTTPException(status_code=400, detail="Expected JSON with server configs")

    added = []
    for name, conf in servers.items():
        mem.set_memory(f"mcp:{name}", json.dumps(conf))
        added.append(name)

    return {"success": True, "added": added}


@app.delete("/api/mcp/servers/{name}", dependencies=[Depends(verify_token)])
async def remove_mcp_server(name: str):
    """Remove an MCP server configuration."""
    key = f"mcp:{name}"
    if not mem.get_memory(key):
        raise HTTPException(status_code=404, detail=f"MCP server not found: {name}")
    try:
        conn = mem._get_conn()
        conn.execute("DELETE FROM memory WHERE key = ?", (key,))
        conn.commit()
        conn.close()
    except Exception:
        pass
    return {"success": True}


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
    """List Composio connected accounts (fetched live from Composio API)."""
    composio_key = cfg.COMPOSIO_API_KEY or mem.get_credential("COMPOSIO_API_KEY") or ""
    if not composio_key:
        return {"connections": [], "available": False}
    try:
        import requests as req
        resp = req.get(
            "https://backend.composio.dev/api/v3/connected_accounts",
            headers={"x-api-key": composio_key},
            params={"limit": 100},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", data.get("connected_accounts", []))
            connections = []
            for item in items:
                toolkit = item.get("toolkit", {})
                connections.append({
                    "id": item.get("id") or item.get("nanoid", ""),
                    "app": toolkit.get("name") or toolkit.get("slug", "") if isinstance(toolkit, dict) else str(toolkit),
                    "slug": toolkit.get("slug", "") if isinstance(toolkit, dict) else str(toolkit),
                    "status": item.get("status", "unknown"),
                    "user_id": item.get("user_id", ""),
                })
            return {"connections": connections, "available": True}
        return {"connections": [], "available": True}
    except Exception:
        return {"connections": [], "available": True}


@app.get("/api/composio/tools", dependencies=[Depends(verify_token)])
async def composio_tools(page: int = 1, search: str = "", toolkit: str = ""):
    """List available Composio toolkits (apps) with pagination."""
    composio_key = cfg.COMPOSIO_API_KEY or mem.get_credential("COMPOSIO_API_KEY") or ""
    if not composio_key:
        return {"items": [], "available": False, "total_pages": 0}
    try:
        import requests as req
        params = {"page": page, "limit": 30}
        if search:
            params["search"] = search

        # Fetch toolkits (apps) — not individual tools (23K+)
        resp = req.get(
            "https://backend.composio.dev/api/v3/toolkits",
            headers={"x-api-key": composio_key},
            params=params,
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            raw_items = data.get("items", data.get("toolkits", []))
            # Normalize fields (API may use icon/logo_url/logo inconsistently)
            items = []
            for item in raw_items:
                items.append({
                    "slug": item.get("slug") or item.get("name", ""),
                    "name": item.get("name") or item.get("display_name") or item.get("slug", ""),
                    "description": item.get("description") or item.get("short_description") or "",
                    "logo": item.get("logo") or item.get("icon") or item.get("logo_url") or item.get("image", ""),
                    "tags": item.get("tags") or item.get("categories") or [],
                })
            return {
                "items": items,
                "total_pages": data.get("total_pages", 1),
                "current_page": data.get("current_page", page),
                "total_items": data.get("total_items", len(items)),
                "available": True,
            }
        # Fallback: try /tools endpoint with toolkit grouping
        resp2 = req.get(
            "https://backend.composio.dev/api/v3/tools",
            headers={"x-api-key": composio_key},
            params={"page": page, "limit": 30, "search": search} if search else {"page": page, "limit": 30},
            timeout=15,
        )
        if resp2.status_code == 200:
            data = resp2.json()
            return {
                "items": data.get("items", []),
                "total_pages": data.get("total_pages", 0),
                "current_page": data.get("current_page", page),
                "total_items": data.get("total_items", 0),
                "available": True,
            }
        return {"items": [], "available": True, "total_pages": 0, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"items": [], "available": True, "total_pages": 0, "error": str(e)}


@app.post("/api/composio/connect/{toolkit}", dependencies=[Depends(verify_token)])
async def composio_connect(toolkit: str, request: Request):
    """Initiate connection for a Composio toolkit.
    
    Uses POST /api/v3/connected_accounts directly with toolkit slug.
    This is how the official Composio SDK works — no auth_config lookup needed
    for Composio-managed OAuth apps.
    """
    composio_key = cfg.COMPOSIO_API_KEY or mem.get_credential("COMPOSIO_API_KEY") or ""
    if not composio_key:
        raise HTTPException(status_code=400, detail="Composio API key not configured")
    try:
        import requests as req
        headers = {"x-api-key": composio_key, "Content-Type": "application/json"}
        base = "https://backend.composio.dev/api/v3"
        body = await request.json() if request.headers.get("content-length", "0") != "0" else {}

        # Direct approach: POST /connected_accounts with toolkit slug
        # This uses Composio's managed OAuth for the toolkit
        link_resp = req.post(
            f"{base}/connected_accounts",
            headers=headers,
            json={
                "toolkit": toolkit,
                "user_id": body.get("user_id", "default"),
                "redirect_uri": body.get("callback_url", ""),
            },
            timeout=15,
        )

        if link_resp.status_code in (200, 201):
            data = link_resp.json()
            redirect_url = (
                data.get("redirect_url") or
                data.get("redirectUrl") or
                data.get("url") or
                data.get("connectionRequest", {}).get("redirectUrl") or
                ""
            )
            return {
                "success": True,
                "redirectUrl": redirect_url,
                "connection_id": data.get("id") or data.get("nanoid") or data.get("connected_account_id", ""),
            }

        # If direct fails, try the link endpoint with auth_config discovery
        # Get auth_configs filtered to this toolkit
        auth_resp = req.get(
            f"{base}/auth_configs",
            headers=headers,
            params={"toolkit": toolkit},
            timeout=10,
        )
        auth_config_id = ""
        if auth_resp.status_code == 200:
            configs = auth_resp.json().get("items", [])
            for ac in configs:
                tk = ac.get("toolkit_slug") or ""
                if isinstance(ac.get("toolkit"), dict):
                    tk = ac["toolkit"].get("slug", "")
                elif isinstance(ac.get("toolkit"), str):
                    tk = ac["toolkit"]
                if tk.lower() == toolkit.lower():
                    auth_config_id = ac.get("id") or ac.get("nanoid", "")
                    break

        if auth_config_id:
            link_resp2 = req.post(
                f"{base}/connected_accounts/link",
                headers=headers,
                json={
                    "auth_config_id": auth_config_id,
                    "user_id": body.get("user_id", "default"),
                    "callback_url": body.get("callback_url", ""),
                },
                timeout=15,
            )
            if link_resp2.status_code in (200, 201):
                data = link_resp2.json()
                redirect_url = data.get("redirect_url") or data.get("redirectUrl") or data.get("url", "")
                return {
                    "success": True,
                    "redirectUrl": redirect_url,
                    "connection_id": data.get("id") or data.get("connected_account_id", ""),
                }

        # Return the original error
        return {
            "error": f"Connection failed: HTTP {link_resp.status_code}",
            "detail": link_resp.text[:300],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
