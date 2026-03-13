"""
SynthClaw-CoAgent — Configuration
All secrets are loaded from environment variables or a .env file.
Run `python setup_cli.py` to configure.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(os.environ.get("SYNTHCLAW_BASE_DIR", "/opt/agent"))
WORKSPACE_DIR = BASE_DIR / "workspace"
MEDIA_DIR = WORKSPACE_DIR / "media"
DB_PATH = BASE_DIR / "agent.db"
LOG_PATH = BASE_DIR / "agent.log"

# ── Messaging ─────────────────────────────────────────────────────────────────

# Telegram
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")

# WhatsApp (Meta Cloud API)
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "synthclaw-verify")
WHATSAPP_PORT = int(os.environ.get("WHATSAPP_PORT", "8443"))

# ── LLM ───────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://inference.do-ai.run/v1")

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "llama3.3-70b-instruct")
AVAILABLE_MODELS = [
    "llama3.3-70b-instruct",
    "llama3-8b-instruct",
    "mistral-nemo-instruct-2407",
    "deepseek-r1-distill-llama-70b",
]

# ── Agent settings ────────────────────────────────────────────────────────────

MAX_TOOL_ITERATIONS = int(os.environ.get("MAX_TOOL_ITERATIONS", "40"))
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "20"))

# ── Interface mode ────────────────────────────────────────────────────────────

# Which bot(s) to run: "telegram", "whatsapp", or "both"
INTERFACE_MODE = os.environ.get("INTERFACE_MODE", "telegram").lower()
