from pathlib import Path

BASE_DIR = Path("/opt/agent")
WORKSPACE_DIR = BASE_DIR / "workspace"
MEDIA_DIR = WORKSPACE_DIR / "media"
DB_PATH = BASE_DIR / "agent.db"
LOG_PATH = BASE_DIR / "agent.log"

TELEGRAM_TOKEN = "8738793728:AAGPWBDVqfb6ukUA87XbMP-4pGAVr3gLNUw"

OPENAI_API_KEY = "sk-do-qUSAyRNb9gxU_IdkSOUXOcQCAvKZUjEcWSMcOW2MHjMdZ0zpSW0CVx3kku"
OPENAI_API_BASE = "https://inference.do-ai.run/v1"

DEFAULT_MODEL = "llama3.3-70b-instruct"
AVAILABLE_MODELS = [
    # ── DigitalOcean-hosted ──────────────────────────
    "llama3.3-70b-instruct",
    "llama3-8b-instruct",
    "mistral-nemo-instruct-2407",
    "deepseek-r1-distill-llama-70b",
    "alibaba-qwen3-32b",
    "glm-5",
    "kimi-k2.5",
    "minimax-m2.5",
    "openai-gpt-oss-20b",
    "openai-gpt-oss-120b",
    # ── Anthropic ────────────────────────────────────
    "anthropic-claude-haiku-4.5",
    "anthropic-claude-sonnet-4",
    "anthropic-claude-4.5-sonnet",
    "anthropic-claude-4.6-sonnet",
    "anthropic-claude-opus-4",
    "anthropic-claude-opus-4.5",
    "anthropic-claude-opus-4.6",
    "anthropic-claude-4.1-opus",
    # ── OpenAI ───────────────────────────────────────
    "openai-gpt-4o-mini",
    "openai-gpt-4o",
    "openai-gpt-4.1",
    "openai-gpt-5-nano",
    "openai-gpt-5-mini",
    "openai-gpt-5",
    "openai-gpt-5.2",
    "openai-gpt-5.2-pro",
    "openai-gpt-5.3-codex",
    "openai-gpt-5.4",
    "openai-gpt-5.1-codex-max",
    "openai-o1",
    "openai-o3-mini",
    "openai-o3",
]

MAX_TOOL_ITERATIONS = 200
MAX_HISTORY_MESSAGES = 20

# ── Command timeout tiers ────────────────────────────────────────────────────
DEFAULT_CMD_TIMEOUT = 30       # normal shell commands
INSTALL_CMD_TIMEOUT = 180      # pip install, npm install, apt install, cargo install
BUILD_CMD_TIMEOUT = 300        # builds, compiles, docker build, make
