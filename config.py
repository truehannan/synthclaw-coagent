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
    "llama3.3-70b-instruct",
    "llama3-8b-instruct",
    "mistral-nemo-instruct-2407",
    "deepseek-r1-distill-llama-70b",
]

MAX_TOOL_ITERATIONS = 200
MAX_HISTORY_MESSAGES = 20

# ── Command timeout tiers ────────────────────────────────────────────────────
DEFAULT_CMD_TIMEOUT = 30       # normal shell commands
INSTALL_CMD_TIMEOUT = 180      # pip install, npm install, apt install, cargo install
BUILD_CMD_TIMEOUT = 300        # builds, compiles, docker build, make
