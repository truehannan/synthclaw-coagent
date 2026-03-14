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

# ── Model pricing (USD per 1M tokens) ──────────────────────────────────────
# Source: https://docs.digitalocean.com/products/gradient-ai-platform/details/pricing/
# and each provider's published rates (DO aligns with them per their docs).
# None = pricing not yet confirmed from DO docs or provider — shown as ? in /models.
MODEL_PRICING: dict[str, tuple[float, float] | None] = {
    # ── DigitalOcean-hosted open source ────────────────────────────────
    # Llama & Mistral: DO published starter rates
    "llama3-8b-instruct":           (0.10, 0.10),
    "llama3.3-70b-instruct":        (0.60, 0.60),
    "mistral-nemo-instruct-2407":   (0.15, 0.15),
    # Alibaba: confirmed from DO docs page
    "alibaba-qwen3-32b":            (0.25, 0.55),
    # DeepSeek / others: not yet confirmed on DO pricing page
    "deepseek-r1-distill-llama-70b": None,
    "glm-5":                        None,
    "kimi-k2.5":                    None,
    "minimax-m2.5":                 None,
    "openai-gpt-oss-20b":           None,
    "openai-gpt-oss-120b":          None,
    # ── Anthropic (all tiers confirmed from DO pricing page) ───────────
    "anthropic-claude-haiku-4.5":   (0.80,  4.00),
    "anthropic-claude-sonnet-4":    (3.00, 15.00),
    "anthropic-claude-4.5-sonnet":  (3.00, 15.00),
    "anthropic-claude-4.6-sonnet":  (3.00, 15.00),
    "anthropic-claude-opus-4":      (15.00, 75.00),
    "anthropic-claude-opus-4.5":    (15.00, 75.00),
    "anthropic-claude-opus-4.6":    (15.00, 75.00),
    "anthropic-claude-4.1-opus":    (15.00, 75.00),
    # ── OpenAI (provider published rates; DO aligns per docs) ──────────
    "openai-gpt-4o-mini":           (0.15,  0.60),
    "openai-gpt-4o":                (2.50, 10.00),
    "openai-gpt-4.1":               None,   # post-cutoff, unconfirmed
    "openai-gpt-5-nano":            None,
    "openai-gpt-5-mini":            None,
    "openai-gpt-5":                 None,
    "openai-gpt-5.2":               None,
    "openai-gpt-5.2-pro":           None,
    "openai-gpt-5.3-codex":         None,
    "openai-gpt-5.4":               None,
    "openai-gpt-5.1-codex-max":     None,
    "openai-o1":                    (15.00, 60.00),
    "openai-o3-mini":               (1.10,  4.40),
    "openai-o3":                    (10.00, 40.00),
}

# ── Command timeout tiers ────────────────────────────────────────────────────
DEFAULT_CMD_TIMEOUT = 30       # normal shell commands
INSTALL_CMD_TIMEOUT = 180      # pip install, npm install, apt install, cargo install
BUILD_CMD_TIMEOUT = 300        # builds, compiles, docker build, make
