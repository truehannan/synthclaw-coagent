import os
from pathlib import Path

# Load .env file if present (for local development / non-systemd usage)
try:
    from dotenv import load_dotenv
    _env_path = Path(os.getenv("CONCLAVE_BASE_DIR", "/opt/agent")) / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        # Also check current directory
        _local_env = Path(".env")
        if _local_env.exists():
            load_dotenv(_local_env)
except ImportError:
    pass  # python-dotenv not installed, rely on system environment


def _clean_env(key: str, default: str = "") -> str:
    """Get env var and strip quotes + whitespace (guards against systemd EnvironmentFile quoting)."""
    val = os.getenv(key, default).strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
        val = val[1:-1]
    return val


BASE_DIR = Path(_clean_env("CONCLAVE_BASE_DIR", "/opt/agent"))
WORKSPACE_DIR = BASE_DIR / "workspace"
MEDIA_DIR = WORKSPACE_DIR / "media"
DB_PATH = BASE_DIR / "agent.db"
LOG_PATH = BASE_DIR / "agent.log"

# Ensure base directories exist (prevents FileNotFoundError on first run)
BASE_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

INTERFACE_MODE = _clean_env("INTERFACE_MODE", "telegram").lower()
TELEGRAM_TOKEN = _clean_env("TELEGRAM_TOKEN")

# Discord support removed; keep Telegram-only
# Provide TELEGRAM_TOKEN in .env file (never commit secrets)

OPENAI_API_KEY = _clean_env("OPENAI_API_KEY")
OPENAI_API_BASE = _clean_env("OPENAI_API_BASE", "https://inference.do-ai.run/v1")
OPENROUTER_API_BASE = _clean_env("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
GITHUB_MODELS_API_BASE = _clean_env("GITHUB_MODELS_API_BASE", "https://models.inference.ai.azure.com")
NVIDIA_API_BASE = _clean_env("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
HUGGINGFACE_API_BASE = _clean_env("HUGGINGFACE_API_BASE", "https://router.huggingface.co/v1")
GOOGLE_AI_API_BASE = _clean_env("GOOGLE_AI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai")
QWEN_API_BASE = _clean_env("QWEN_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
GOOGLE_SEARCH_API_KEY = _clean_env("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_CX = _clean_env("GOOGLE_SEARCH_CX")

DEFAULT_MODEL = "qwen:qwen-plus"
MODEL_CATALOG = {
    "Qwen": [
        "qwen:qwen-max",
        "qwen:qwen-plus",
        "qwen:qwen-turbo",
        "qwen:qwen3-235b-a22b",
        "qwen:qwen3-32b",
        "qwen:qwen3-30b-a3b",
        "qwen:qwen3-14b",
        "qwen:qwen3-8b",
        "qwen:qwen3-4b",
        "qwen:qwen-coder-plus",
        "qwen:qwen-long",
        "qwen:qwen-vl-max",
        "qwen:qwen-vl-plus",
    ],
    "DigitalOcean": [
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
        # Keep overlaps available in Gradient too when endpoint supports them
        "anthropic-claude-sonnet-4",
        "openai-gpt-4o",
        # Optional additions: shown only if endpoint reports availability
        "google-gemini-2.0-flash-001",
        "xai-grok-3",
        "xai-grok-3-mini",
    ],
    "Anthropic": [
        "anthropic-claude-haiku-4.5",
        "anthropic-claude-sonnet-4",
        "anthropic-claude-4.5-sonnet",
        "anthropic-claude-4.6-sonnet",
        "anthropic-claude-opus-4",
        "anthropic-claude-opus-4.5",
        "anthropic-claude-opus-4.6",
        "anthropic-claude-4.1-opus",
    ],
    "OpenAI": [
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
    ],
    "OpenRouter": [
        "openrouter:openai/gpt-4o-mini",
        "openrouter:openai/gpt-4.1-mini",
        "openrouter:openai/gpt-4.1",
        "openrouter:anthropic/claude-3.5-sonnet",
        "openrouter:anthropic/claude-3.5-haiku",
        "openrouter:google/gemini-2.0-flash-001",
        "openrouter:meta-llama/llama-3.1-70b-instruct",
        "openrouter:mistralai/mistral-small-3.1-24b-instruct",
        "openrouter:deepseek/deepseek-chat-v3-0324",
        "openrouter:qwen/qwen-2.5-72b-instruct",
    ],
    "GitHub": [
        "github:gpt-4o-mini",
        "github:gpt-4o",
        "github:gpt-4.1",
        "github:Meta-Llama-3.1-70B-Instruct",
        "github:Mistral-Large-2411",
        "github:Phi-3.5-mini-instruct",
        "github:DeepSeek-V3",
        "github:Qwen2.5-72B-Instruct",
    ],
    "NVIDIA": [
        "nvidia:meta/llama-3.3-70b-instruct",
        "nvidia:meta/llama-3.1-405b-instruct",
        "nvidia:mistralai/mistral-large-2-instruct",
        "nvidia:mistralai/magistral-small-2506",
        "nvidia:deepseek-ai/deepseek-r1",
        "nvidia:qwen/qwen3-235b-instruct",
        "nvidia:google/gemma-3-27b-it",
        "nvidia:nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia:nvidia/nemotron-mini-4b-instruct",
    ],
    "HuggingFace": [
        "hf:meta-llama/Llama-3.3-70B-Instruct",
        "hf:meta-llama/Llama-3.1-8B-Instruct",
        "hf:mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "hf:Qwen/Qwen3-235B-A22B",
        "hf:deepseek-ai/DeepSeek-R1",
        "hf:google/gemma-3-27b-it",
        "hf:microsoft/phi-4",
        "hf:NousResearch/Hermes-3-Llama-3.1-8B",
    ],
    "Google": [
        "google:gemini-2.5-flash",
        "google:gemini-2.5-pro",
        "google:gemini-2.0-flash",
        "google:gemini-2.0-flash-lite",
        "google:gemini-1.5-pro",
        "google:gemini-1.5-flash",
    ],
    "Cloudflare": [
        "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "cloudflare:@cf/meta/llama-3.1-8b-instruct",
        "cloudflare:@cf/mistral/mistral-7b-instruct-v0.2-lora",
        "cloudflare:@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        "cloudflare:@cf/qwen/qwen1.5-14b-chat-awq",
        "cloudflare:@hf/google/gemma-7b-it",
    ],
}

AVAILABLE_MODELS = [m for provider_models in MODEL_CATALOG.values() for m in provider_models]

MAX_TOOL_ITERATIONS = 200
MAX_HISTORY_MESSAGES = 16
CHECKPOINT_EVERY = 50          # save state + show Continue button after this many loop iterations
MAX_RPM = int(_clean_env("MAX_RPM", "0") or "0")  # 0 = unlimited
COMPOSIO_API_KEY = _clean_env("COMPOSIO_API_KEY")
STORAGE_MODE = _clean_env("STORAGE_MODE", "local")  # local or cloudflare

# ── Cloudflare D1 config ──────────────────────────────────────────────────────
CF_ACCOUNT_ID = _clean_env("CF_ACCOUNT_ID")
CF_D1_DATABASE_ID = _clean_env("CF_D1_DATABASE_ID")
CF_API_TOKEN = _clean_env("CF_API_TOKEN")
CF_R2_BUCKET = _clean_env("CF_R2_BUCKET")

# ── Model pricing (USD per 1M tokens) ──────────────────────────────────────
# (in, out) = confirmed pricing from DO docs / provider published rates
# ~(in, out) = educated estimates based on model size, capability, provider pricing patterns
# Source: https://docs.digitalocean.com/products/gradient-ai-platform/details/pricing/
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # ── DigitalOcean-hosted open source ────────────────────────────────
    "llama3-8b-instruct":           (0.10, 0.10),
    "llama3.3-70b-instruct":        (0.60, 0.60),
    "mistral-nemo-instruct-2407":   (0.15, 0.15),
    "alibaba-qwen3-32b":            (0.25, 0.55),
    # DeepSeek R1 ~ educated guess: reasoning model, similar tier to o1-mini
    "deepseek-r1-distill-llama-70b": (3.00, 12.00),
    # GLM-5, Kimi, MiniMax ~ Chinese models, similar capability tier to claude-sonnet
    "glm-5":                        (2.00, 8.00),
    "kimi-k2.5":                    (2.00, 8.00),
    "minimax-m2.5":                 (1.50, 6.00),
    # OpenAI OSS ~ open source tier pricing
    "openai-gpt-oss-20b":           (0.20, 0.40),
    "openai-gpt-oss-120b":          (1.00, 3.00),
    # Gemini / Grok in Gradient (estimated unless provider publishes exact DO rates)
    "google-gemini-2.0-flash-001":  (0.40, 1.20),
    "xai-grok-3":                   (5.00, 15.00),
    "xai-grok-3-mini":              (1.00, 3.00),
    # ── Anthropic (confirmed from DO pricing page) ───────────
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
    # GPT-5 series ~ educated guesses: frontier reasoning models, similar tier to gpt-4o or higher
    "openai-gpt-4.1":               (5.00, 20.00),
    "openai-gpt-5-nano":            (0.80,  3.20),
    "openai-gpt-5-mini":            (1.50,  6.00),
    "openai-gpt-5":                 (8.00, 32.00),
    "openai-gpt-5.2":               (10.00, 40.00),
    "openai-gpt-5.2-pro":           (15.00, 60.00),
    "openai-gpt-5.3-codex":         (12.00, 48.00),
    "openai-gpt-5.4":               (12.00, 48.00),
    "openai-gpt-5.1-codex-max":     (15.00, 60.00),
    "openai-o1":                    (15.00, 60.00),
    "openai-o3-mini":               (1.10,  4.40),
    "openai-o3":                    (10.00, 40.00),
    # ── NVIDIA NIM (build.nvidia.com pricing) ──────────────────────────
    "nvidia:meta/llama-3.3-70b-instruct":         (0.60, 0.60),
    "nvidia:meta/llama-3.1-405b-instruct":        (3.00, 3.00),
    "nvidia:mistralai/mistral-large-2-instruct":  (2.00, 6.00),
    "nvidia:mistralai/magistral-small-2506":      (0.40, 1.60),
    "nvidia:deepseek-ai/deepseek-r1":             (3.00, 12.00),
    "nvidia:qwen/qwen3-235b-instruct":            (3.00, 12.00),
    "nvidia:google/gemma-3-27b-it":               (0.20, 0.40),
    "nvidia:nvidia/llama-3.1-nemotron-70b-instruct": (0.60, 0.60),
    "nvidia:nvidia/nemotron-mini-4b-instruct":    (0.05, 0.05),
    # ── HuggingFace Inference (serverless, pay-per-token) ──────────────
    "hf:meta-llama/Llama-3.3-70B-Instruct":       (0.36, 0.36),
    "hf:meta-llama/Llama-3.1-8B-Instruct":        (0.05, 0.05),
    "hf:mistralai/Mistral-Small-3.1-24B-Instruct-2503": (0.14, 0.14),
    "hf:Qwen/Qwen3-235B-A22B":                    (0.30, 0.90),
    "hf:deepseek-ai/DeepSeek-R1":                  (0.55, 2.19),
    "hf:google/gemma-3-27b-it":                    (0.14, 0.14),
    "hf:microsoft/phi-4":                          (0.07, 0.07),
    "hf:NousResearch/Hermes-3-Llama-3.1-8B":      (0.05, 0.05),
}

# Marker to know which models have confirmed vs estimated pricing
CONFIRMED_MODELS = {
    "llama3-8b-instruct",
    "llama3.3-70b-instruct",
    "mistral-nemo-instruct-2407",
    "alibaba-qwen3-32b",
    "anthropic-claude-haiku-4.5",
    "anthropic-claude-sonnet-4",
    "anthropic-claude-4.5-sonnet",
    "anthropic-claude-4.6-sonnet",
    "anthropic-claude-opus-4",
    "anthropic-claude-opus-4.5",
    "anthropic-claude-opus-4.6",
    "anthropic-claude-4.1-opus",
    "openai-gpt-4o-mini",
    "openai-gpt-4o",
    "openai-o1",
    "openai-o3-mini",
    "openai-o3",
}

# ── Command timeout tiers ────────────────────────────────────────────────────
DEFAULT_CMD_TIMEOUT = 30       # normal shell commands
INSTALL_CMD_TIMEOUT = 180      # pip install, npm install, apt install, cargo install
BUILD_CMD_TIMEOUT = 300        # builds, compiles, docker build, make
