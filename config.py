from pathlib import Path

BASE_DIR = Path("/opt/agent")
WORKSPACE_DIR = BASE_DIR / "workspace"
MEDIA_DIR = WORKSPACE_DIR / "media"
DB_PATH = BASE_DIR / "agent.db"
LOG_PATH = BASE_DIR / "agent.log"

TELEGRAM_TOKEN = "8738793728:AAGPWBDVqfb6ukUA87XbMP-4pGAVr3gLNUw"

OPENAI_API_KEY = "sk-do-qUSAyRNb9gxU_IdkSOUXOcQCAvKZUjEcWSMcOW2MHjMdZ0zpSW0CVx3kku"
OPENAI_API_BASE = "https://inference.do-ai.run/v1"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
GITHUB_MODELS_API_BASE = "https://models.inference.ai.azure.com"

DEFAULT_MODEL = "llama3.3-70b-instruct"
MODEL_CATALOG = {
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
}

AVAILABLE_MODELS = [m for provider_models in MODEL_CATALOG.values() for m in provider_models]

MAX_TOOL_ITERATIONS = 200
MAX_HISTORY_MESSAGES = 20
CHECKPOINT_EVERY = 50          # save state + show Continue button after this many loop iterations

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
