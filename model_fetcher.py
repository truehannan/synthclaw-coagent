"""
Real-time model fetching from provider /v1/models endpoints.

Each provider's model list is fetched live and cached for 5 minutes.
Falls back to hardcoded MODEL_CATALOG in config.py if fetch fails.

Providers:
  - DigitalOcean: inference.do-ai.run/v1/models
  - OpenAI: api.openai.com/v1/models
  - NVIDIA: integrate.api.nvidia.com/v1/models
  - HuggingFace: router.huggingface.co/v1/models
  - Google: generativelanguage.googleapis.com/v1beta/models
  - OpenRouter: openrouter.ai/api/v1/models
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Cache ─────────────────────────────────────────────────────────────────────

CACHE_TTL = 300  # 5 minutes
_cache: dict = {}  # provider -> {"ts": float, "models": list[str]}


def _is_fresh(provider: str) -> bool:
    entry = _cache.get(provider)
    if not entry:
        return False
    return (time.time() - entry["ts"]) < CACHE_TTL


def _set_cache(provider: str, models: list):
    _cache[provider] = {"ts": time.time(), "models": models}


def _get_cache(provider: str) -> list:
    entry = _cache.get(provider)
    return entry["models"] if entry else []


def invalidate_cache(provider: str = ""):
    """Clear cache for a specific provider or all."""
    if provider:
        _cache.pop(provider, None)
    else:
        _cache.clear()


# ── Provider endpoints ────────────────────────────────────────────────────────

PROVIDER_ENDPOINTS = {
    "DigitalOcean": {
        "url": "https://inference.do-ai.run/v1/models",
        "key_env": "OPENAI_API_KEY",
        "prefix": "",
    },
    "OpenAI": {
        "url": "https://api.openai.com/v1/models",
        "key_env": "OPENAI_PROVIDER_API_KEY",
        "prefix": "openai-",
    },
    "NVIDIA": {
        "url": "https://integrate.api.nvidia.com/v1/models",
        "key_env": "NVIDIA_API_KEY",
        "prefix": "nvidia:",
    },
    "HuggingFace": {
        "url": "https://router.huggingface.co/v1/models",
        "key_env": "HUGGINGFACE_API_KEY",
        "prefix": "hf:",
    },
    "Google": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models",
        "key_env": "GOOGLE_AI_API_KEY",
        "prefix": "google:",
    },
    "OpenRouter": {
        "url": "https://openrouter.ai/api/v1/models",
        "key_env": "OPENROUTER_API_KEY",
        "prefix": "openrouter:",
    },
    "Qwen": {
        "url": "",  # resolved at runtime from DB/config
        "key_env": "QWEN_API_KEY",
        "prefix": "qwen:",
    },
}


def _get_api_key(provider: str) -> str:
    """Get API key for provider — from credentials store or env."""
    key_env = PROVIDER_ENDPOINTS.get(provider, {}).get("key_env", "")
    if not key_env:
        return ""

    # Try memory credential store first
    try:
        from memory import get_credential
        val = get_credential(key_env)
        if val:
            return val
    except Exception:
        pass

    # Fallback to env / config
    import os
    val = os.getenv(key_env, "").strip()
    if val:
        return val

    # Special fallbacks
    if provider == "DigitalOcean":
        return os.getenv("OPENAI_API_KEY", "").strip()
    if provider == "OpenAI":
        # If no dedicated OpenAI key, no direct fetch possible
        return ""

    return ""


# ── Fetchers ──────────────────────────────────────────────────────────────────

def _fetch_openai_format(url: str, api_key: str, timeout: int = 15) -> list[str]:
    """Fetch models from an OpenAI-compatible /v1/models endpoint."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    models = []
    # Standard format: {"data": [{"id": "model-name", ...}, ...]}
    items = data.get("data", data.get("models", []))
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                model_id = item.get("id", item.get("name", ""))
                if model_id:
                    models.append(model_id)
            elif isinstance(item, str):
                models.append(item)

    return sorted(models)


def _fetch_google_models(api_key: str, timeout: int = 15) -> list[str]:
    """Fetch models from Google's generativelanguage API (different format)."""
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    params = {"key": api_key} if api_key else {}

    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    models = []
    for item in data.get("models", []):
        name = item.get("name", "")
        # Google returns "models/gemini-2.0-flash" — strip prefix
        if name.startswith("models/"):
            name = name[7:]
        if name:
            models.append(name)

    return sorted(models)


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_provider_models(provider: str, force: bool = False) -> list[str]:
    """Fetch live models from a provider. Returns list of model IDs.

    Uses 5-min cache. Returns empty list if provider is unknown or fetch fails.
    The caller should fall back to config.MODEL_CATALOG on empty result.

    Args:
        provider: Provider name (DigitalOcean, OpenAI, NVIDIA, HuggingFace, Google, OpenRouter)
        force: If True, bypass cache and fetch fresh
    """
    if not force and _is_fresh(provider):
        return _get_cache(provider)

    endpoint = PROVIDER_ENDPOINTS.get(provider)
    if not endpoint:
        return []

    api_key = _get_api_key(provider)
    prefix = endpoint["prefix"]
    url = endpoint["url"]

    # Resolve URL from DB if empty (provider-specific override)
    if not url and provider == "Qwen":
        try:
            from memory import get_memory
            url = get_memory("qwen_api_base") or ""
            if url:
                url = url.rstrip("/") + "/models"
        except Exception:
            pass
        if not url:
            import config as cfg
            url = cfg.QWEN_API_BASE.rstrip("/") + "/models"

    if not url:
        return []

    try:
        if provider == "Google":
            raw_models = _fetch_google_models(api_key)
        else:
            raw_models = _fetch_openai_format(url, api_key)

        if not raw_models:
            return _get_cache(provider)  # Return stale cache if fetch returned nothing

        # Apply prefix for providers that need it
        if prefix:
            # Don't double-prefix
            models = []
            for m in raw_models:
                if m.startswith(prefix):
                    models.append(m)
                else:
                    models.append(f"{prefix}{m}")
        else:
            models = raw_models

        _set_cache(provider, models)
        logger.info(f"Fetched {len(models)} models from {provider}")
        return models

    except requests.Timeout:
        logger.warning(f"Timeout fetching models from {provider}")
    except requests.HTTPError as e:
        logger.warning(f"HTTP error fetching models from {provider}: {e}")
    except Exception as e:
        logger.warning(f"Error fetching models from {provider}: {e}")

    # Return stale cache on error
    return _get_cache(provider)


def fetch_all_provider_models(force: bool = False) -> dict:
    """Fetch models from all configured providers.

    Returns {provider: [model_ids]} dict.
    Only fetches from providers that have an API key available.
    """
    results = {}
    for provider in PROVIDER_ENDPOINTS:
        api_key = _get_api_key(provider)
        if api_key or provider in ("OpenRouter",):
            # OpenRouter doesn't always need auth for listing
            models = fetch_provider_models(provider, force=force)
            if models:
                results[provider] = models

    return results


def get_models_for_provider(provider: str) -> list[str]:
    """Get models for a provider — live fetch with fallback to hardcoded catalog."""
    models = fetch_provider_models(provider)
    if models:
        return models

    # Fallback to hardcoded catalog
    try:
        from config import MODEL_CATALOG
        return list(MODEL_CATALOG.get(provider, []))
    except ImportError:
        return []


def get_all_available_models() -> list[str]:
    """Get a flat list of all available models across all providers."""
    all_models = []
    for provider in PROVIDER_ENDPOINTS:
        models = get_models_for_provider(provider)
        all_models.extend(models)

    # Also include providers not in PROVIDER_ENDPOINTS (Anthropic, GitHub)
    try:
        from config import MODEL_CATALOG
        for provider in ("Anthropic", "GitHub"):
            if provider in MODEL_CATALOG:
                all_models.extend(MODEL_CATALOG[provider])
    except ImportError:
        pass

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for m in all_models:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    return unique
