import chalk from "chalk";
import { config } from "../utils.js";

const RED = chalk.hex("#e85d04");

// Static fallback catalog (used when live fetch fails)
const MODEL_CATALOG = {
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
    "anthropic-claude-sonnet-4",
    "openai-gpt-4o",
  ],
  "Anthropic": [
    "anthropic-claude-haiku-4.5",
    "anthropic-claude-sonnet-4",
    "anthropic-claude-opus-4",
  ],
  "OpenAI": [
    "openai-gpt-4o-mini",
    "openai-gpt-4o",
    "openai-gpt-4.1",
    "openai-o3-mini",
    "openai-o3",
  ],
  "OpenRouter": [
    "openrouter:openai/gpt-4o-mini",
    "openrouter:anthropic/claude-3.5-sonnet",
    "openrouter:google/gemini-2.0-flash-001",
    "openrouter:meta-llama/llama-3.1-70b-instruct",
  ],
  "GitHub": [
    "github:gpt-4o-mini",
    "github:gpt-4o",
    "github:Meta-Llama-3.1-70B-Instruct",
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
};

// Provider endpoint map for live fetching
const PROVIDER_ENDPOINTS = {
  "DigitalOcean": { url: "https://inference.do-ai.run/v1/models", keyConfig: "openai_api_key", prefix: "" },
  "OpenAI": { url: "https://api.openai.com/v1/models", keyConfig: "openai_api_key", prefix: "openai-" },
  "NVIDIA": { url: "https://integrate.api.nvidia.com/v1/models", keyConfig: null, prefix: "nvidia:" },
  "HuggingFace": { url: "https://router.huggingface.co/v1/models", keyConfig: null, prefix: "hf:" },
  "Google": { url: "https://generativelanguage.googleapis.com/v1beta/models", keyConfig: null, prefix: "google:", isGoogle: true },
  "OpenRouter": { url: "https://openrouter.ai/api/v1/models", keyConfig: null, prefix: "openrouter:" },
};

// In-memory cache: { provider: { ts, models } }
const _cache = {};
const CACHE_TTL = 300000; // 5 min in ms

async function fetchProviderModels(provider) {
  const ep = PROVIDER_ENDPOINTS[provider];
  if (!ep) return null;

  // Check cache
  const cached = _cache[provider];
  if (cached && (Date.now() - cached.ts) < CACHE_TTL) {
    return cached.models;
  }

  const apiKey = ep.keyConfig ? config.get(ep.keyConfig) : null;

  try {
    const headers = { "Content-Type": "application/json" };
    if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

    let models = [];

    if (ep.isGoogle) {
      // Google uses different format + query param auth
      const url = apiKey ? `${ep.url}?key=${apiKey}` : ep.url;
      const resp = await fetch(url, { headers: { "Content-Type": "application/json" }, signal: AbortSignal.timeout(10000) });
      if (!resp.ok) return null;
      const data = await resp.json();
      models = (data.models || []).map(m => {
        const name = (m.name || "").replace("models/", "");
        return name ? `${ep.prefix}${name}` : null;
      }).filter(Boolean);
    } else {
      const resp = await fetch(ep.url, { headers, signal: AbortSignal.timeout(10000) });
      if (!resp.ok) return null;
      const data = await resp.json();
      const items = data.data || data.models || [];
      models = items.map(item => {
        const id = typeof item === "string" ? item : (item.id || item.name || "");
        if (!id) return null;
        return id.startsWith(ep.prefix) ? id : `${ep.prefix}${id}`;
      }).filter(Boolean);
    }

    if (models.length > 0) {
      _cache[provider] = { ts: Date.now(), models };
      return models;
    }
  } catch (err) {
    // Silently fall through to fallback
  }

  return null;
}

export async function runModels() {
  const current = config.get("default_model");
  console.log(chalk.bold("\n  Available Models\n"));

  for (const [provider, fallbackModels] of Object.entries(MODEL_CATALOG)) {
    // Try live fetch
    let models = await fetchProviderModels(provider);
    let liveTag = "";
    if (models && models.length > 0) {
      liveTag = chalk.dim(` (${models.length} live)`);
    } else {
      models = fallbackModels;
      liveTag = chalk.dim(" (cached)");
    }

    console.log(RED.bold(`  ${provider}`) + liveTag);
    for (const m of models.slice(0, 20)) { // Cap display at 20 per provider
      const isCurrent = m === current;
      const prefix = isCurrent ? chalk.green("  → ") : chalk.dim("    ");
      const name = isCurrent ? chalk.green.bold(m) : m;
      console.log(prefix + name);
    }
    if (models.length > 20) {
      console.log(chalk.dim(`    ... and ${models.length - 20} more`));
    }
    console.log("");
  }

  console.log(
    chalk.dim("  Switch: conclave model <name>")
  );
  console.log("");
}

// Export for use by setup wizard
export { fetchProviderModels, MODEL_CATALOG, PROVIDER_ENDPOINTS };
