import chalk from "chalk";
import { config } from "../utils.js";

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
};

export async function runModels() {
  const current = config.get("default_model");
  console.log(chalk.bold("\n  Available Models\n"));

  for (const [provider, models] of Object.entries(MODEL_CATALOG)) {
    console.log(chalk.hex("#e85d04").bold(`  ${provider}`));
    for (const m of models) {
      const isCurrent = m === current;
      const prefix = isCurrent ? chalk.green("  → ") : chalk.dim("    ");
      const name = isCurrent ? chalk.green.bold(m) : m;
      console.log(prefix + name);
    }
    console.log("");
  }

  console.log(
    chalk.dim("  Switch: synthclaw model <name>")
  );
  console.log("");
}
