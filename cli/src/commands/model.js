import chalk from "chalk";
import { config, printSuccess, printInfo } from "../utils.js";

const ALL_MODELS = [
  "llama3.3-70b-instruct",
  "llama3-8b-instruct",
  "mistral-nemo-instruct-2407",
  "deepseek-r1-distill-llama-70b",
  "alibaba-qwen3-32b",
  "anthropic-claude-sonnet-4",
  "anthropic-claude-opus-4",
  "openai-gpt-4o",
  "openai-gpt-4o-mini",
  "openai-gpt-4.1",
  "openai-o3-mini",
  "openai-o3",
];

export async function runModel(args) {
  const modelName = args.join(" ").trim();

  if (!modelName) {
    // Show current model
    const current = config.get("default_model");
    console.log(chalk.bold("\n  Current model: ") + chalk.cyan(current));
    console.log(chalk.dim("\n  Use: conclave model <name>  to switch"));
    console.log(chalk.dim("  Use: conclave models        to list all\n"));
    return;
  }

  // Switch model
  config.set("default_model", modelName);
  printSuccess(`Switched model to: ${chalk.cyan(modelName)}`);
  printInfo("The agent will use this model for new conversations.");
}
