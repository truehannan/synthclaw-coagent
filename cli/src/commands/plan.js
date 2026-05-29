import chalk from "chalk";
import ora from "ora";
import { config, printError } from "../utils.js";

export async function runPlan(args) {
  const task = args.join(" ").trim();

  if (!task) {
    console.log(chalk.bold("\n  Usage: ") + "synthclaw plan <describe what you want>");
    console.log(
      chalk.dim('  Example: synthclaw plan "set up nginx reverse proxy for port 3000"\n')
    );
    return;
  }

  const apiKey = config.get("openai_api_key");
  const apiBase = config.get("openai_api_base");
  const model = config.get("default_model");

  if (!apiKey) {
    printError("No API key configured. Run: synthclaw setup");
    return;
  }

  const spinner = ora("Thinking...").start();

  try {
    const response = await fetch(`${apiBase}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages: [
          {
            role: "system",
            content:
              "You are a thoughtful planner. Break the request into clear numbered steps. Be specific about what each step does. Do NOT use any tools. Just produce a structured plan. State assumptions upfront.",
          },
          { role: "user", content: task },
        ],
        temperature: 0.7,
        max_tokens: 2048,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      spinner.fail("API error");
      printError(data.error?.message || JSON.stringify(data));
      return;
    }

    const reply = data.choices?.[0]?.message?.content;
    spinner.succeed("Plan ready\n");
    console.log(chalk.dim("─".repeat(50)));
    console.log(reply || "(empty response)");
    console.log(chalk.dim("─".repeat(50)));
    console.log(
      chalk.dim('\n  Execute with: synthclaw agent "' + task.slice(0, 60) + '"\n')
    );
  } catch (err) {
    spinner.fail("Failed to generate plan");
    printError(err.message);
  }
}
