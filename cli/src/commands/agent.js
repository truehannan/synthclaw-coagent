import chalk from "chalk";
import ora from "ora";
import { config, remoteExec, printError, printSuccess } from "../utils.js";

export async function runAgent(args) {
  const task = args.join(" ").trim();

  if (!task) {
    console.log(chalk.bold("\n  Usage: ") + "conclave agent <task to execute>");
    console.log(
      chalk.dim('  Example: conclave agent "deploy a node server on port 3000"\n')
    );
    return;
  }

  const apiKey = config.get("openai_api_key");
  const apiBase = config.get("openai_api_base");
  const model = config.get("default_model");

  if (!apiKey) {
    printError("No API key configured. Run: conclave setup");
    return;
  }

  console.log(chalk.hex("#e85d04")(`\n  Agent Mode: `) + chalk.dim(task));
  console.log(chalk.dim("  Model: " + model + "\n"));

  const maxIterations = parseInt(config.get("max_tool_iterations")) || 10;

  const TOOLS_DESC = `Available tools (execute via run_command on server):
- run_command: Execute shell commands
- write_file: Create/overwrite files
- read_file: Read file contents
- spawn_service: Create systemd service
- stop_service: Stop a service
- service_status: Check service status

To use a tool, output JSON: {"tool": "run_command", "args": {"command": "..."}}`;

  const messages = [
    {
      role: "system",
      content: `You are Conclave agent in autonomous mode. Execute the task step by step.
For each step that requires server action, output a JSON block:
{"tool": "run_command", "args": {"command": "shell command here"}}

After each tool result, continue to the next step. When done, say DONE and summarize.
${TOOLS_DESC}`,
    },
    { role: "user", content: task },
  ];

  for (let i = 0; i < maxIterations; i++) {
    const spinner = ora(`Step ${i + 1}...`).start();

    try {
      const response = await fetch(`${apiBase}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages,
          temperature: 0.2,
          max_tokens: 2048,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        spinner.fail("API error");
        printError(data.error?.message || JSON.stringify(data));
        return;
      }

      const reply = data.choices?.[0]?.message?.content || "";
      messages.push({ role: "assistant", content: reply });

      // Check for tool call
      const toolMatch = reply.match(/\{"tool":\s*"run_command",\s*"args":\s*\{"command":\s*"([^"]+)"\}\}/);
      if (toolMatch) {
        const cmd = toolMatch[1];
        spinner.text = `Executing: ${cmd.slice(0, 60)}...`;

        try {
          const output = remoteExec(cmd, { timeout: 60000 });
          spinner.succeed(chalk.dim(`$ ${cmd.slice(0, 70)}`));
          if (output.trim()) {
            console.log(chalk.dim(output.trim().slice(0, 500)));
          }
          messages.push({
            role: "user",
            content: `Tool result (returncode 0):\n${output.slice(0, 2000)}\n\nContinue to next step or say DONE.`,
          });
        } catch (err) {
          spinner.warn(chalk.yellow(`$ ${cmd.slice(0, 70)} (failed)`));
          const errOutput = (err.stderr || err.message || "").slice(0, 1000);
          console.log(chalk.red(errOutput));
          messages.push({
            role: "user",
            content: `Tool FAILED:\n${errOutput}\n\nDiagnose and fix, or report error.`,
          });
        }
      } else {
        // No tool call — agent is done or chatting
        spinner.stop();
        // Clean reply from internal markup
        const clean = reply
          .replace(/<think>[\s\S]*?<\/think>/g, "")
          .replace(/```json[\s\S]*?```/g, "")
          .trim();
        if (clean) {
          console.log("\n" + clean);
        }
        if (reply.toLowerCase().includes("done") || !toolMatch) {
          break;
        }
      }
    } catch (err) {
      spinner.fail("Request failed");
      printError(err.message);
      return;
    }
  }

  console.log("");
}
