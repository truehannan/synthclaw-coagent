import chalk from "chalk";
import { remoteExec, isAgentRunning, config, printSuccess, printInfo } from "../utils.js";

export async function runStatus() {
  console.log(chalk.bold("\n  SynthClaw Agent Status\n"));

  const running = isAgentRunning();
  if (running) {
    printSuccess("Agent is " + chalk.green.bold("RUNNING"));
  } else {
    printInfo("Agent is " + chalk.red.bold("STOPPED"));
  }

  console.log("");
  console.log(chalk.dim("  Interface:  ") + config.get("interface_mode"));
  console.log(chalk.dim("  Model:      ") + config.get("default_model"));
  console.log(chalk.dim("  API Base:   ") + config.get("openai_api_base"));
  console.log(chalk.dim("  Base Dir:   ") + config.get("base_dir"));

  const host = config.get("remote_host");
  if (host) {
    console.log(
      chalk.dim("  Remote:     ") +
        `${config.get("remote_user")}@${host}`
    );
  } else {
    console.log(chalk.dim("  Remote:     ") + "local");
  }

  // Try to get systemd service info
  if (running) {
    try {
      const output = remoteExec(
        "systemctl show agent --property=ActiveEnterTimestamp,MainPID,MemoryCurrent 2>/dev/null"
      );
      console.log("");
      for (const line of output.trim().split("\n")) {
        const [key, val] = line.split("=");
        if (key && val) {
          console.log(chalk.dim(`  ${key}: `) + val);
        }
      }
    } catch {}
  }

  console.log("");
}
