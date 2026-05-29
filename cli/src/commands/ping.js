import chalk from "chalk";
import { isAgentRunning, remoteExec, config, printSuccess, printError } from "../utils.js";

export async function runPing() {
  const host = config.get("remote_host");

  if (host) {
    // Ping remote
    try {
      remoteExec("echo pong", { timeout: 5000 });
      const running = isAgentRunning();
      if (running) {
        printSuccess(`Agent is alive on ${host}`);
      } else {
        console.log(chalk.yellow("⚠ ") + `Server reachable but agent is ${chalk.red("not running")}`);
        console.log(chalk.dim("  Run: synthclaw start"));
      }
    } catch (err) {
      printError(`Cannot reach ${host}: ${err.message}`);
    }
  } else {
    // Ping local
    const running = isAgentRunning();
    if (running) {
      printSuccess("Agent is alive (local)");
    } else {
      printError("Agent is not running.");
      console.log(chalk.dim("  Run: synthclaw start"));
    }
  }
}
