import chalk from "chalk";
import { execSync } from "child_process";
import { config, remoteExec, printError, printInfo } from "../utils.js";

export async function runLogs(args) {
  const lines = args[0] || "50";
  const follow = args.includes("-f") || args.includes("--follow");

  printInfo(`Showing last ${lines} lines of agent logs...`);
  console.log("");

  try {
    if (follow) {
      // For follow mode, we need to use spawn with stdio inherit
      const host = config.get("remote_host");
      const user = config.get("remote_user") || "root";
      const baseDir = config.get("base_dir");

      if (host) {
        execSync(
          `ssh ${user}@${host} 'journalctl -u agent -f --no-pager -n ${lines}'`,
          { stdio: "inherit" }
        );
      } else {
        execSync(`journalctl -u agent -f --no-pager -n ${lines}`, {
          stdio: "inherit",
        });
      }
    } else {
      const output = remoteExec(
        `journalctl -u agent --no-pager -n ${lines} 2>/dev/null || tail -n ${lines} ${config.get("base_dir")}/agent.log 2>/dev/null || echo "No logs found."`,
        { timeout: 10000 }
      );
      console.log(output);
    }
  } catch (err) {
    printError("Could not retrieve logs: " + err.message);
  }
}
