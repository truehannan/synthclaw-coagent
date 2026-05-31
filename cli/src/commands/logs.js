import chalk from "chalk";
import { existsSync } from "fs";
import { join } from "path";
import { execSync } from "child_process";
import { config, getProjectRoot, printError, printInfo } from "../utils.js";

export async function runLogs(args) {
  const lines = args.find((a) => !a.startsWith("-")) || "50";
  const follow = args.includes("-f") || args.includes("--follow");
  const root = getProjectRoot();
  const logFile = join(root, "agent.log");

  printInfo(`Showing last ${lines} lines of agent logs...`);
  console.log("");

  try {
    if (follow) {
      // Follow mode — try journalctl first, fall back to tail -f
      try {
        execSync("systemctl cat agent 2>/dev/null", { encoding: "utf-8" });
        execSync(`journalctl -u agent -f --no-pager -n ${lines}`, {
          stdio: "inherit",
          timeout: 0,
        });
      } catch {
        if (existsSync(logFile)) {
          execSync(`tail -f -n ${lines} "${logFile}"`, {
            stdio: "inherit",
            timeout: 0,
          });
        } else {
          printError("No log file found. Is the agent running?");
        }
      }
    } else {
      // One-shot: try journalctl, then local log file
      let output = "";
      try {
        output = execSync(
          `journalctl -u agent --no-pager -n ${lines} 2>/dev/null`,
          { encoding: "utf-8", timeout: 5000 }
        );
      } catch {}

      if (!output.trim() && existsSync(logFile)) {
        output = execSync(`tail -n ${lines} "${logFile}"`, {
          encoding: "utf-8",
          timeout: 5000,
        });
      }

      if (output.trim()) {
        console.log(output);
      } else {
        printInfo("No logs found. Agent may not have started yet.");
        console.log(chalk.dim(`  Log file: ${logFile}`));
      }
    }
  } catch (err) {
    if (err.signal === "SIGINT") {
      console.log("\n");
    } else {
      printError("Could not retrieve logs: " + err.message);
    }
  }
}
