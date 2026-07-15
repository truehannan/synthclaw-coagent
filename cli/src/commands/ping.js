import chalk from "chalk";
import { existsSync } from "fs";
import { join } from "path";
import { execSync } from "child_process";
import { config, getProjectRoot, printSuccess, printError } from "../utils.js";

export async function runPing() {
  const host = config.get("remote_host");
  const root = getProjectRoot();

  if (host) {
    // Ping remote
    const user = config.get("remote_user") || "root";
    try {
      execSync(`ssh ${user}@${host} 'echo pong'`, { encoding: "utf-8", timeout: 5000 });
      try {
        const status = execSync(`ssh ${user}@${host} 'systemctl is-active agent 2>/dev/null || echo inactive'`, {
          encoding: "utf-8",
          timeout: 5000,
        }).trim();
        if (status === "active") {
          printSuccess(`Agent is alive on ${host}`);
        } else {
          console.log(chalk.yellow("⚠ ") + `Server reachable but agent is ${chalk.red("not running")}`);
          console.log(chalk.dim("  Run: conclave start"));
        }
      } catch {
        console.log(chalk.yellow("⚠ ") + "Server reachable but couldn't check agent status");
      }
    } catch (err) {
      printError(`Cannot reach ${host}: ${err.message}`);
    }
  } else {
    // Ping local
    let running = false;

    // Check systemd
    try {
      const status = execSync("systemctl is-active agent 2>/dev/null || echo inactive", {
        encoding: "utf-8",
      }).trim();
      if (status === "active") running = true;
    } catch {}

    // Check pid
    if (!running) {
      const pidFile = join(root, "agent.pid");
      if (existsSync(pidFile)) {
        try {
          const pid = execSync(`cat "${pidFile}"`, { encoding: "utf-8" }).trim();
          execSync(`kill -0 ${pid} 2>/dev/null`, { encoding: "utf-8" });
          running = true;
        } catch {}
      }
    }

    // Check process
    if (!running) {
      try {
        const pids = execSync("pgrep -f 'python.*main.py' 2>/dev/null", { encoding: "utf-8" }).trim();
        if (pids) running = true;
      } catch {}
    }

    if (running) {
      printSuccess("Agent is alive (local)");
    } else {
      printError("Agent is not running.");
      console.log(chalk.dim("  Run: conclave start"));
    }
  }
}
