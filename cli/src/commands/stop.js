import chalk from "chalk";
import ora from "ora";
import { existsSync, unlinkSync } from "fs";
import { join } from "path";
import { execSync } from "child_process";
import { getProjectRoot, printSuccess, printError, printInfo } from "../utils.js";

export async function runStop() {
  const root = getProjectRoot();
  const spinner = ora("Stopping Conclave agent...").start();

  let stopped = false;

  // Try systemd first
  try {
    const status = execSync("systemctl is-active agent 2>/dev/null || echo inactive", {
      encoding: "utf-8",
    }).trim();

    if (status === "active") {
      execSync("systemctl stop agent", { encoding: "utf-8", timeout: 10000 });
      await new Promise((r) => setTimeout(r, 1000));
      stopped = true;
      spinner.succeed("Agent stopped (systemd).");
    }
  } catch {}

  // Try pid file
  if (!stopped) {
    const pidFile = join(root, "agent.pid");
    if (existsSync(pidFile)) {
      try {
        const pid = execSync(`cat "${pidFile}"`, { encoding: "utf-8" }).trim();
        execSync(`kill ${pid} 2>/dev/null`, { encoding: "utf-8" });
        await new Promise((r) => setTimeout(r, 1000));
        // Verify it's dead
        try {
          execSync(`kill -0 ${pid} 2>/dev/null`, { encoding: "utf-8" });
          // Still alive, force kill
          execSync(`kill -9 ${pid} 2>/dev/null`, { encoding: "utf-8" });
        } catch {
          // Good, it's dead
        }
        try { unlinkSync(pidFile); } catch {}
        stopped = true;
        spinner.succeed("Agent stopped (PID: " + pid + ").");
      } catch (err) {
        // pid file exists but process not running
        try { unlinkSync(pidFile); } catch {}
      }
    }
  }

  // Try finding by process name
  if (!stopped) {
    try {
      const pids = execSync("pgrep -f 'python.*main.py' 2>/dev/null", {
        encoding: "utf-8",
      }).trim();
      if (pids) {
        execSync(`kill ${pids.split("\n").join(" ")} 2>/dev/null`, { encoding: "utf-8" });
        stopped = true;
        spinner.succeed("Agent stopped (found by process name).");
      }
    } catch {}
  }

  if (!stopped) {
    spinner.info("Agent is not running.");
  }
}
