import chalk from "chalk";
import { existsSync } from "fs";
import { join } from "path";
import { execSync } from "child_process";
import { config, getProjectRoot, printSuccess, printInfo } from "../utils.js";

export async function runStatus() {
  const root = getProjectRoot();
  console.log(chalk.bold("\n  SynthClaw Agent Status\n"));

  let running = false;
  let runMode = "unknown";

  // Check systemd
  try {
    const status = execSync("systemctl is-active agent 2>/dev/null || echo inactive", {
      encoding: "utf-8",
    }).trim();
    if (status === "active") {
      running = true;
      runMode = "systemd";
    }
  } catch {}

  // Check pid file
  if (!running) {
    const pidFile = join(root, "agent.pid");
    if (existsSync(pidFile)) {
      try {
        const pid = execSync(`cat "${pidFile}"`, { encoding: "utf-8" }).trim();
        execSync(`kill -0 ${pid} 2>/dev/null`, { encoding: "utf-8" });
        running = true;
        runMode = `nohup (PID: ${pid})`;
      } catch {}
    }
  }

  // Check by process name
  if (!running) {
    try {
      const pids = execSync("pgrep -f 'python.*main.py' 2>/dev/null", {
        encoding: "utf-8",
      }).trim();
      if (pids) {
        running = true;
        runMode = `process (PID: ${pids.split("\n")[0]})`;
      }
    } catch {}
  }

  if (running) {
    printSuccess("Agent is " + chalk.green.bold("RUNNING") + chalk.dim(` [${runMode}]`));
  } else {
    printInfo("Agent is " + chalk.red.bold("STOPPED"));
  }

  console.log("");
  console.log(chalk.dim("  Interface:  ") + config.get("interface_mode"));
  console.log(chalk.dim("  Model:      ") + config.get("default_model"));
  console.log(chalk.dim("  API Base:   ") + config.get("openai_api_base"));
  console.log(chalk.dim("  Project:    ") + root);

  const host = config.get("remote_host");
  if (host) {
    console.log(chalk.dim("  Remote:     ") + `${config.get("remote_user")}@${host}`);
  } else {
    console.log(chalk.dim("  Remote:     ") + "local");
  }

  // Check if venv exists
  const venvDir = join(root, "venv");
  const envFile = join(root, ".env");
  console.log(chalk.dim("  Venv:       ") + (existsSync(venvDir) ? chalk.green("✓") : chalk.red("✗ (run synthclaw setup)")));
  console.log(chalk.dim("  .env:       ") + (existsSync(envFile) ? chalk.green("✓") : chalk.red("✗ (run synthclaw setup)")));

  console.log("");
}
