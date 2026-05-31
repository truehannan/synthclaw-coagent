import chalk from "chalk";
import ora from "ora";
import { existsSync } from "fs";
import { join } from "path";
import { execSync, spawn } from "child_process";
import { config, getProjectRoot, generateEnvContent, printSuccess, printError, printInfo } from "../utils.js";

/**
 * Detect the Python binary inside venv (or system fallback).
 */
function findPython(root) {
  const venvPython = join(root, "venv", "bin", "python");
  if (existsSync(venvPython)) return venvPython;
  const venvPython3 = join(root, "venv", "bin", "python3");
  if (existsSync(venvPython3)) return venvPython3;
  // system fallback
  try {
    execSync("python3 --version", { encoding: "utf-8" });
    return "python3";
  } catch {
    return "python";
  }
}

/**
 * Check if systemd agent.service is installed.
 */
function hasSystemdService() {
  try {
    const result = execSync("systemctl cat agent 2>/dev/null", { encoding: "utf-8" });
    return result.length > 0;
  } catch {
    return false;
  }
}

/**
 * Create and install the systemd service file.
 */
function installSystemdService(root) {
  const pythonBin = findPython(root);
  const mainPy = join(root, "main.py");
  const envFile = join(root, ".env");

  const serviceContent = `[Unit]
Description=SynthClaw-CoAgent — Personal AI Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${root}
ExecStart=${pythonBin} ${mainPy}
Restart=always
RestartSec=10
StandardOutput=append:${join(root, "agent.log")}
StandardError=append:${join(root, "agent.log")}
${existsSync(envFile) ? `EnvironmentFile=${envFile}` : ""}

[Install]
WantedBy=multi-user.target
`;

  execSync(`echo '${serviceContent.replace(/'/g, "'\\''")}' > /etc/systemd/system/agent.service`, {
    encoding: "utf-8",
  });
  execSync("systemctl daemon-reload", { encoding: "utf-8" });
  execSync("systemctl enable agent.service 2>/dev/null", { encoding: "utf-8" });
}

export async function runStart(args) {
  const foreground = args.includes("--foreground") || args.includes("-f");
  const root = getProjectRoot();
  const mainPy = join(root, "main.py");

  // Verify main.py exists
  if (!existsSync(mainPy)) {
    printError(`Cannot find main.py at ${root}`);
    printInfo("Run 'synthclaw setup' first, or cd into the synthclaw-coagent directory.");
    return;
  }

  // Verify .env exists
  const envFile = join(root, ".env");
  if (!existsSync(envFile)) {
    printError("No .env file found. Run 'synthclaw setup' first.");
    return;
  }

  // Verify venv exists
  const venvDir = join(root, "venv");
  if (!existsSync(venvDir)) {
    printError("Python venv not found. Run 'synthclaw setup' and install dependencies.");
    return;
  }

  // Create workspace dir if missing
  const workspaceDir = join(root, "workspace");
  if (!existsSync(workspaceDir)) {
    try { execSync(`mkdir -p "${workspaceDir}"`, { encoding: "utf-8" }); } catch {}
  }

  // Touch log file if missing
  const logFile = join(root, "agent.log");
  if (!existsSync(logFile)) {
    try { execSync(`touch "${logFile}"`, { encoding: "utf-8" }); } catch {}
  }

  // --- Foreground mode: just run directly ---
  if (foreground) {
    const pythonBin = findPython(root);
    printInfo(`Starting agent in foreground... (Ctrl+C to stop)`);
    console.log(chalk.dim(`  ${pythonBin} ${mainPy}\n`));
    try {
      execSync(`${pythonBin} "${mainPy}"`, {
        cwd: root,
        stdio: "inherit",
        env: { ...process.env, SYNTHCLAW_BASE_DIR: root },
        timeout: 0,
      });
    } catch (err) {
      if (err.signal === "SIGINT") {
        console.log("\n");
        printInfo("Agent stopped.");
      } else {
        printError(err.message);
      }
    }
    return;
  }

  // --- Background mode: use systemd if possible, else nohup ---
  const spinner = ora("Starting SynthClaw agent...").start();

  // Check if we have systemd access
  let canSystemd = false;
  try {
    execSync("systemctl --version 2>/dev/null", { encoding: "utf-8" });
    // Check if we can write to /etc/systemd (root access)
    execSync("test -w /etc/systemd/system", { encoding: "utf-8" });
    canSystemd = true;
  } catch {
    canSystemd = false;
  }

  if (canSystemd) {
    // Install service if not present
    if (!hasSystemdService()) {
      spinner.text = "Installing systemd service...";
      try {
        installSystemdService(root);
        spinner.text = "Starting agent via systemd...";
      } catch (err) {
        spinner.fail("Could not install systemd service");
        printError(err.message);
        printInfo("Falling back to nohup...");
        canSystemd = false;
      }
    }

    if (canSystemd) {
      try {
        execSync("systemctl start agent", { encoding: "utf-8", timeout: 10000 });
        await new Promise((r) => setTimeout(r, 2000));

        // Verify
        const status = execSync("systemctl is-active agent 2>/dev/null || echo inactive", {
          encoding: "utf-8",
        }).trim();

        if (status === "active") {
          spinner.succeed("Agent started (systemd service — persists until machine stops)");
          console.log(chalk.dim("  synthclaw logs     — view output"));
          console.log(chalk.dim("  synthclaw status   — check status"));
          console.log(chalk.dim("  synthclaw stop     — stop the agent"));
        } else {
          spinner.fail("Agent failed to start");
          printError("Check logs: synthclaw logs");
          try {
            const output = execSync("journalctl -u agent --no-pager -n 15 2>/dev/null", { encoding: "utf-8" });
            console.log(chalk.dim(output));
          } catch {}
        }
        return;
      } catch (err) {
        spinner.fail("systemctl start failed");
        printError(err.message);
        printInfo("Falling back to nohup...");
      }
    }
  }

  // Fallback: nohup (works without root/systemd)
  const pythonBin = findPython(root);
  const pidFile = join(root, "agent.pid");

  // Check if already running via pid
  if (existsSync(pidFile)) {
    try {
      const pid = execSync(`cat "${pidFile}"`, { encoding: "utf-8" }).trim();
      execSync(`kill -0 ${pid} 2>/dev/null`, { encoding: "utf-8" });
      spinner.info("Agent is already running (PID: " + pid + ")");
      console.log(chalk.dim("  synthclaw stop     — to stop it"));
      return;
    } catch {
      // stale pid file
    }
  }

  try {
    execSync(
      `SYNTHCLAW_BASE_DIR="${root}" nohup "${pythonBin}" "${mainPy}" >> "${logFile}" 2>&1 & echo $! > "${pidFile}"`,
      { encoding: "utf-8", cwd: root, timeout: 5000 }
    );

    await new Promise((r) => setTimeout(r, 2000));

    // Verify process is alive
    const pid = execSync(`cat "${pidFile}"`, { encoding: "utf-8" }).trim();
    try {
      execSync(`kill -0 ${pid}`, { encoding: "utf-8" });
      spinner.succeed(`Agent started (PID: ${pid} — running in background)`);
      printInfo("Agent will run until you stop it or the machine restarts.");
      console.log(chalk.dim("  synthclaw logs     — view output"));
      console.log(chalk.dim("  synthclaw stop     — stop the agent"));
    } catch {
      spinner.fail("Agent process died immediately after starting");
      printError("Check logs: synthclaw logs");
      try {
        const logTail = execSync(`tail -15 "${logFile}"`, { encoding: "utf-8" });
        console.log(chalk.dim(logTail));
      } catch {}
    }
  } catch (err) {
    spinner.fail("Failed to start agent");
    printError(err.message);
  }
}
