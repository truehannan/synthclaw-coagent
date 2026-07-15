import chalk from "chalk";
import ora from "ora";
import { existsSync, writeFileSync } from "fs";
import { join } from "path";
import { execSync } from "child_process";
import { config, getProjectRoot, generateEnvContent, printSuccess, printError, printInfo } from "../utils.js";

/**
 * Get the absolute path to venv python. Returns null if venv doesn't exist.
 */
function getVenvPython(root) {
  const venvPython = join(root, "venv", "bin", "python");
  if (existsSync(venvPython)) return venvPython;
  const venvPython3 = join(root, "venv", "bin", "python3");
  if (existsSync(venvPython3)) return venvPython3;
  return null;
}

/**
 * Create venv and install dependencies.
 * Handles Ubuntu/Debian where python3-venv package may not be installed.
 */
function createVenv(root) {
  let systemPython = "python3";
  try { execSync(`${systemPython} --version`, { encoding: "utf-8" }); }
  catch {
    systemPython = "python";
    try { execSync(`${systemPython} --version`, { encoding: "utf-8" }); }
    catch { throw new Error("Python 3 not found. Install Python 3.10+ first."); }
  }

  const venvDir = join(root, "venv");

  // Try creating venv — if it fails, install the correct venv package and retry
  try {
    execSync(`${systemPython} -m venv "${venvDir}"`, { encoding: "utf-8", cwd: root, timeout: 30000 });
  } catch (err) {
    const errMsg = (err.stderr || err.stdout || err.message || "");
    if (errMsg.includes("ensurepip") || errMsg.includes("not created successfully") || errMsg.includes("returned non-zero")) {
      // Detect Python version to install correct package (python3.12-venv, not python3-venv)
      let pyVersion = "";
      try {
        const verOut = execSync(`${systemPython} --version`, { encoding: "utf-8" }).trim();
        const match = verOut.match(/(\d+\.\d+)/);
        if (match) pyVersion = match[1]; // e.g. "3.12"
      } catch {}

      // Install the version-specific venv package
      const packages = pyVersion
        ? `python${pyVersion}-venv python3-venv python3-pip`
        : "python3-venv python3-pip";

      try {
        execSync(`apt-get update -qq && apt-get install -y -qq ${packages}`, {
          encoding: "utf-8", timeout: 90000,
        });
      } catch {
        try {
          execSync(`apt-get install -y ${packages} 2>/dev/null || yum install -y python3-pip 2>/dev/null || true`, {
            encoding: "utf-8", timeout: 60000,
          });
        } catch {}
      }

      // Clean up failed venv attempt and retry
      try { execSync(`rm -rf "${venvDir}"`, { encoding: "utf-8" }); } catch {}
      execSync(`${systemPython} -m venv "${venvDir}"`, { encoding: "utf-8", cwd: root, timeout: 30000 });
    } else {
      throw err;
    }
  }

  const pipBin = join(venvDir, "bin", "pip");
  if (!existsSync(pipBin)) {
    // ensurepip might have failed — install pip manually
    execSync(`${systemPython} -m ensurepip --default-pip 2>/dev/null || curl -sS https://bootstrap.pypa.io/get-pip.py | "${join(venvDir, "bin", "python")}"`, {
      encoding: "utf-8", cwd: root, timeout: 30000,
    });
  }

  execSync(`"${pipBin}" install --upgrade pip -q`, { encoding: "utf-8", timeout: 60000, cwd: root });

  const requirementsPath = join(root, "requirements.txt");
  if (existsSync(requirementsPath)) {
    execSync(`"${pipBin}" install -r "${requirementsPath}" -q`, { encoding: "utf-8", timeout: 180000, cwd: root });
  }
}

/**
 * Verify that the venv python has the critical packages installed.
 */
function verifyImports(pythonBin, root) {
  try {
    execSync(
      `"${pythonBin}" -c "import openai; import telegram; print('ok')"`,
      { encoding: "utf-8", cwd: root, timeout: 10000 }
    );
    return true;
  } catch (err) {
    return false;
  }
}

/**
 * Write systemd service file with absolute paths.
 */
function installSystemdService(root, pythonBin) {
  const mainPy = join(root, "main.py");
  const envFile = join(root, ".env");
  const logFile = join(root, "agent.log");

  const serviceContent = [
    "[Unit]",
    "Description=Conclave-CoAgent — Personal AI Agent",
    "After=network-online.target",
    "Wants=network-online.target",
    "",
    "[Service]",
    "Type=simple",
    `WorkingDirectory=${root}`,
    `Environment=CONCLAVE_BASE_DIR=${root}`,
    `ExecStart=${pythonBin} ${mainPy}`,
    "Restart=always",
    "RestartSec=10",
    `StandardOutput=append:${logFile}`,
    `StandardError=append:${logFile}`,
    existsSync(envFile) ? `EnvironmentFile=${envFile}` : "",
    "",
    "[Install]",
    "WantedBy=multi-user.target",
  ].filter(Boolean).join("\n");

  writeFileSync("/etc/systemd/system/agent.service", serviceContent + "\n");
  execSync("systemctl daemon-reload", { encoding: "utf-8" });
  execSync("systemctl enable agent.service 2>/dev/null", { encoding: "utf-8" });
}

export async function runStart(args) {
  const foreground = args.includes("--foreground") || args.includes("-f");
  const root = getProjectRoot();
  const mainPy = join(root, "main.py");

  // Show what root we resolved
  printInfo(`Project root: ${root}`);

  // Verify main.py exists
  if (!existsSync(mainPy)) {
    printError(`Cannot find main.py at ${root}`);
    printInfo("cd into the conclave-coagent directory and try again.");
    return;
  }

  // Ensure .env exists (write from stored config)
  const envFile = join(root, ".env");
  try {
    writeFileSync(envFile, generateEnvContent());
  } catch (err) {
    if (!existsSync(envFile)) {
      printError("Cannot write .env file. Run 'conclave setup' first.");
      return;
    }
  }

  // Ensure venv exists and is healthy — recreate if broken
  const venvDir = join(root, "venv");
  const venvPip = join(venvDir, "bin", "pip");
  let pythonBin = getVenvPython(root);

  // If venv dir exists but pip or python is missing, it's corrupted — nuke it
  if (existsSync(venvDir) && (!pythonBin || !existsSync(venvPip))) {
    printInfo("Venv is corrupted (missing binaries). Recreating...");
    execSync(`rm -rf "${venvDir}"`, { encoding: "utf-8" });
    pythonBin = null;
  }

  if (!pythonBin) {
    const spinnerVenv = ora("Creating Python virtual environment...").start();
    try {
      createVenv(root);
      pythonBin = getVenvPython(root);
      spinnerVenv.succeed("Virtual environment created + deps installed");
    } catch (err) {
      spinnerVenv.fail("Could not create Python venv");
      printError(err.message);
      return;
    }
  }

  if (!pythonBin) {
    printError("Could not find Python in venv after creation. Something went wrong.");
    return;
  }

  // Verify imports work
  if (!verifyImports(pythonBin, root)) {
    const spinnerDeps = ora("Missing dependencies — installing...").start();
    try {
      const pipBin = join(venvDir, "bin", "pip");
      if (!existsSync(pipBin)) {
        // pip missing — recreate entire venv
        spinnerDeps.text = "Venv broken — recreating...";
        execSync(`rm -rf "${venvDir}"`, { encoding: "utf-8" });
        createVenv(root);
        pythonBin = getVenvPython(root);
      } else {
        const requirementsPath = join(root, "requirements.txt");
        execSync(`"${pipBin}" install -r "${requirementsPath}" -q`, { encoding: "utf-8", timeout: 180000, cwd: root });
      }
      spinnerDeps.succeed("Dependencies installed");

      if (!verifyImports(pythonBin, root)) {
        spinnerDeps.fail("Still can't import modules after install");
        printError("Try: rm -rf venv && conclave start");
        return;
      }
    } catch (err) {
      spinnerDeps.fail("Install failed: " + (err.stderr || err.message).slice(0, 200));
      printInfo("Try: rm -rf venv && conclave start");
      return;
    }
  }

  // Create workspace dir if missing
  const workspaceDir = join(root, "workspace");
  if (!existsSync(workspaceDir)) {
    try { execSync(`mkdir -p "${workspaceDir}"`, { encoding: "utf-8" }); } catch {}
  }

  // Touch log file if missing
  const logFile = join(root, "agent.log");
  if (!existsSync(logFile)) {
    try { writeFileSync(logFile, ""); } catch {}
  }

  // --- Foreground mode ---
  if (foreground) {
    printInfo(`Starting in foreground: ${pythonBin} ${mainPy}`);
    try {
      execSync(`"${pythonBin}" "${mainPy}"`, {
        cwd: root,
        stdio: "inherit",
        env: { ...process.env, CONCLAVE_BASE_DIR: root },
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

  // --- Background mode ---
  const spinner = ora("Starting Conclave agent...").start();

  // Stop any existing agent first
  try { execSync("systemctl stop agent 2>/dev/null", { encoding: "utf-8", timeout: 5000 }); } catch {}
  const pidFile = join(root, "agent.pid");
  if (existsSync(pidFile)) {
    try {
      const oldPid = execSync(`cat "${pidFile}"`, { encoding: "utf-8" }).trim();
      execSync(`kill ${oldPid} 2>/dev/null`, { encoding: "utf-8" });
    } catch {}
  }

  // Check if we have systemd + root access
  let canSystemd = false;
  try {
    execSync("systemctl --version 2>/dev/null", { encoding: "utf-8" });
    execSync("test -w /etc/systemd/system", { encoding: "utf-8" });
    canSystemd = true;
  } catch {}

  if (canSystemd) {
    spinner.text = "Configuring systemd service...";
    try {
      installSystemdService(root, pythonBin);
      spinner.text = "Starting agent via systemd...";
      execSync("systemctl start agent", { encoding: "utf-8", timeout: 10000 });
      await new Promise((r) => setTimeout(r, 3000));

      const status = execSync("systemctl is-active agent 2>/dev/null || echo inactive", {
        encoding: "utf-8",
      }).trim();

      if (status === "active") {
        spinner.succeed("Agent started (systemd — persists until machine stops)");
        printInfo(`Using: ${pythonBin}`);
        console.log(chalk.dim("  conclave logs     — view output"));
        console.log(chalk.dim("  conclave status   — check status"));
        console.log(chalk.dim("  conclave stop     — stop the agent"));
        return;
      } else {
        spinner.fail("Agent failed to start via systemd");
        try {
          const output = execSync("journalctl -u agent --no-pager -n 10 2>/dev/null", { encoding: "utf-8" });
          console.log(chalk.dim(output));
        } catch {}
        // Show the actual error from agent.log
        try {
          const logTail = execSync(`tail -10 "${logFile}" 2>/dev/null`, { encoding: "utf-8" });
          if (logTail.trim()) {
            console.log(chalk.dim("\nagent.log:"));
            console.log(chalk.dim(logTail));
          }
        } catch {}
        return;
      }
    } catch (err) {
      spinner.warn("systemd failed, trying nohup...");
    }
  }

  // Fallback: nohup
  try {
    execSync(
      `CONCLAVE_BASE_DIR="${root}" nohup "${pythonBin}" "${mainPy}" >> "${logFile}" 2>&1 & echo $! > "${pidFile}"`,
      { encoding: "utf-8", cwd: root, timeout: 5000 }
    );
    await new Promise((r) => setTimeout(r, 3000));

    const pid = execSync(`cat "${pidFile}"`, { encoding: "utf-8" }).trim();
    try {
      execSync(`kill -0 ${pid}`, { encoding: "utf-8" });
      spinner.succeed(`Agent started (PID: ${pid})`);
      printInfo(`Using: ${pythonBin}`);
      console.log(chalk.dim("  conclave logs     — view output"));
      console.log(chalk.dim("  conclave stop     — stop the agent"));
    } catch {
      spinner.fail("Agent crashed on startup");
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
