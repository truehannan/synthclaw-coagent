import chalk from "chalk";
import ora from "ora";
import inquirer from "inquirer";
import { execSync } from "child_process";
import { existsSync, writeFileSync } from "fs";
import { join } from "path";
import { config, getProjectRoot, generateEnvContent, printSuccess, printError, printInfo } from "../utils.js";

const RD = chalk.hex("#e85d04");
const D = chalk.dim;
const isWin = process.platform === "win32";

export async function runDeploy(args) {
  const root = getProjectRoot();

  console.log("");
  console.log("  " + RD("╭─── Deployment Mode ────────────────────────────╮"));
  console.log("  " + RD("│") + "  Where should SynthClaw run?                   " + RD("│"));
  console.log("  " + RD("╰───────────────────────────────────────────────╯"));
  console.log("");

  const { deployMode } = await inquirer.prompt([{
    type: "list",
    name: "deployMode",
    message: "Deploy target:",
    choices: [
      { name: "Localhost — run on this machine (no SSH)", value: "localhost" },
      { name: "Remote server — deploy via SSH", value: "remote" },
    ],
  }]);

  if (deployMode === "localhost") {
    return await deployLocalhost(root);
  }

  // ── Remote deployment ──────────────────────────────────────────────────────
  return await deployRemote(root);
}

// ══════════════════════════════════════════════════════════════════════════════
//  LOCALHOST DEPLOYMENT
// ══════════════════════════════════════════════════════════════════════════════

async function deployLocalhost(root) {
  console.log("");
  console.log(chalk.bold("  Deploying locally at: ") + chalk.cyan(root));
  console.log("");

  // Step 1: Python deps
  const s1 = ora("Installing Python dependencies...").start();
  try {
    const pipCmd = isWin ? "pip install -r requirements.txt -q" : "python3 -m pip install -r requirements.txt -q";
    execSync(pipCmd, { encoding: "utf-8", timeout: 300000, cwd: root, stdio: ["pipe", "pipe", "pipe"] });
    s1.succeed("Python dependencies installed");
  } catch (err) {
    s1.warn("Some deps may have issues");
    printInfo("Run manually: pip install -r requirements.txt");
  }

  // Step 2: Build frontend (if exists)
  const frontendDir = join(root, "frontend");
  if (existsSync(join(frontendDir, "package.json"))) {
    const s2 = ora("Building frontend...").start();
    try {
      execSync("npm install", { encoding: "utf-8", timeout: 120000, cwd: frontendDir, stdio: ["pipe", "pipe", "pipe"] });
      execSync("npx vite build", { encoding: "utf-8", timeout: 120000, cwd: frontendDir, stdio: ["pipe", "pipe", "pipe"] });
      s2.succeed("Frontend built → frontend/dist/");
    } catch (err) {
      s2.warn("Frontend build failed (optional — web UI won't work)");
      printInfo("Run manually: cd frontend && npm install && npx vite build");
    }
  }

  // Step 3: Generate .env if needed
  const envPath = join(root, ".env");
  if (!existsSync(envPath)) {
    const s3 = ora("Generating .env...").start();
    try {
      let envContent = generateEnvContent();
      envContent += `\nSYNTHCLAW_API_PORT=8000\nSYNTHCLAW_API_HOST=127.0.0.1\nSYNTHCLAW_BASE_DIR=${root}\n`;
      writeFileSync(envPath, envContent);
      s3.succeed(".env created");
    } catch { s3.fail(".env write failed"); }
  } else {
    printSuccess(".env already exists");
  }

  // Step 4: Start
  const { startNow } = await inquirer.prompt([{ type: "confirm", name: "startNow", message: "Start the agent now?", default: true }]);
  if (startNow) {
    const s4 = ora("Starting agent...").start();
    try {
      const pythonCmd = isWin ? "python" : "python3";
      const pidFile = join(root, "agent.pid");

      // Kill old process if running
      if (existsSync(pidFile)) {
        try {
          const oldPid = execSync(`cat "${pidFile}"`, { encoding: "utf-8" }).trim();
          if (oldPid) execSync(isWin ? `taskkill /PID ${oldPid} /F 2>nul` : `kill ${oldPid} 2>/dev/null`, { stdio: "pipe" });
        } catch {}
      }

      // Start new process
      const startCmd = isWin
        ? `start /b ${pythonCmd} "${join(root, "main.py")}" > "${join(root, "agent.log")}" 2>&1`
        : `cd "${root}" && nohup ${pythonCmd} main.py >> agent.log 2>&1 & echo $! > agent.pid`;
      execSync(startCmd, { encoding: "utf-8", timeout: 10000, shell: true, cwd: root });
      await new Promise(r => setTimeout(r, 2000));

      // Health check
      try {
        const resp = await fetch("http://127.0.0.1:8000/api/system/health", { signal: AbortSignal.timeout(3000) });
        if (resp.ok) { s4.succeed("Agent running! API healthy."); }
        else { s4.warn("Agent started but health check returned " + resp.status); }
      } catch { s4.warn("Agent started (API not responding yet — may need a moment)"); }
    } catch (err) {
      s4.fail("Start failed: " + (err.message || "").slice(0, 80));
      printInfo("Start manually: python3 main.py");
    }
  }

  // Summary + auto-serve frontend
  console.log("");
  console.log(RD("━".repeat(50)));
  console.log(chalk.bold("  ✓ LOCAL DEPLOYMENT COMPLETE"));
  console.log(RD("━".repeat(50)));
  console.log("");
  printSuccess("API:      http://localhost:8000");
  printSuccess("Health:   http://localhost:8000/api/system/health");
  if (existsSync(join(root, "frontend", "dist"))) {
    // Actually run the serve command in background
    const servePath = join(root, "frontend", "dist");
    try {
      const serveCmd = isWin
        ? `start "" /b npx --yes serve "${servePath}" -l 3000 -s`
        : `npx --yes serve "${servePath}" -l 3000 -s > /dev/null 2>&1 &`;
      execSync(serveCmd, { encoding: "utf-8", timeout: 15000, shell: true, cwd: root });
      printSuccess("Frontend: http://localhost:3000 (running)");
    } catch {
      printInfo("Frontend: run → npx serve frontend/dist -l 3000 -s");
    }
  }
  console.log("");
  printInfo("Login with the same API key you used in CLI setup.");
  printInfo("Logs: " + join(root, "agent.log"));
  console.log("");
}

// ══════════════════════════════════════════════════════════════════════════════
//  REMOTE DEPLOYMENT
// ══════════════════════════════════════════════════════════════════════════════

async function deployRemote(root) {
  // Check if SSH is available
  try {
    execSync("ssh -V", { encoding: "utf-8", timeout: 5000, stdio: ["pipe", "pipe", "pipe"] });
  } catch {
    printError("SSH not found. Install OpenSSH or use 'Localhost' mode.");
    printInfo("Windows: Settings → Apps → Optional Features → OpenSSH Client");
    return;
  }

  let host = config.get("remote_host") || "";
  let user = config.get("remote_user") || "root";

  const answers = await inquirer.prompt([
    { type: "input", name: "host", message: "Server IP/hostname:", default: host || undefined, validate: v => v.length > 0 || "Required" },
    { type: "input", name: "user", message: "SSH user:", default: user },
  ]);
  host = answers.host;
  user = answers.user;
  config.set("remote_host", host);
  config.set("remote_user", user);

  const baseDir = config.get("base_dir") || "/opt/agent";

  // Test SSH connection first
  const sTest = ora("Testing SSH connection...").start();
  try {
    execSync(`ssh -o ConnectTimeout=5 ${user}@${host} 'echo ok'`, { encoding: "utf-8", timeout: 10000, stdio: ["pipe", "pipe", "pipe"] });
    sTest.succeed("SSH connection OK");
  } catch (err) {
    sTest.fail("Cannot connect to " + host);
    printError("Check: SSH key, host reachability, firewall");
    printInfo("Test: ssh " + user + "@" + host);
    return;
  }

  console.log(chalk.bold("\n  Deploying to ") + chalk.cyan(`${user}@${host}:${baseDir}`) + "\n");

  // Upload files
  const s1 = ora("Uploading agent files...").start();
  try {
    execSync(`ssh ${user}@${host} 'mkdir -p ${baseDir}'`, { encoding: "utf-8", timeout: 10000 });
    const files = ["main.py", "agent.py", "agents.py", "tools.py", "memory.py", "config.py", "model_fetcher.py", "d1_storage.py", "api_server.py", "requirements.txt"]
      .filter(f => existsSync(join(root, f)))
      .map(f => join(root, f))
      .join(" ");
    execSync(`scp ${files} ${user}@${host}:${baseDir}/`, { encoding: "utf-8", timeout: 60000 });
    s1.succeed("Files uploaded");
  } catch (err) {
    s1.fail("Upload failed: " + (err.message || "").slice(0, 80));
    return;
  }

  // Write .env
  const s2 = ora("Writing .env...").start();
  try {
    let envContent = generateEnvContent();
    envContent += `\nSYNTHCLAW_API_PORT=8000\nSYNTHCLAW_API_HOST=0.0.0.0\nSYNTHCLAW_BASE_DIR=${baseDir}\n`;
    const b64 = Buffer.from(envContent).toString("base64");
    execSync(`ssh ${user}@${host} 'echo "${b64}" | base64 -d > ${baseDir}/.env && chmod 600 ${baseDir}/.env'`, { encoding: "utf-8", timeout: 10000 });
    s2.succeed(".env configured");
  } catch (err) {
    s2.fail(".env write failed");
    printError(err.message?.slice(0, 80));
    return;
  }

  // Install deps + start
  const s3 = ora("Installing Python deps (may take a minute)...").start();
  try {
    execSync(`ssh ${user}@${host} 'cd ${baseDir} && pip3 install -r requirements.txt -q 2>&1 || python3 -m pip install -r requirements.txt -q 2>&1'`, { encoding: "utf-8", timeout: 300000 });
    s3.succeed("Dependencies installed");
  } catch { s3.warn("Deps install had issues — may need manual fix"); }

  // Create systemd service and start
  const s4 = ora("Creating systemd service...").start();
  try {
    const service = `[Unit]\nDescription=SynthClaw Agent\nAfter=network.target\n\n[Service]\nType=simple\nUser=${user}\nWorkingDirectory=${baseDir}\nExecStart=/usr/bin/python3 ${baseDir}/main.py\nRestart=always\nRestartSec=5\nEnvironment=SYNTHCLAW_BASE_DIR=${baseDir}\n\n[Install]\nWantedBy=multi-user.target\n`;
    const svcB64 = Buffer.from(service).toString("base64");
    execSync(`ssh ${user}@${host} 'echo "${svcB64}" | base64 -d > /etc/systemd/system/synthclaw.service && systemctl daemon-reload && systemctl enable synthclaw && systemctl restart synthclaw'`, { encoding: "utf-8", timeout: 15000 });
    await new Promise(r => setTimeout(r, 2000));
    const status = execSync(`ssh ${user}@${host} 'systemctl is-active synthclaw 2>/dev/null || echo inactive'`, { encoding: "utf-8", timeout: 5000 }).trim();
    if (status === "active") { s4.succeed("Agent service running!"); }
    else { s4.warn("Service status: " + status); }
  } catch (err) {
    s4.fail("Service setup failed");
    printInfo("Start manually: ssh " + user + "@" + host + " 'cd " + baseDir + " && python3 main.py'");
  }

  // Summary
  console.log("");
  console.log(RD("━".repeat(50)));
  console.log(chalk.bold("  ✓ REMOTE DEPLOYMENT COMPLETE"));
  console.log(RD("━".repeat(50)));
  console.log("");
  printSuccess(`API: http://${host}:8000`);
  printSuccess(`Health: http://${host}:8000/api/system/health`);
  console.log("");
  printInfo("API token: " + baseDir + "/.api_token");
  printInfo("Logs: ssh " + user + "@" + host + " 'journalctl -u synthclaw -f'");
  console.log("");
}
