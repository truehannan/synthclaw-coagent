import chalk from "chalk";
import ora from "ora";
import inquirer from "inquirer";
import { execSync } from "child_process";
import { existsSync, writeFileSync } from "fs";
import { join } from "path";
import { config, getProjectRoot, generateEnvContent, printSuccess, printError, printInfo } from "../utils.js";

const RD = chalk.hex("#e85d04");
const R = chalk.hex("#cc0000");
const D = chalk.dim;
const isWin = process.platform === "win32";

export async function runDeploy(args) {
  let host = config.get("remote_host");
  let user = config.get("remote_user") || "root";

  if (!host) {
    const answers = await inquirer.prompt([
      { type: "input", name: "host", message: "Remote server IP/hostname:", validate: (v) => v.length > 0 || "Required" },
      { type: "input", name: "user", message: "SSH user:", default: "root" },
    ]);
    host = answers.host;
    user = answers.user;
    config.set("remote_host", host);
    config.set("remote_user", user);
  }

  // ── Frontend Deployment Target ─────────────────────────────────────────────
  console.log("");
  console.log("  " + RD("╭─── Frontend Deployment ───────────────────────╮"));
  console.log("  " + RD("│") + "                                               " + RD("│"));
  console.log("  " + RD("│") + "  Where should the frontend be accessible?      " + RD("│"));
  console.log("  " + RD("│") + "                                               " + RD("│"));
  console.log("  " + RD("╰───────────────────────────────────────────────╯"));
  console.log("");

  const { deployMode } = await inquirer.prompt([{
    type: "list",
    name: "deployMode",
    message: "Frontend access mode:",
    choices: [
      { name: `IP only — http://${host}:3000`, value: "ip" },
      { name: `Domain — https://your-domain.com`, value: "domain" },
      { name: `Path — http://${host}/path`, value: "path" },
      { name: `Localhost — http://localhost:3000 (local dev/personal)`, value: "localhost" },
    ],
  }]);

  let domain = "";
  let basePath = "/";
  let frontendPort = 3000;
  let useSSL = false;

  if (deployMode === "domain") {
    const { d } = await inquirer.prompt([{ type: "input", name: "d", message: "Domain (e.g. agent.example.com):" }]);
    domain = d;
    useSSL = true;
    basePath = "/";
  } else if (deployMode === "path") {
    const { p } = await inquirer.prompt([{ type: "input", name: "p", message: "Path (e.g. /agent):", default: "/agent" }]);
    basePath = p.startsWith("/") ? p : "/" + p;
  } else if (deployMode === "localhost") {
    // Localhost mode — run everything locally, no SSH
    return await deployLocalhost(root);
  }

  config.set("deploy_mode", deployMode);
  config.set("deploy_domain", domain);
  config.set("deploy_base_path", basePath);

  const baseDir = config.get("base_dir") || "/opt/agent";
  const root = getProjectRoot();

  console.log("");
  console.log(chalk.bold("  Deploying SynthClaw to ") + chalk.cyan(`${user}@${host}:${baseDir}`));
  if (domain) console.log("  Domain: " + chalk.cyan(`https://${domain}`));
  if (basePath !== "/") console.log("  Path: " + chalk.cyan(`http://${host}${basePath}`));
  console.log("");

  // Step 1: Upload Python agent files
  const spinner1 = ora("Uploading agent files...").start();
  try {
    execSync(`ssh ${user}@${host} 'mkdir -p ${baseDir}'`, { encoding: "utf-8", timeout: 10000 });
    execSync(
      `scp -r ${root}/main.py ${root}/agent.py ${root}/agents.py ${root}/whatsapp_bot.py ${root}/tools.py ${root}/memory.py ${root}/config.py ${root}/model_fetcher.py ${root}/d1_storage.py ${root}/api_server.py ${root}/requirements.txt ${root}/setup_server.sh ${user}@${host}:${baseDir}/`,
      { encoding: "utf-8", timeout: 60000 }
    );
    spinner1.succeed("Agent files uploaded");
  } catch (err) {
    spinner1.fail("Upload failed");
    printError(err.message);
    return;
  }

  // Step 2: Build and upload frontend
  const spinner2 = ora("Building frontend...").start();
  try {
    const frontendDir = `${root}/frontend`;
    // Set base path for Vite build
    const envPrefix = basePath !== "/" ? `VITE_BASE_PATH=${basePath}/ ` : "";
    execSync(`cd ${frontendDir} && ${envPrefix}npx vite build`, { encoding: "utf-8", timeout: 120000 });
    spinner2.text = "Uploading frontend build...";
    execSync(`ssh ${user}@${host} 'mkdir -p ${baseDir}/frontend/dist'`, { encoding: "utf-8", timeout: 10000 });
    execSync(`scp -r ${frontendDir}/dist/* ${user}@${host}:${baseDir}/frontend/dist/`, { encoding: "utf-8", timeout: 60000 });
    spinner2.succeed("Frontend built & uploaded");
  } catch (err) {
    spinner2.fail("Frontend build/upload failed");
    printError(err.message);
    printInfo("You can build manually: cd frontend && npx vite build");
  }

  // Step 3: Write .env on remote
  const spinner3 = ora("Writing .env configuration...").start();
  try {
    let envContent = generateEnvContent();
    envContent += `\nSYNTHCLAW_API_PORT=8000\nSYNTHCLAW_API_HOST=0.0.0.0\n`;
    const b64 = Buffer.from(envContent).toString("base64");
    execSync(
      `ssh ${user}@${host} 'echo "${b64}" | base64 -d > ${baseDir}/.env && chmod 600 ${baseDir}/.env'`,
      { encoding: "utf-8", timeout: 10000 }
    );
    spinner3.succeed(".env written");
  } catch (err) {
    spinner3.fail(".env write failed");
    printError(err.message);
    return;
  }

  // Step 4: Install Python deps
  const spinner4 = ora("Installing Python dependencies...").start();
  try {
    execSync(`ssh ${user}@${host} 'cd ${baseDir} && python3 -m pip install -r requirements.txt -q'`, { encoding: "utf-8", timeout: 300000 });
    spinner4.succeed("Python deps installed");
  } catch (err) {
    spinner4.warn("Deps install had issues (may need manual fix)");
  }

  // Step 5: Configure nginx
  const spinner5 = ora("Configuring nginx...").start();
  try {
    execSync(`ssh ${user}@${host} 'apt-get install -y nginx > /dev/null 2>&1 || true'`, { encoding: "utf-8", timeout: 60000 });

    let nginxConf = "";
    if (deployMode === "domain") {
      nginxConf = generateNginxDomain(domain, baseDir);
    } else if (deployMode === "path") {
      nginxConf = generateNginxPath(basePath, baseDir);
    } else {
      nginxConf = generateNginxIP(frontendPort, baseDir);
    }

    const confB64 = Buffer.from(nginxConf).toString("base64");
    const confName = domain || "synthclaw";
    execSync(
      `ssh ${user}@${host} 'echo "${confB64}" | base64 -d > /etc/nginx/sites-available/${confName} && ln -sf /etc/nginx/sites-available/${confName} /etc/nginx/sites-enabled/${confName} && nginx -t 2>/dev/null && systemctl reload nginx'`,
      { encoding: "utf-8", timeout: 15000 }
    );
    spinner5.succeed("Nginx configured");
  } catch (err) {
    spinner5.warn("Nginx config issues — may need manual setup");
    printInfo(err.message?.slice(0, 100));
  }

  // Step 6: SSL with certbot (domain only)
  if (useSSL && domain) {
    const spinner6 = ora("Setting up SSL (Let's Encrypt)...").start();
    try {
      execSync(`ssh ${user}@${host} 'apt-get install -y certbot python3-certbot-nginx > /dev/null 2>&1 && certbot --nginx -d ${domain} --non-interactive --agree-tos -m admin@${domain} 2>/dev/null || true'`, { encoding: "utf-8", timeout: 120000 });
      spinner6.succeed("SSL certificate installed");
    } catch (err) {
      spinner6.warn("SSL setup failed — add manually with: certbot --nginx -d " + domain);
    }
  }

  // Step 7: Start/restart agent service
  const spinner7 = ora("Starting agent service...").start();
  try {
    // Create systemd service
    const service = generateSystemdService(baseDir, user);
    const svcB64 = Buffer.from(service).toString("base64");
    execSync(`ssh ${user}@${host} 'echo "${svcB64}" | base64 -d > /etc/systemd/system/synthclaw.service && systemctl daemon-reload && systemctl enable synthclaw && systemctl restart synthclaw'`, { encoding: "utf-8", timeout: 15000 });
    await new Promise(r => setTimeout(r, 2000));
    const status = execSync(`ssh ${user}@${host} 'systemctl is-active synthclaw 2>/dev/null || echo inactive'`, { encoding: "utf-8", timeout: 5000 }).trim();
    if (status === "active") {
      spinner7.succeed("Agent service running!");
    } else {
      spinner7.warn("Service status: " + status);
    }
  } catch (err) {
    spinner7.fail("Service start failed");
    printError(err.message?.slice(0, 100));
  }

  // ── Summary ────────────────────────────────────────────────────────────────
  console.log("");
  console.log(RD("━".repeat(54)));
  console.log(chalk.bold("  ✓ DEPLOYMENT COMPLETE"));
  console.log(RD("━".repeat(54)));
  console.log("");

  if (deployMode === "domain") {
    printSuccess(`Frontend: https://${domain}`);
    printSuccess(`API:      https://${domain}/api`);
    console.log("");
    printInfo(`Add an A record: ${domain} → ${host}`);
    printInfo("If DNS isn't propagated yet, SSL may fail. Re-run certbot after DNS is live.");
  } else if (deployMode === "path") {
    printSuccess(`Frontend: http://${host}${basePath}`);
    printSuccess(`API:      http://${host}${basePath}/api`);
  } else {
    printSuccess(`Frontend: http://${host}:80`);
    printSuccess(`API:      http://${host}:8000`);
  }

  console.log("");
  printInfo("API token is stored at: " + baseDir + "/.api_token");
  printInfo("Use that token to login to the web interface.");
  console.log("");
  console.log(D("  synthclaw status   — check status"));
  console.log(D("  synthclaw logs     — view logs"));
  console.log(D("  synthclaw stop     — stop the agent"));
  console.log("");
}

// ── Localhost deployment ──────────────────────────────────────────────────────

async function deployLocalhost(root) {
  console.log("");
  console.log(chalk.bold("  Deploying SynthClaw locally..."));
  console.log("");

  // Step 1: Install Python dependencies
  const spinner1 = ora("Installing Python dependencies...").start();
  try {
    execSync(`cd "${root}" && python3 -m pip install -r requirements.txt -q`, {
      encoding: "utf-8", timeout: 300000, stdio: ["pipe", "pipe", "pipe"],
    });
    spinner1.succeed("Python dependencies installed");
  } catch (err) {
    spinner1.warn("Some deps may have failed — continuing");
    printInfo((err.stderr || "").slice(0, 100));
  }

  // Step 2: Build frontend
  const spinner2 = ora("Building frontend...").start();
  try {
    const frontendDir = join(root, "frontend");
    if (existsSync(join(frontendDir, "package.json"))) {
      execSync(`cd "${frontendDir}" && npm install --silent 2>/dev/null`, { encoding: "utf-8", timeout: 120000, stdio: ["pipe", "pipe", "pipe"] });
      execSync(`cd "${frontendDir}" && npx vite build`, { encoding: "utf-8", timeout: 120000, stdio: ["pipe", "pipe", "pipe"] });
      spinner2.succeed("Frontend built → frontend/dist/");
    } else {
      spinner2.warn("No frontend/package.json found — skipping");
    }
  } catch (err) {
    spinner2.fail("Frontend build failed");
    printError((err.stderr || err.message || "").slice(0, 120));
    printInfo("Try manually: cd frontend && npm install && npx vite build");
  }

  // Step 3: Generate .env if missing
  const envPath = join(root, ".env");
  if (!existsSync(envPath)) {
    const spinner3 = ora("Generating .env...").start();
    try {
      let envContent = generateEnvContent();
      envContent += `\nSYNTHCLAW_API_PORT=8000\nSYNTHCLAW_API_HOST=127.0.0.1\nSYNTHCLAW_BASE_DIR=${root}\n`;
      writeFileSync(envPath, envContent);
      spinner3.succeed(".env created");
    } catch (err) {
      spinner3.fail(".env generation failed");
    }
  } else {
    printSuccess(".env already exists — keeping current config");
  }

  // Step 4: Setup nginx for localhost (optional — can just use Vite proxy)
  const { useNginx } = await inquirer.prompt([{
    type: "confirm",
    name: "useNginx",
    message: "Set up nginx on localhost? (No = use API directly on :8000, frontend on :3000)",
    default: false,
  }]);

  if (useNginx) {
    const spinner4 = ora("Configuring nginx for localhost...").start();
    try {
      const nginxConf = `server {
    listen 3000;
    server_name localhost;
    root ${root}/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_buffering off;
    }
}
`;
      const confPath = isWin ? join(root, "nginx-localhost.conf") : "/etc/nginx/sites-available/synthclaw-local";
      if (isWin) {
        writeFileSync(confPath, nginxConf);
        spinner4.succeed(`Nginx config written to ${confPath}`);
        printInfo("Add this to your nginx.conf manually (Windows)");
      } else {
        writeFileSync(confPath, nginxConf);
        try {
          execSync(`ln -sf ${confPath} /etc/nginx/sites-enabled/synthclaw-local && nginx -t 2>/dev/null && systemctl reload nginx`, {
            encoding: "utf-8", timeout: 10000, stdio: ["pipe", "pipe", "pipe"],
          });
          spinner4.succeed("Nginx configured on localhost:3000");
        } catch {
          spinner4.warn("Nginx config written but couldn't reload (need sudo?)");
          printInfo(`Config at: ${confPath}`);
        }
      }
    } catch (err) {
      spinner4.fail("Nginx setup failed");
    }
  }

  // Step 5: Start the agent
  const { startNow } = await inquirer.prompt([{
    type: "confirm",
    name: "startNow",
    message: "Start the agent now?",
    default: true,
  }]);

  if (startNow) {
    const spinner5 = ora("Starting agent...").start();
    try {
      // Check if already running
      try {
        const pid = execSync(`cat "${join(root, "agent.pid")}" 2>/dev/null`, { encoding: "utf-8" }).trim();
        if (pid) {
          try { process.kill(parseInt(pid), 0); execSync(`kill ${pid} 2>/dev/null`); } catch {}
        }
      } catch {}

      // Start in background
      const cmd = isWin
        ? `start /b python "${join(root, "main.py")}" > "${join(root, "agent.log")}" 2>&1`
        : `cd "${root}" && nohup python3 main.py >> agent.log 2>&1 & echo $! > agent.pid`;

      execSync(cmd, { encoding: "utf-8", timeout: 10000, stdio: ["pipe", "pipe", "pipe"], shell: true });
      await new Promise(r => setTimeout(r, 2000));

      // Verify it started
      try {
        const resp = await fetch("http://127.0.0.1:8000/api/system/health", { signal: AbortSignal.timeout(3000) });
        if (resp.ok) {
          spinner5.succeed("Agent is running!");
        } else {
          spinner5.warn("Agent started but health check failed");
        }
      } catch {
        spinner5.warn("Agent started (health check timed out — may still be initializing)");
      }
    } catch (err) {
      spinner5.fail("Could not start agent");
      printError(err.message?.slice(0, 120));
      printInfo(`Start manually: cd ${root} && python3 main.py`);
    }
  }

  // ── Summary ──────────────────────────────────────────────────────────────
  console.log("");
  console.log(RD("━".repeat(54)));
  console.log(chalk.bold("  ✓ LOCAL DEPLOYMENT COMPLETE"));
  console.log(RD("━".repeat(54)));
  console.log("");

  if (useNginx) {
    printSuccess("Frontend: http://localhost:3000");
    printSuccess("API:      http://localhost:3000/api");
  } else {
    printSuccess("API:      http://localhost:8000");
    printSuccess("Frontend: http://localhost:3000 (run: cd frontend && npx vite)");
    printInfo("Or serve the built frontend: npx serve frontend/dist -l 3000");
  }

  console.log("");
  printInfo("API token is in: " + join(root, ".api_token"));
  printInfo("Logs: " + join(root, "agent.log"));
  console.log("");
  console.log(D("  synthclaw status   — check status"));
  console.log(D("  synthclaw logs     — view logs"));
  console.log(D("  synthclaw stop     — stop the agent"));
  console.log("");
}

// ── Nginx config generators ──────────────────────────────────────────────────

function generateNginxIP(port, baseDir) {
  return `server {
    listen 80 default_server;
    server_name _;
    root ${baseDir}/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }
}
`;
}

function generateNginxDomain(domain, baseDir) {
  return `server {
    listen 80;
    server_name ${domain};
    root ${baseDir}/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }
}
`;
}

function generateNginxPath(basePath, baseDir) {
  const cleanPath = basePath.replace(/\/$/, "");
  return `server {
    listen 80 default_server;
    server_name _;

    location ${cleanPath}/ {
        alias ${baseDir}/frontend/dist/;
        try_files $uri $uri/ ${cleanPath}/index.html;
    }

    location ${cleanPath}/api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }
}
`;
}

function generateSystemdService(baseDir, user) {
  return `[Unit]
Description=SynthClaw Agent Society
After=network.target

[Service]
Type=simple
User=${user}
WorkingDirectory=${baseDir}
ExecStart=/usr/bin/python3 ${baseDir}/main.py
Restart=always
RestartSec=5
Environment=SYNTHCLAW_BASE_DIR=${baseDir}

[Install]
WantedBy=multi-user.target
`;
}
