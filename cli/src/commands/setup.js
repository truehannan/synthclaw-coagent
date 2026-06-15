import chalk from "chalk";
import ora from "ora";
import inquirer from "inquirer";
import { writeFileSync, existsSync } from "fs";
import { join } from "path";
import { execSync } from "child_process";
import { randomBytes } from "crypto";
import { config, generateEnvContent, getProjectRoot, printSuccess, printError, printInfo } from "../utils.js";

function isConfigured() {
  return !!(config.get("telegram_token") || config.get("openai_api_key"));
}

export async function runSetup() {
  const editMode = isConfigured();

  console.log(
    chalk.bold(editMode ? "  Edit Configuration" : "  Setup Wizard") +
      chalk.dim(editMode ? " — modify existing settings\n" : " — configure your SynthClaw agent\n")
  );

  if (editMode) {
    printInfo("Existing config detected. Current values shown as defaults — press Enter to keep.\n");
  }

  // Step 1: Storage mode
  console.log(chalk.hex("#e85d04")("━".repeat(50)));
  console.log(chalk.bold("  1. STORAGE"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));

  const { storageMode } = await inquirer.prompt([
    {
      type: "list",
      name: "storageMode",
      message: "Where to store agent data?",
      choices: [
        { name: "Local SQLite (default, on this machine)", value: "local" },
        { name: "Cloudflare D1 + R2 (cloud, synced across devices)", value: "cloudflare" },
      ],
      default: config.get("storage_mode"),
    },
  ]);
  config.set("storage_mode", storageMode);

  if (storageMode === "cloudflare") {
    console.log(chalk.dim("  Cloudflare D1 for database, R2 for file storage (optional)\n"));

    const cfAnswers = await inquirer.prompt([
      {
        type: "input",
        name: "cfAccountId",
        message: "Cloudflare Account ID:",
        validate: (v) => (v.length > 10 ? true : "Account ID seems too short"),
        default: config.get("cf_account_id") || undefined,
      },
      {
        type: "password",
        name: "cfApiToken",
        message: "Cloudflare API Token (needs D1 + R2 permissions):",
        mask: "*",
        validate: (v) => (v.length > 10 ? true : "Token seems too short"),
        default: config.get("cf_api_token") || undefined,
      },
      {
        type: "input",
        name: "cfD1DatabaseId",
        message: "D1 Database ID (create at dash.cloudflare.com/d1):",
        validate: (v) => (v.length > 10 ? true : "ID seems too short"),
        default: config.get("cf_d1_database_id") || undefined,
      },
      {
        type: "input",
        name: "cfR2Bucket",
        message: "R2 Bucket name (optional, for file storage):",
        default: config.get("cf_r2_bucket") || "",
      },
    ]);

    config.set("cf_account_id", cfAnswers.cfAccountId);
    config.set("cf_api_token", cfAnswers.cfApiToken);
    config.set("cf_d1_database_id", cfAnswers.cfD1DatabaseId);
    config.set("cf_r2_bucket", cfAnswers.cfR2Bucket || "");

    // Test connection
    const spinnerCf = ora("Testing Cloudflare D1 connection...").start();
    try {
      const resp = execSync(
        `curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer ${cfAnswers.cfApiToken}" "https://api.cloudflare.com/client/v4/accounts/${cfAnswers.cfAccountId}/d1/database/${cfAnswers.cfD1DatabaseId}"`,
        { encoding: "utf-8", timeout: 10000 }
      ).trim();
      if (resp === "200") {
        spinnerCf.succeed("Cloudflare D1 connected");
      } else {
        spinnerCf.warn(`Cloudflare returned HTTP ${resp} — check credentials`);
        const { fallback } = await inquirer.prompt([
          { type: "confirm", name: "fallback", message: "Switch to local SQLite instead?", default: true },
        ]);
        if (fallback) {
          config.set("storage_mode", "local");
        }
      }
    } catch (err) {
      spinnerCf.warn("Could not reach Cloudflare — using local SQLite as fallback");
      config.set("storage_mode", "local");
    }
  }

  // Step 2: Interface mode
  console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
  console.log(chalk.bold("  2. INTERFACE MODE"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));

  const { interfaceMode } = await inquirer.prompt([
    {
      type: "list",
      name: "interfaceMode",
      message: "Which messaging interface(s)?",
      choices: [
        { name: "Telegram (polling)", value: "telegram" },
        { name: "WhatsApp (webhooks via Meta Cloud API)", value: "whatsapp" },
        { name: "Both (Telegram + WhatsApp)", value: "both" },
      ],
      default: config.get("interface_mode"),
    },
  ]);
  config.set("interface_mode", interfaceMode);

  // Step 3: Telegram config
  if (interfaceMode === "telegram" || interfaceMode === "both") {
    console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
    console.log(chalk.bold("  3. TELEGRAM"));
    console.log(chalk.hex("#e85d04")("━".repeat(50)));

    const { telegramToken } = await inquirer.prompt([
      {
        type: "password",
        name: "telegramToken",
        message: editMode ? "Telegram Bot Token (Enter to keep current):" : "Telegram Bot Token:",
        mask: "*",
        validate: (v) => (v.length > 10 || (editMode && v === "") ? true : "Token seems too short"),
        default: editMode ? config.get("telegram_token") : undefined,
      },
    ]);
    if (telegramToken) config.set("telegram_token", telegramToken);
  }

  // Step 4: WhatsApp config
  if (interfaceMode === "whatsapp" || interfaceMode === "both") {
    console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
    console.log(chalk.bold("  4. WHATSAPP"));
    console.log(chalk.hex("#e85d04")("━".repeat(50)));

    const waAnswers = await inquirer.prompt([
      {
        type: "password",
        name: "whatsappToken",
        message: "WhatsApp API Access Token:",
        mask: "*",
        validate: (v) => (v.length > 10 || (editMode && v === "") ? true : "Token seems too short"),
        default: editMode ? config.get("whatsapp_token") : undefined,
      },
      {
        type: "input",
        name: "whatsappPhoneId",
        message: "WhatsApp Phone Number ID:",
        default: config.get("whatsapp_phone_number_id") || undefined,
      },
      {
        type: "input",
        name: "whatsappVerifyToken",
        message: "Webhook Verify Token:",
        default: config.get("whatsapp_verify_token") || randomBytes(16).toString("hex"),
      },
      {
        type: "input",
        name: "whatsappPort",
        message: "Webhook Port:",
        default: config.get("whatsapp_port") || "8443",
      },
    ]);
    if (waAnswers.whatsappToken) config.set("whatsapp_token", waAnswers.whatsappToken);
    config.set("whatsapp_phone_number_id", waAnswers.whatsappPhoneId || config.get("whatsapp_phone_number_id"));
    config.set("whatsapp_verify_token", waAnswers.whatsappVerifyToken);
    config.set("whatsapp_port", waAnswers.whatsappPort);
  }

  // Step 5: LLM Provider
  console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
  console.log(chalk.bold("  5. LLM PROVIDER"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));

  const llmAnswers = await inquirer.prompt([
    {
      type: "password",
      name: "apiKey",
      message: editMode ? "API Key (Enter to keep current):" : "API Key:",
      mask: "*",
      validate: (v) => (v.length > 5 || (editMode && v === "") ? true : "Key seems too short"),
      default: editMode ? config.get("openai_api_key") : undefined,
    },
    {
      type: "input",
      name: "apiBase",
      message: "API Base URL:",
      default: config.get("openai_api_base") || "https://inference.do-ai.run/v1",
    },
    {
      type: "list",
      name: "defaultModel",
      message: "Default Model:",
      choices: [
        "llama3.3-70b-instruct",
        "deepseek-r1-distill-llama-70b",
        "anthropic-claude-sonnet-4",
        "openai-gpt-4o",
        new inquirer.Separator("── Google ──"),
        "google:gemini-2.5-flash",
        "google:gemini-2.5-pro",
        new inquirer.Separator("── NVIDIA ──"),
        "nvidia:meta/llama-3.3-70b-instruct",
        "nvidia:deepseek-ai/deepseek-r1",
        new inquirer.Separator("── HuggingFace ──"),
        "hf:meta-llama/Llama-3.3-70B-Instruct",
        new inquirer.Separator(),
        { name: "Custom (type your own)", value: "__custom__" },
      ],
      default: config.get("default_model"),
    },
  ]);

  let model = llmAnswers.defaultModel;
  if (model === "__custom__") {
    const { customModel } = await inquirer.prompt([
      { type: "input", name: "customModel", message: "Custom model name:" },
    ]);
    model = customModel;
  }

  if (llmAnswers.apiKey) config.set("openai_api_key", llmAnswers.apiKey);
  config.set("openai_api_base", llmAnswers.apiBase);
  config.set("default_model", model);

  // Step 6: Rate limiting & server
  console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
  console.log(chalk.bold("  6. RATE LIMITS & SERVER"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));

  const serverAnswers = await inquirer.prompt([
    {
      type: "input",
      name: "maxRpm",
      message: "Max requests per minute (0 = unlimited):",
      default: config.get("max_rpm") || "0",
    },
    {
      type: "input",
      name: "maxIterations",
      message: "Max tool iterations per message:",
      default: config.get("max_tool_iterations") || "10",
    },
    {
      type: "input",
      name: "maxHistory",
      message: "Max conversation history messages:",
      default: config.get("max_history_messages") || "20",
    },
    {
      type: "input",
      name: "remoteHost",
      message: "Remote server IP (blank for local):",
      default: config.get("remote_host") || "",
    },
    {
      type: "input",
      name: "remoteUser",
      message: "SSH user:",
      default: config.get("remote_user") || "root",
      when: (a) => a.remoteHost !== "",
    },
  ]);

  config.set("max_rpm", serverAnswers.maxRpm);
  config.set("max_tool_iterations", serverAnswers.maxIterations);
  config.set("max_history_messages", serverAnswers.maxHistory);
  config.set("remote_host", serverAnswers.remoteHost || "");
  config.set("remote_user", serverAnswers.remoteUser || "root");

  // Step 7: Composio (optional)
  console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
  console.log(chalk.bold("  7. COMPOSIO (optional — 1000+ tool integrations)"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));
  console.log(chalk.dim("  Get your key at app.composio.dev. Skip if not using.\n"));

  const { composioKey } = await inquirer.prompt([
    {
      type: "input",
      name: "composioKey",
      message: "Composio API Key (Enter to skip):",
      default: config.get("composio_api_key") || "",
    },
  ]);
  if (composioKey) config.set("composio_api_key", composioKey);

  // Step 8: Write .env
  const spinner1 = ora("Writing .env configuration...").start();
  const root = getProjectRoot();
  try {
    writeFileSync(join(root, ".env"), generateEnvContent());
    spinner1.succeed(`Configuration saved`);
  } catch (err) {
    spinner1.warn("Could not write .env (will use stored config)");
  }

  // Step 9: Install Python deps
  console.log("");
  const { installDeps } = await inquirer.prompt([
    {
      type: "confirm",
      name: "installDeps",
      message: "Install/update Python dependencies?",
      default: !editMode,
    },
  ]);

  if (installDeps) {
    const venvPath = join(root, "venv");
    const requirementsPath = join(root, "requirements.txt");
    if (existsSync(requirementsPath)) {
      let pythonBin = "python3";
      try { execSync(`${pythonBin} --version`, { encoding: "utf-8" }); }
      catch { pythonBin = "python"; }

      if (!existsSync(venvPath)) {
        const sv = ora("Creating venv...").start();
        try {
          execSync(`${pythonBin} -m venv "${venvPath}"`, { encoding: "utf-8", cwd: root, timeout: 30000 });
          sv.succeed("Venv created");
        } catch {
          sv.fail("Venv creation failed — try: apt install python3.12-venv");
        }
      }
      if (existsSync(venvPath)) {
        const sp = ora("Installing dependencies...").start();
        try {
          const pip = join(venvPath, "bin", "pip");
          execSync(`"${pip}" install --upgrade pip -q && "${pip}" install -r "${requirementsPath}" -q`, {
            encoding: "utf-8", timeout: 180000, cwd: root,
          });
          sp.succeed("Dependencies installed");
        } catch (err) {
          sp.fail("pip install failed");
        }
      }
    }
  }

  // Summary
  console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
  console.log(chalk.bold(editMode ? "  ✓ CONFIG UPDATED" : "  ✓ SETUP COMPLETE"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));
  console.log("");
  printSuccess(`Interface: ${config.get("interface_mode")}`);
  printSuccess(`Model: ${config.get("default_model")}`);
  printSuccess(`Storage: ${config.get("storage_mode")}`);
  if (config.get("max_rpm") !== "0") printSuccess(`RPM limit: ${config.get("max_rpm")}`);
  if (config.get("composio_api_key")) printSuccess("Composio: configured");
  console.log("");
  printInfo("Run: synthclaw start");
  console.log("");
}
