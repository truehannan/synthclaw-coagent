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

function masked(val) {
  if (!val) return "";
  if (val.length <= 8) return "••••";
  return val.slice(0, 4) + "••••" + val.slice(-4);
}

const PROVIDERS = {
  "DigitalOcean Gradient AI": { base: "https://inference.do-ai.run/v1", prefix: "sk-do-" },
  "OpenAI": { base: "https://api.openai.com/v1", prefix: "sk-" },
  "Anthropic (via DO)": { base: "https://inference.do-ai.run/v1", prefix: "" },
  "Google Gemini": { base: "https://generativelanguage.googleapis.com/v1beta/openai", prefix: "AIza" },
  "NVIDIA NIM": { base: "https://integrate.api.nvidia.com/v1", prefix: "nvapi-" },
  "HuggingFace": { base: "https://router.huggingface.co/v1", prefix: "hf_" },
  "OpenRouter": { base: "https://openrouter.ai/api/v1", prefix: "sk-or-" },
  "GitHub Models": { base: "https://models.inference.ai.azure.com", prefix: "ghp_" },
  "Ollama (local)": { base: "http://localhost:11434/v1", prefix: "" },
  "Custom URL": { base: "", prefix: "" },
};

export async function runSetup() {
  const editMode = isConfigured();

  console.log(
    chalk.bold(editMode ? "  Edit Configuration" : "  Setup Wizard") +
      chalk.dim(editMode ? " — modify existing settings\n" : " — configure your SynthClaw agent\n")
  );

  if (editMode) {
    printInfo("Current values shown. Press Enter to keep unchanged.\n");
  }

  // Step 1: Storage
  console.log(chalk.hex("#e85d04")("━".repeat(50)));
  console.log(chalk.bold("  1. STORAGE"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));

  const { storageMode } = await inquirer.prompt([{
    type: "list", name: "storageMode", message: "Where to store data?",
    choices: [
      { name: "Local SQLite (default)", value: "local" },
      { name: "Cloudflare D1 + R2 (cloud sync)", value: "cloudflare" },
    ],
    default: config.get("storage_mode"),
  }]);
  config.set("storage_mode", storageMode);

  if (storageMode === "cloudflare") {
    const cfAnswers = await inquirer.prompt([
      { type: "input", name: "cfAccountId", message: "Cloudflare Account ID:", default: config.get("cf_account_id") || undefined },
      { type: "password", name: "cfApiToken", message: "Cloudflare API Token:", mask: "*", default: config.get("cf_api_token") || undefined },
      { type: "input", name: "cfD1DatabaseId", message: "D1 Database ID:", default: config.get("cf_d1_database_id") || undefined },
      { type: "input", name: "cfR2Bucket", message: "R2 Bucket name (optional):", default: config.get("cf_r2_bucket") || "" },
    ]);
    config.set("cf_account_id", cfAnswers.cfAccountId);
    config.set("cf_api_token", cfAnswers.cfApiToken);
    config.set("cf_d1_database_id", cfAnswers.cfD1DatabaseId);
    config.set("cf_r2_bucket", cfAnswers.cfR2Bucket || "");

    const spinnerCf = ora("Testing Cloudflare connection...").start();
    try {
      const resp = execSync(
        `curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer ${cfAnswers.cfApiToken}" "https://api.cloudflare.com/client/v4/accounts/${cfAnswers.cfAccountId}/d1/database/${cfAnswers.cfD1DatabaseId}"`,
        { encoding: "utf-8", timeout: 10000 }
      ).trim();
      if (resp === "200") {
        spinnerCf.succeed("Connected");

        // ── DB-first flow: Load existing config from D1 ──
        const syncSpinner = ora("Loading config from D1...").start();
        try {
          const queryResp = execSync(
            `curl -s -X POST -H "Authorization: Bearer ${cfAnswers.cfApiToken}" -H "Content-Type: application/json" ` +
            `-d '{"sql":"SELECT key, value FROM config"}' ` +
            `"https://api.cloudflare.com/client/v4/accounts/${cfAnswers.cfAccountId}/d1/database/${cfAnswers.cfD1DatabaseId}/query"`,
            { encoding: "utf-8", timeout: 15000 }
          );
          const d1Data = JSON.parse(queryResp);
          if (d1Data.success && d1Data.result && d1Data.result[0] && d1Data.result[0].results) {
            const rows = d1Data.result[0].results;
            let synced = 0;
            // Map D1 config keys to local config keys
            const keyMap = {
              "interface_mode": "interface_mode",
              "telegram_token": "telegram_token",
              "whatsapp_token": "whatsapp_token",
              "whatsapp_phone_number_id": "whatsapp_phone_number_id",
              "whatsapp_verify_token": "whatsapp_verify_token",
              "whatsapp_port": "whatsapp_port",
              "openai_api_key": "openai_api_key",
              "openai_api_base": "openai_api_base",
              "default_model": "default_model",
              "current_model": "default_model",
              "max_rpm": "max_rpm",
              "max_tool_iterations": "max_tool_iterations",
              "composio_api_key": "composio_api_key",
              "owner_telegram_id": "owner_telegram_id",
            };
            for (const row of rows) {
              const localKey = keyMap[row.key];
              if (localKey && row.value && row.value !== "null") {
                // Only fill if local is empty
                const current = config.get(localKey);
                if (!current) {
                  config.set(localKey, row.value);
                  synced++;
                }
              }
            }
            if (synced > 0) {
              syncSpinner.succeed(`Loaded ${synced} settings from D1`);
              printInfo("Wizard will show stored values. Press Enter to keep them.\n");
            } else {
              syncSpinner.info("No stored config in D1 yet (fresh database)");
            }
          } else {
            syncSpinner.info("D1 config table empty or not initialized yet");
          }
        } catch (syncErr) {
          syncSpinner.info("Could not load config from D1 (will proceed with local values)");
        }
      }
      else {
        spinnerCf.warn(`HTTP ${resp} — check credentials`);
        const { fb } = await inquirer.prompt([{ type: "confirm", name: "fb", message: "Use local SQLite instead?", default: true }]);
        if (fb) config.set("storage_mode", "local");
      }
    } catch { spinnerCf.warn("Connection failed — using local SQLite"); config.set("storage_mode", "local"); }
  }

  // Step 2: Interface
  console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
  console.log(chalk.bold("  2. INTERFACE"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));

  const { interfaceMode } = await inquirer.prompt([{
    type: "list", name: "interfaceMode", message: "Messaging platform?",
    choices: [
      { name: "Telegram", value: "telegram" },
      { name: "WhatsApp", value: "whatsapp" },
      { name: "Both", value: "both" },
    ],
    default: config.get("interface_mode"),
  }]);
  config.set("interface_mode", interfaceMode);

  // Step 3: Telegram
  if (interfaceMode === "telegram" || interfaceMode === "both") {
    console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
    console.log(chalk.bold("  3. TELEGRAM"));
    console.log(chalk.hex("#e85d04")("━".repeat(50)));
    if (editMode && config.get("telegram_token")) {
      printInfo(`Current: ${masked(config.get("telegram_token"))}`);
    }
    const { telegramToken } = await inquirer.prompt([{
      type: "password", name: "telegramToken", mask: "*",
      message: editMode ? "Bot Token (Enter to keep):" : "Bot Token (from @BotFather):",
      validate: (v) => (v.length > 10 || (editMode && v === "") ? true : "Too short"),
    }]);
    if (telegramToken) config.set("telegram_token", telegramToken);
  }

  // Step 4: WhatsApp
  if (interfaceMode === "whatsapp" || interfaceMode === "both") {
    console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
    console.log(chalk.bold("  4. WHATSAPP"));
    console.log(chalk.hex("#e85d04")("━".repeat(50)));
    if (editMode && config.get("whatsapp_token")) {
      printInfo(`Current token: ${masked(config.get("whatsapp_token"))}`);
    }
    const waAnswers = await inquirer.prompt([
      { type: "password", name: "whatsappToken", message: "Access Token:", mask: "*",
        validate: (v) => (v.length > 10 || (editMode && v === "") ? true : "Too short") },
      { type: "input", name: "whatsappPhoneId", message: "Phone Number ID:", default: config.get("whatsapp_phone_number_id") || undefined },
      { type: "input", name: "whatsappVerifyToken", message: "Verify Token:", default: config.get("whatsapp_verify_token") || randomBytes(16).toString("hex") },
      { type: "input", name: "whatsappPort", message: "Port:", default: config.get("whatsapp_port") || "8443" },
    ]);
    if (waAnswers.whatsappToken) config.set("whatsapp_token", waAnswers.whatsappToken);
    config.set("whatsapp_phone_number_id", waAnswers.whatsappPhoneId || config.get("whatsapp_phone_number_id"));
    config.set("whatsapp_verify_token", waAnswers.whatsappVerifyToken);
    config.set("whatsapp_port", waAnswers.whatsappPort);
  }

  // Step 5: AI Provider (simplified — select provider, then enter key)
  console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
  console.log(chalk.bold("  5. AI PROVIDER"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));

  const providerNames = Object.keys(PROVIDERS);
  const { provider } = await inquirer.prompt([{
    type: "list", name: "provider", message: "Select LLM provider:",
    choices: providerNames,
    default: (() => {
      const currentBase = config.get("openai_api_base") || "";
      for (const [name, p] of Object.entries(PROVIDERS)) {
        if (p.base && currentBase.includes(new URL(p.base).hostname)) return name;
      }
      return "DigitalOcean Gradient AI";
    })(),
  }]);

  const providerInfo = PROVIDERS[provider];
  let apiBase = providerInfo.base;

  if (provider === "Custom URL") {
    const { customBase } = await inquirer.prompt([{
      type: "input", name: "customBase", message: "API Base URL:",
      default: config.get("openai_api_base") || "https://",
    }]);
    apiBase = customBase;
  }

  if (editMode && config.get("openai_api_key")) {
    printInfo(`Current key: ${masked(config.get("openai_api_key"))}`);
  }

  const { apiKey } = await inquirer.prompt([{
    type: "password", name: "apiKey", mask: "*",
    message: editMode ? `${provider} API Key (Enter to keep):` : `${provider} API Key:`,
    validate: (v) => (v.length > 5 || (editMode && v === "") ? true : "Too short"),
  }]);

  if (apiKey) config.set("openai_api_key", apiKey);
  config.set("openai_api_base", apiBase);

  // Model selection — fetch live from provider if possible
  const defaultModels = {
    "DigitalOcean Gradient AI": ["llama3.3-70b-instruct", "deepseek-r1-distill-llama-70b", "anthropic-claude-sonnet-4", "openai-gpt-4o"],
    "OpenAI": ["openai-gpt-4o", "openai-gpt-4o-mini", "openai-gpt-4.1", "openai-o3-mini"],
    "Anthropic (via DO)": ["anthropic-claude-sonnet-4", "anthropic-claude-opus-4", "anthropic-claude-haiku-4.5"],
    "Google Gemini": ["google:gemini-2.5-flash", "google:gemini-2.5-pro", "google:gemini-2.0-flash"],
    "NVIDIA NIM": ["nvidia:meta/llama-3.3-70b-instruct", "nvidia:deepseek-ai/deepseek-r1", "nvidia:meta/llama-3.1-405b-instruct"],
    "HuggingFace": ["hf:meta-llama/Llama-3.3-70B-Instruct", "hf:deepseek-ai/DeepSeek-R1", "hf:Qwen/Qwen3-235B-A22B"],
    "OpenRouter": ["openrouter:anthropic/claude-3.5-sonnet", "openrouter:openai/gpt-4o", "openrouter:google/gemini-2.0-flash-001"],
    "GitHub Models": ["github:gpt-4o", "github:Meta-Llama-3.1-70B-Instruct"],
    "Ollama (local)": ["llama3.3", "mistral", "codellama"],
    "Custom URL": [],
  };

  // Try to fetch live models from the selected provider
  let liveModels = null;
  const effectiveKey = apiKey || config.get("openai_api_key");
  if (effectiveKey && providerInfo.base) {
    const fetchSpinner = ora("Fetching available models from provider...").start();
    try {
      const modelsUrl = `${providerInfo.base}/models`;
      const headers = { "Authorization": `Bearer ${effectiveKey}`, "Content-Type": "application/json" };
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 10000);
      const resp = await fetch(modelsUrl, { headers, signal: controller.signal });
      clearTimeout(timeout);
      if (resp.ok) {
        const data = await resp.json();
        const items = data.data || data.models || [];
        liveModels = items.map(item => typeof item === "string" ? item : (item.id || item.name || "")).filter(Boolean).slice(0, 30);
        if (liveModels.length > 0) {
          fetchSpinner.succeed(`Found ${liveModels.length} models from ${provider}`);
        } else {
          fetchSpinner.info("No models returned, using defaults");
          liveModels = null;
        }
      } else {
        fetchSpinner.info(`Could not fetch live models (HTTP ${resp.status}), using defaults`);
      }
    } catch (err) {
      fetchSpinner.info("Could not fetch live models, using defaults");
    }
  }

  const modelList = liveModels || defaultModels[provider] || [];
  const modelChoices = [...modelList.slice(0, 25), new inquirer.Separator(), { name: "Custom", value: "__custom__" }];
  const { defaultModel } = await inquirer.prompt([{
    type: "list", name: "defaultModel", message: "Default model:",
    choices: modelChoices,
    default: config.get("default_model"),
  }]);

  let model = defaultModel;
  if (model === "__custom__") {
    const { cm } = await inquirer.prompt([{ type: "input", name: "cm", message: "Model name:" }]);
    model = cm;
  }
  config.set("default_model", model);

  // Step 6: Limits
  console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
  console.log(chalk.bold("  6. LIMITS"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));

  const limitsAnswers = await inquirer.prompt([
    { type: "input", name: "maxRpm", message: "Max requests/minute (0 = unlimited):", default: config.get("max_rpm") || "0" },
    { type: "input", name: "maxIterations", message: "Max tool iterations/message:", default: config.get("max_tool_iterations") || "10" },
    { type: "input", name: "remoteHost", message: "Remote server IP (blank for local):", default: config.get("remote_host") || "" },
  ]);

  config.set("max_rpm", limitsAnswers.maxRpm);
  config.set("max_tool_iterations", limitsAnswers.maxIterations);
  config.set("remote_host", limitsAnswers.remoteHost || "");

  // Step 7: Composio
  console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
  console.log(chalk.bold("  7. COMPOSIO (optional)"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));
  if (editMode && config.get("composio_api_key")) {
    printInfo(`Current: ${masked(config.get("composio_api_key"))}`);
  }
  const { composioKey } = await inquirer.prompt([{
    type: "input", name: "composioKey", message: "Composio API Key (Enter to skip):",
    default: config.get("composio_api_key") || "",
  }]);
  if (composioKey) config.set("composio_api_key", composioKey);

  // Write .env
  const spinner1 = ora("Saving configuration...").start();
  const root = getProjectRoot();
  try {
    writeFileSync(join(root, ".env"), generateEnvContent());
    spinner1.succeed("Configuration saved");
  } catch { spinner1.warn("Could not write .env"); }

  // Sync config to D1 if cloudflare mode
  if (config.get("storage_mode") === "cloudflare" && config.get("cf_api_token")) {
    const d1Spinner = ora("Syncing config to D1...").start();
    try {
      const cfToken = config.get("cf_api_token");
      const cfAccount = config.get("cf_account_id");
      const cfDb = config.get("cf_d1_database_id");
      // Push key config values to D1
      const configPairs = {
        interface_mode: config.get("interface_mode"),
        telegram_token: config.get("telegram_token"),
        openai_api_key: config.get("openai_api_key"),
        openai_api_base: config.get("openai_api_base"),
        default_model: config.get("default_model"),
        max_rpm: config.get("max_rpm"),
        max_tool_iterations: config.get("max_tool_iterations"),
        composio_api_key: config.get("composio_api_key"),
      };
      // Build batch SQL statements
      const statements = Object.entries(configPairs)
        .filter(([_, v]) => v)
        .map(([k, v]) => `INSERT OR REPLACE INTO config (key, value) VALUES ('${k}', '${(v || "").replace(/'/g, "''")}');`)
        .join(" ");
      if (statements) {
        execSync(
          `curl -s -X POST -H "Authorization: Bearer ${cfToken}" -H "Content-Type: application/json" ` +
          `-d '{"sql":"${statements}"}' ` +
          `"https://api.cloudflare.com/client/v4/accounts/${cfAccount}/d1/database/${cfDb}/query"`,
          { encoding: "utf-8", timeout: 15000 }
        );
        d1Spinner.succeed("Config synced to D1");
      } else {
        d1Spinner.info("No config to sync");
      }
    } catch (err) {
      d1Spinner.warn("D1 sync failed (config saved locally only)");
    }
  }

  // Install deps
  console.log("");
  const { installDeps } = await inquirer.prompt([{
    type: "confirm", name: "installDeps",
    message: "Install/update Python dependencies?",
    default: !editMode,
  }]);

  if (installDeps) {
    const venvPath = join(root, "venv");
    const reqPath = join(root, "requirements.txt");
    if (!existsSync(reqPath)) {
      printError("requirements.txt not found");
    } else {
      let py = "python3";
      try { execSync(`${py} --version`, { encoding: "utf-8" }); }
      catch { py = "python"; }

      if (!existsSync(venvPath)) {
        const sv = ora("Creating venv...").start();
        try {
          execSync(`${py} -m venv "${venvPath}"`, { encoding: "utf-8", cwd: root, timeout: 60000 });
          sv.succeed("Venv created");
        } catch (err) {
          sv.fail("Venv failed");
          // Try installing python3-venv
          try {
            const ver = execSync(`${py} --version`, { encoding: "utf-8" }).match(/(\d+\.\d+)/)?.[1] || "3";
            execSync(`apt-get install -y python${ver}-venv 2>/dev/null || true`, { encoding: "utf-8", timeout: 60000 });
            execSync(`${py} -m venv "${venvPath}"`, { encoding: "utf-8", cwd: root, timeout: 30000 });
            printSuccess("Venv created (after installing python-venv package)");
          } catch { printError("Could not create venv. Install python3-venv manually."); }
        }
      }

      if (existsSync(venvPath)) {
        const sp = ora("Installing Python dependencies...").start();
        try {
          const pip = join(venvPath, "bin", "pip");
          execSync(`${pip} install --upgrade pip -q`, { encoding: "utf-8", timeout: 60000, cwd: root });
          execSync(`${pip} install -r ${reqPath} -q`, { encoding: "utf-8", timeout: 180000, cwd: root });
          sp.succeed("Dependencies installed");
        } catch (err) {
          sp.fail("pip install failed: " + (err.stderr || err.message || "").slice(0, 200));
          printInfo("Try manually: venv/bin/pip install -r requirements.txt");
        }
      }
    }
  }

  // Summary
  console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
  console.log(chalk.bold(editMode ? "  ✓ UPDATED" : "  ✓ SETUP COMPLETE"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));
  console.log("");
  printSuccess(`Provider: ${provider}`);
  printSuccess(`Model: ${config.get("default_model")}`);
  printSuccess(`Interface: ${config.get("interface_mode")}`);
  printSuccess(`Storage: ${config.get("storage_mode")}`);
  if (config.get("max_rpm") !== "0") printSuccess(`RPM: ${config.get("max_rpm")}/min`);
  console.log("");
  printInfo("Run: synthclaw start");
  console.log("");
}
