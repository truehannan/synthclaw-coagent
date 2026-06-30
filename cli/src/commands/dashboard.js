import chalk from "chalk";
import { execSync } from "child_process";
import { createInterface } from "readline";
import { existsSync } from "fs";
import { join } from "path";
import inquirer from "inquirer";
import ora from "ora";
import { config, generateEnvContent, getProjectRoot, printSuccess, printError, printInfo } from "../utils.js";
import { SYNTHCLAW_BLOCK } from "../ascii.js";

// ── Color palette (TRUE RED theme, not orange) ──────────────────────────────
const RED = chalk.hex("#cc0000");
const BRIGHT_RED = chalk.hex("#ff1a1a");
const DIM_RED = chalk.hex("#660000");
const DARK_BG = chalk.bgHex("#0d0000");
const ACCENT = chalk.hex("#ff3333");
const DIM = chalk.dim;

// ── System metrics ──────────────────────────────────────────────────────────
function getSystemMetrics() {
  const m = { cpu: 0, mem: 0, memUsed: 0, memTotal: 0, disk: 0, diskUsed: "", diskTotal: "", uptime: "", ip: "", hostname: "", loadAvg: [0,0,0], procs: 0, netRx: "", netTx: "" };
  try {
    const la = execSync("cat /proc/loadavg 2>/dev/null", { encoding: "utf-8" }).trim().split(" ");
    const cores = parseInt(execSync("nproc 2>/dev/null", { encoding: "utf-8" }).trim()) || 1;
    m.cpu = Math.min(100, Math.round((parseFloat(la[0]) / cores) * 100));
    m.loadAvg = [parseFloat(la[0]), parseFloat(la[1]), parseFloat(la[2])];
  } catch {}
  try {
    const mi = execSync("free -m 2>/dev/null | awk 'NR==2{print $3,$2}'", { encoding: "utf-8" }).trim().split(" ");
    m.memUsed = parseInt(mi[0]) || 0; m.memTotal = parseInt(mi[1]) || 1;
    m.mem = Math.round((m.memUsed / m.memTotal) * 100);
  } catch {}
  try {
    const di = execSync("df -h / 2>/dev/null | awk 'NR==2{print $5,$3,$2}'", { encoding: "utf-8" }).trim().split(" ");
    m.disk = parseInt(di[0]) || 0; m.diskUsed = di[1] || "?"; m.diskTotal = di[2] || "?";
  } catch {}
  try { m.uptime = execSync("uptime -p 2>/dev/null", { encoding: "utf-8" }).trim().replace("up ", ""); } catch {}
  try { m.hostname = execSync("hostname 2>/dev/null", { encoding: "utf-8" }).trim(); } catch {}
  try { m.ip = execSync("hostname -I 2>/dev/null | awk '{print $1}'", { encoding: "utf-8" }).trim(); } catch {}
  try { m.procs = parseInt(execSync("ps aux 2>/dev/null | wc -l", { encoding: "utf-8" }).trim()) || 0; } catch {}
  return m;
}


// ── Gauge renderers (JARVIS-style circular/bar gauges) ──────────────────────

function circleGauge(percent, label, size = 5) {
  // Semicircle gauge using block chars
  const filled = Math.round((percent / 100) * size);
  const segs = [];
  for (let i = 0; i < size; i++) {
    segs.push(i < filled ? BRIGHT_RED("█") : DIM_RED("░"));
  }
  const pct = (percent + "%").padStart(4);
  return `${segs.join("")} ${ACCENT(pct)} ${DIM(label)}`;
}

function barGauge(percent, width = 16) {
  const filled = Math.round((percent / 100) * width);
  return BRIGHT_RED("█".repeat(filled)) + DIM_RED("░".repeat(width - filled));
}

function sparkline(values, width = 12) {
  // Mini sparkline using block chars
  const chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
  const max = Math.max(...values, 1);
  return values.slice(-width).map(v => {
    const idx = Math.min(7, Math.round((v / max) * 7));
    return RED(chars[idx]);
  }).join("");
}


// ── Fixed header panel ──────────────────────────────────────────────────────

// CPU history for sparkline
const cpuHistory = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

function renderHeaderPanel() {
  const m = getSystemMetrics();
  cpuHistory.push(m.cpu);
  if (cpuHistory.length > 12) cpuHistory.shift();

  const w = Math.min(process.stdout.columns || 70, 72);
  const border = RED("─".repeat(w - 4));
  const model = config.get("default_model") || "not set";
  const provider = (() => {
    const base = config.get("openai_api_base") || "";
    if (base.includes("do-ai")) return "DigitalOcean";
    if (base.includes("openai.com")) return "OpenAI";
    if (base.includes("openrouter")) return "OpenRouter";
    if (base.includes("nvidia")) return "NVIDIA";
    if (base.includes("huggingface")) return "HuggingFace";
    if (base.includes("googleapis")) return "Google";
    if (base.includes("cloudflare")) return "Cloudflare";
    if (base.includes("localhost")) return "Ollama";
    return "Custom";
  })();
  const iface = config.get("interface_mode") || "cli";
  const storage = config.get("storage_mode") || "local";
  const configured = !!(config.get("openai_api_key"));

  const lines = [];
  lines.push("  " + border);

  // SYNTHCLAW block letters in bright red
  for (const row of SYNTHCLAW_BLOCK) {
    lines.push("  " + BRIGHT_RED(row));
  }

  lines.push("  " + border);

  // Status row 1: Model + Provider + Interface
  const statusLine1 = [
    DIM("MODEL ") + ACCENT(model.length > 28 ? model.slice(0, 25) + "…" : model),
    DIM("VIA ") + ACCENT(provider),
    DIM("CH ") + ACCENT(iface.toUpperCase()),
  ].join("  ");
  lines.push("  " + statusLine1);

  // Status row 2: Storage + Config status + Uptime
  const statusLine2 = [
    DIM("STORE ") + ACCENT(storage === "cloudflare" ? "D1" : "SQLite"),
    configured ? ACCENT("● READY") : RED("○ UNCONFIGURED"),
    DIM("UP ") + ACCENT(m.uptime || "?"),
  ].join("  ");
  lines.push("  " + statusLine2);

  lines.push("  " + border);

  // System gauges row — JARVIS style
  const cpuBar = barGauge(m.cpu, 10);
  const memBar = barGauge(m.mem, 10);
  const diskBar = barGauge(m.disk, 10);
  const cpuSpark = sparkline(cpuHistory);

  lines.push(
    "  " + DIM("CPU ") + cpuBar + " " + ACCENT((m.cpu + "%").padStart(4)) +
    "  " + DIM("MEM ") + memBar + " " + ACCENT((m.mem + "%").padStart(4)) +
    "  " + DIM("DSK ") + diskBar + " " + ACCENT((m.disk + "%").padStart(4))
  );

  // Sparkline + network info
  lines.push(
    "  " + DIM("LOAD ") + cpuSpark +
    "  " + DIM("HOST ") + ACCENT(m.hostname || "localhost") +
    "  " + DIM("IP ") + ACCENT(m.ip || "127.0.0.1")
  );

  lines.push("  " + border);
  return lines.join("\n");
}


// ── Provider configs (which need more than just API key) ────────────────────

const PROVIDERS = {
  "DigitalOcean": { base: "https://inference.do-ai.run/v1", fields: ["api_key"] },
  "OpenAI": { base: "https://api.openai.com/v1", fields: ["api_key"] },
  "Anthropic (via DO)": { base: "https://inference.do-ai.run/v1", fields: ["api_key"] },
  "Google Gemini": { base: "https://generativelanguage.googleapis.com/v1beta/openai", fields: ["api_key"] },
  "NVIDIA NIM": { base: "https://integrate.api.nvidia.com/v1", fields: ["api_key"] },
  "HuggingFace": { base: "https://router.huggingface.co/v1", fields: ["api_key"] },
  "OpenRouter": { base: "https://openrouter.ai/api/v1", fields: ["api_key"] },
  "GitHub Models": { base: "https://models.inference.ai.azure.com", fields: ["api_key"] },
  "Cloudflare Workers AI": { base: "", fields: ["account_id", "api_key"], buildBase: (cfg) => `https://api.cloudflare.com/client/v4/accounts/${cfg.account_id}/ai/v1` },
  "Azure OpenAI": { base: "", fields: ["endpoint_url", "deployment", "api_key"], buildBase: (cfg) => `${cfg.endpoint_url}/openai/deployments/${cfg.deployment}` },
  "Ollama (local)": { base: "http://localhost:11434/v1", fields: [] },
  "Custom URL": { base: "", fields: ["base_url", "api_key"] },
};

// ── Inline wizard (runs inside the TUI) ─────────────────────────────────────

async function runInlineWizard() {
  console.log("");
  console.log(RED("  ┌─ SETUP WIZARD"));
  console.log(RED("  │"));

  // Step 1: Storage
  console.log(RED("  │ ") + chalk.bold("Storage"));
  const { storageMode } = await inquirer.prompt([{
    type: "list", name: "storageMode", message: "Where to store data?",
    choices: [
      { name: "Local SQLite (default)", value: "local" },
      { name: "Cloudflare D1 (cloud sync)", value: "cloudflare" },
    ],
    default: config.get("storage_mode") || "local",
    prefix: RED("  │"),
  }]);
  config.set("storage_mode", storageMode);

  if (storageMode === "cloudflare") {
    const cfAnswers = await inquirer.prompt([
      { type: "input", name: "cfAccountId", message: "Cloudflare Account ID:", default: config.get("cf_account_id") || undefined, prefix: RED("  │") },
      { type: "password", name: "cfApiToken", message: "Cloudflare API Token:", mask: "*", default: config.get("cf_api_token") || undefined, prefix: RED("  │") },
      { type: "input", name: "cfD1DatabaseId", message: "D1 Database ID:", default: config.get("cf_d1_database_id") || undefined, prefix: RED("  │") },
    ]);
    config.set("cf_account_id", cfAnswers.cfAccountId);
    config.set("cf_api_token", cfAnswers.cfApiToken);
    config.set("cf_d1_database_id", cfAnswers.cfD1DatabaseId);
  }

  // Step 2: Interface (with CLI-only option)
  console.log(RED("  │"));
  console.log(RED("  │ ") + chalk.bold("Interface"));
  const { interfaceMode } = await inquirer.prompt([{
    type: "list", name: "interfaceMode", message: "How will you interact?",
    choices: [
      { name: "CLI only (no messaging platform)", value: "cli" },
      { name: "Telegram", value: "telegram" },
      { name: "WhatsApp", value: "whatsapp" },
      { name: "Telegram + WhatsApp", value: "both" },
    ],
    default: config.get("interface_mode") || "cli",
    prefix: RED("  │"),
  }]);
  config.set("interface_mode", interfaceMode);

  // Step 3: Telegram (if needed)
  if (interfaceMode === "telegram" || interfaceMode === "both") {
    const { telegramToken } = await inquirer.prompt([{
      type: "password", name: "telegramToken", mask: "*",
      message: "Telegram Bot Token (from @BotFather):",
      default: config.get("telegram_token") ? undefined : undefined,
      prefix: RED("  │"),
      validate: (v) => (v.length > 10 || (config.get("telegram_token") && v === "") ? true : "Too short"),
    }]);
    if (telegramToken) config.set("telegram_token", telegramToken);
  }

  // Step 4: WhatsApp (if needed)
  if (interfaceMode === "whatsapp" || interfaceMode === "both") {
    const waAnswers = await inquirer.prompt([
      { type: "password", name: "whatsappToken", message: "WhatsApp Access Token:", mask: "*", prefix: RED("  │") },
      { type: "input", name: "whatsappPhoneId", message: "Phone Number ID:", default: config.get("whatsapp_phone_number_id") || undefined, prefix: RED("  │") },
    ]);
    if (waAnswers.whatsappToken) config.set("whatsapp_token", waAnswers.whatsappToken);
    if (waAnswers.whatsappPhoneId) config.set("whatsapp_phone_number_id", waAnswers.whatsappPhoneId);
  }

  // Step 5: AI Provider
  console.log(RED("  │"));
  console.log(RED("  │ ") + chalk.bold("AI Provider"));
  const providerNames = Object.keys(PROVIDERS);
  const { provider } = await inquirer.prompt([{
    type: "list", name: "provider", message: "Select LLM provider:",
    choices: providerNames,
    default: (() => {
      const base = config.get("openai_api_base") || "";
      for (const [name, p] of Object.entries(PROVIDERS)) {
        if (p.base && base.includes(p.base.split("//")[1]?.split("/")[0] || "___")) return name;
      }
      return "DigitalOcean";
    })(),
    prefix: RED("  │"),
  }]);

  const providerCfg = PROVIDERS[provider];
  const providerFields = {};

  // Ask for each required field
  for (const field of providerCfg.fields) {
    if (field === "api_key") {
      const { val } = await inquirer.prompt([{
        type: "password", name: "val", mask: "*",
        message: `${provider} API Key:`,
        prefix: RED("  │"),
        validate: (v) => (v.length > 5 || (config.get("openai_api_key") && v === "") ? true : "Too short"),
      }]);
      if (val) { config.set("openai_api_key", val); providerFields.api_key = val; }
    } else if (field === "account_id") {
      const { val } = await inquirer.prompt([{
        type: "input", name: "val", message: `${provider} Account ID:`,
        default: config.get("cf_account_id") || undefined,
        prefix: RED("  │"),
      }]);
      providerFields.account_id = val;
      config.set("cf_account_id", val);
    } else if (field === "endpoint_url") {
      const { val } = await inquirer.prompt([{
        type: "input", name: "val", message: "Endpoint URL:",
        default: config.get("openai_api_base") || "https://",
        prefix: RED("  │"),
      }]);
      providerFields.endpoint_url = val;
    } else if (field === "deployment") {
      const { val } = await inquirer.prompt([{
        type: "input", name: "val", message: "Deployment name:",
        prefix: RED("  │"),
      }]);
      providerFields.deployment = val;
    } else if (field === "base_url") {
      const { val } = await inquirer.prompt([{
        type: "input", name: "val", message: "API Base URL:",
        default: config.get("openai_api_base") || "https://",
        prefix: RED("  │"),
      }]);
      providerFields.base_url = val;
    }
  }

  // Set base URL
  let apiBase = providerCfg.base;
  if (providerCfg.buildBase) {
    apiBase = providerCfg.buildBase(providerFields);
  } else if (providerFields.base_url) {
    apiBase = providerFields.base_url;
  }
  if (apiBase) config.set("openai_api_base", apiBase);

  // Model selection — try live fetch
  console.log(RED("  │"));
  const effectiveKey = providerFields.api_key || config.get("openai_api_key");
  let modelChoices = ["llama3.3-70b-instruct", "Custom"];

  if (effectiveKey && apiBase) {
    try {
      const headers = { "Authorization": `Bearer ${effectiveKey}`, "Content-Type": "application/json" };
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 8000);
      const resp = await fetch(`${apiBase}/models`, { headers, signal: controller.signal });
      clearTimeout(timeout);
      if (resp.ok) {
        const data = await resp.json();
        const items = data.data || data.models || [];
        const live = items.map(i => typeof i === "string" ? i : (i.id || i.name || "")).filter(Boolean).slice(0, 20);
        if (live.length > 0) modelChoices = [...live, "Custom"];
      }
    } catch {}
  }

  const { defaultModel } = await inquirer.prompt([{
    type: "list", name: "defaultModel", message: "Default model:",
    choices: modelChoices,
    default: config.get("default_model"),
    prefix: RED("  │"),
  }]);

  if (defaultModel === "Custom") {
    const { cm } = await inquirer.prompt([{ type: "input", name: "cm", message: "Model name:", prefix: RED("  │") }]);
    config.set("default_model", cm);
  } else {
    config.set("default_model", defaultModel);
  }

  // Save .env
  console.log(RED("  │"));
  const root = getProjectRoot();
  try {
    const { writeFileSync } = await import("fs");
    writeFileSync(join(root, ".env"), generateEnvContent());
    console.log(RED("  │ ") + chalk.green("✓") + " Configuration saved");
  } catch { console.log(RED("  │ ") + chalk.yellow("⚠") + " Could not write .env"); }

  console.log(RED("  │"));
  console.log(RED("  └─ ") + chalk.bold("SETUP COMPLETE"));
  console.log("");
}


// ── Chat / LLM interaction ──────────────────────────────────────────────────

const chatHistory = [];

function printReply(text) {
  const lines = text.split("\n");
  for (const line of lines) {
    console.log(RED("  │ ") + line);
  }
}

async function sendMessage(message) {
  const apiKey = config.get("openai_api_key");
  const apiBase = config.get("openai_api_base");
  const model = config.get("default_model");

  if (!apiKey) {
    console.log(RED("  │ ") + chalk.yellow("Not configured. Running /setup..."));
    await runInlineWizard();
    return;
  }

  chatHistory.push({ role: "user", content: message });
  process.stdout.write(DIM("  ⠋ thinking...\r"));

  try {
    const msgs = [
      { role: "system", content: "You are SynthClaw, a personal AI assistant. Be concise. Plain text only." },
      ...chatHistory.slice(-10),
    ];

    const resp = await fetch(`${apiBase}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ model, messages: msgs, temperature: 0.7, max_tokens: 2048 }),
    });

    const data = await resp.json();
    process.stdout.write("               \r");

    if (!resp.ok) {
      console.log(RED("  ✗ ") + (data.error?.message || `HTTP ${resp.status}`));
      return;
    }

    const reply = (data.choices?.[0]?.message?.content || "").replace(/<think>[\s\S]*?<\/think>/g, "").trim();
    chatHistory.push({ role: "assistant", content: reply });
    console.log("");
    console.log(RED("  ┌─ SYNTHCLAW"));
    printReply(reply || "(empty)");
    console.log(RED("  └─"));
    console.log("");
  } catch (err) {
    process.stdout.write("               \r");
    console.log(RED("  ✗ ") + err.message);
  }
}

// ── Command handling ────────────────────────────────────────────────────────

const COMMANDS = [
  "/setup", "/status", "/model", "/models", "/clear", "/help",
  "/run", "/providers", "/skills", "/memory", "/quit",
];

function autocomplete(line) {
  if (!line.startsWith("/")) return [[], line];
  return [COMMANDS.filter(c => c.startsWith(line)), line];
}

async function handleCommand(input) {
  const [cmd, ...args] = input.split(" ");
  const arg = args.join(" ");

  switch (cmd) {
    case "/setup":
      await runInlineWizard();
      break;
    case "/status":
      console.clear();
      console.log(renderHeaderPanel());
      break;
    case "/model":
      if (arg) {
        config.set("default_model", arg);
        console.log(DIM("  Model → ") + ACCENT(arg));
      } else {
        console.log(DIM("  Current model: ") + ACCENT(config.get("default_model") || "none"));
      }
      break;
    case "/clear":
      chatHistory.length = 0;
      console.clear();
      console.log(renderHeaderPanel());
      console.log(DIM("  Chat cleared."));
      break;
    case "/run":
      if (!arg) { console.log(DIM("  Usage: /run <command>")); break; }
      try {
        const out = execSync(arg, { encoding: "utf-8", timeout: 30000, cwd: getProjectRoot() });
        console.log(DIM_RED("  ┌─ output"));
        for (const line of out.trim().split("\n").slice(0, 25)) {
          console.log(DIM_RED("  │ ") + line);
        }
        console.log(DIM_RED("  └─"));
      } catch (err) {
        console.log(RED("  ✗ ") + (err.stderr || err.message || "").slice(0, 200));
      }
      break;
    case "/help":
      console.log("");
      console.log(RED("  Commands:"));
      console.log(DIM("    /setup     ") + "Run setup wizard");
      console.log(DIM("    /status    ") + "Refresh system panel");
      console.log(DIM("    /model     ") + "Show/switch model");
      console.log(DIM("    /run <cmd> ") + "Execute shell command");
      console.log(DIM("    /clear     ") + "Clear chat history");
      console.log(DIM("    /quit      ") + "Exit");
      console.log(DIM("    <text>     ") + "Chat with AI");
      console.log("");
      break;
    case "/quit":
    case "/exit":
      process.exit(0);
      break;
    default:
      await sendMessage(input);
  }
}


// ── Main entry point ────────────────────────────────────────────────────────

export async function runDashboard() {
  console.clear();
  console.log(renderHeaderPanel());

  // If not configured, immediately run wizard
  const hasKey = config.get("openai_api_key");
  if (!hasKey) {
    console.log(DIM("  First run detected. Starting setup...\n"));
    await runInlineWizard();
    // Re-render header with new config
    console.clear();
    console.log(renderHeaderPanel());
  }

  console.log(DIM("  Type a message to chat, or /help for commands.\n"));

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: RED("  ▸ "),
    completer: autocomplete,
  });

  rl.prompt();

  rl.on("line", async (line) => {
    const input = line.trim();
    if (!input) { rl.prompt(); return; }

    if (input.startsWith("/")) {
      await handleCommand(input);
    } else {
      await sendMessage(input);
    }
    rl.prompt();
  });

  rl.on("close", () => {
    console.log(DIM("\n  Session ended.\n"));
    process.exit(0);
  });

  // Refresh header metrics every 30s
  setInterval(() => {
    const saved = "\x1b7";
    const restored = "\x1b8";
    process.stdout.write(saved + "\x1b[H" + renderHeaderPanel() + "\n" + restored);
  }, 30000);
}

// Also export for use by `synthclaw start`
export { runInlineWizard };
