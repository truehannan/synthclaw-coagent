import chalk from "chalk";
import { execSync } from "child_process";
import { createInterface } from "readline";
import { existsSync, writeFileSync } from "fs";
import { join } from "path";
import inquirer from "inquirer";
import ora from "ora";
import { config, generateEnvContent, getProjectRoot, printSuccess, printError, printInfo } from "../utils.js";
import { SYNTHCLAW_BLOCK } from "../ascii.js";

// ══════════════════════════════════════════════════════════════════════════════
//  COLOR SYSTEM — Deep red on black, no orange, no rainbow
// ══════════════════════════════════════════════════════════════════════════════
const R = chalk.hex("#cc0000");        // primary red
const RB = chalk.hex("#ff1a1a");       // bright red (highlights)
const RD = chalk.hex("#4d0000");       // dark red (borders, inactive)
const RA = chalk.hex("#ff3333");       // accent (values, numbers)
const D = chalk.dim;                   // dim text (labels)
const W = chalk.white;                 // white (user text)
const G = chalk.hex("#33ff33");        // green (success only)
const Y = chalk.hex("#ffaa00");        // amber (warnings only)

// Box drawing chars
const BOX = { tl: "╭", tr: "╮", bl: "╰", br: "╯", h: "─", v: "│", vr: "├", vl: "┤" };

// ══════════════════════════════════════════════════════════════════════════════
//  SYSTEM METRICS
// ══════════════════════════════════════════════════════════════════════════════
function getMetrics() {
  const m = { cpu: 0, mem: 0, memUsed: 0, memTotal: 0, disk: 0, diskUsed: "", diskTotal: "", uptime: "", ip: "", host: "", procs: 0 };
  try { const la = execSync("cat /proc/loadavg", { encoding: "utf-8", timeout: 2000 }).split(" "); const c = parseInt(execSync("nproc", { encoding: "utf-8", timeout: 2000 })) || 1; m.cpu = Math.min(100, Math.round((parseFloat(la[0]) / c) * 100)); } catch {}
  try { const mi = execSync("free -m | awk 'NR==2{print $3,$2}'", { encoding: "utf-8", timeout: 2000 }).trim().split(" "); m.memUsed = +mi[0] || 0; m.memTotal = +mi[1] || 1; m.mem = Math.round((m.memUsed / m.memTotal) * 100); } catch {}
  try { const di = execSync("df -h / | awk 'NR==2{print $5,$3,$2}'", { encoding: "utf-8", timeout: 2000 }).trim().split(" "); m.disk = parseInt(di[0]) || 0; m.diskUsed = di[1] || "?"; m.diskTotal = di[2] || "?"; } catch {}
  try { m.uptime = execSync("uptime -p", { encoding: "utf-8", timeout: 2000 }).trim().replace("up ", ""); } catch {}
  try { m.host = execSync("hostname", { encoding: "utf-8", timeout: 2000 }).trim(); } catch {}
  try { m.ip = execSync("hostname -I | awk '{print $1}'", { encoding: "utf-8", timeout: 2000 }).trim(); } catch {}
  try { m.procs = parseInt(execSync("ps -e --no-headers | wc -l", { encoding: "utf-8", timeout: 2000 })) || 0; } catch {}
  return m;
}


// ══════════════════════════════════════════════════════════════════════════════
//  VISUAL COMPONENTS
// ══════════════════════════════════════════════════════════════════════════════

const cpuHist = new Array(16).fill(0);

function bar(pct, w = 12) {
  const f = Math.round((pct / 100) * w);
  const color = pct > 85 ? RB : pct > 50 ? R : RD;
  return color("█".repeat(f)) + D("░".repeat(w - f));
}

function spark(arr, w = 16) {
  const chars = "▁▂▃▄▅▆▇█";
  const max = Math.max(...arr, 1);
  return arr.slice(-w).map(v => R(chars[Math.min(7, Math.round((v / max) * 7))])).join("");
}

function pulse() {
  // Returns a pulsing dot based on current time (changes every 500ms)
  const t = Math.floor(Date.now() / 500) % 4;
  const frames = [RD("◦"), R("●"), RB("●"), R("●")];
  return frames[t];
}

function statusDot(active) {
  return active ? G("●") : RD("○");
}

function miniBox(label, value, w = 18) {
  const inner = w - 2;
  const lbl = D(label.padEnd(inner));
  const val = RA(String(value).slice(0, inner).padEnd(inner));
  return RD("┌" + "─".repeat(inner) + "┐") + "\n" +
         RD("│") + lbl + RD("│") + "\n" +
         RD("│") + val + RD("│") + "\n" +
         RD("└" + "─".repeat(inner) + "┘");
}


// ══════════════════════════════════════════════════════════════════════════════
//  HEADER PANEL (fixed at top)
// ══════════════════════════════════════════════════════════════════════════════

function getTermW() { return Math.min(process.stdout.columns || 72, 80); }

function renderPanel() {
  const m = getMetrics();
  cpuHist.push(m.cpu); if (cpuHist.length > 16) cpuHist.shift();

  const w = getTermW();
  const iw = w - 4; // inner width
  const hline = RD("─".repeat(iw));
  const model = config.get("default_model") || "—";
  const prov = (() => { const b = config.get("openai_api_base") || ""; if (b.includes("do-ai")) return "DO"; if (b.includes("openai.com")) return "OAI"; if (b.includes("openrouter")) return "OR"; if (b.includes("nvidia")) return "NV"; if (b.includes("huggingface")) return "HF"; if (b.includes("googleapis")) return "GG"; if (b.includes("cloudflare")) return "CF"; if (b.includes("localhost")) return "OLL"; return "?"; })();
  const iface = (config.get("interface_mode") || "cli").toUpperCase();
  const ready = !!config.get("openai_api_key");

  const L = [];

  // Top border
  L.push("  " + RD(BOX.tl + BOX.h.repeat(iw) + BOX.tr));

  // SYNTHCLAW identity — compact, red glow effect
  for (const row of SYNTHCLAW_BLOCK) {
    L.push("  " + RD(BOX.v) + " " + RB(row).padEnd(iw + 10) + RD(BOX.v));
  }

  // Separator
  L.push("  " + RD(BOX.vr + BOX.h.repeat(iw) + BOX.vl));

  // Status indicators row
  const st = [
    `${pulse()} ${D("SYS")} ${statusDot(ready)}`,
    `${D("MODEL")} ${RA(model.length > 22 ? model.slice(0, 20) + "…" : model)}`,
    `${D("PROV")} ${RA(prov)}`,
    `${D("CH")} ${RA(iface)}`,
  ].join("  ");
  L.push("  " + RD(BOX.v) + " " + st + " ".repeat(Math.max(0, iw - 62)) + RD(BOX.v));

  // Separator
  L.push("  " + RD(BOX.vr + BOX.h.repeat(iw) + BOX.vl));

  // Metrics row 1: gauges
  const cpuG = `${D("CPU")} ${bar(m.cpu, 8)} ${RA((m.cpu + "%").padStart(4))}`;
  const memG = `${D("MEM")} ${bar(m.mem, 8)} ${RA((m.mem + "%").padStart(4))}`;
  const dskG = `${D("DSK")} ${bar(m.disk, 8)} ${RA((m.disk + "%").padStart(4))}`;
  L.push("  " + RD(BOX.v) + " " + cpuG + " " + memG + " " + dskG + " " + RD(BOX.v));

  // Metrics row 2: sparkline + info
  const sp = spark(cpuHist, 16);
  const info = `${D("HOST")} ${RA(m.host || "?")}  ${D("IP")} ${RA(m.ip || "?")}  ${D("UP")} ${RA(m.uptime || "?")}`;
  L.push("  " + RD(BOX.v) + " " + D("LOAD ") + sp + "  " + info.slice(0, iw - 24) + " " + RD(BOX.v));

  // Bottom border
  L.push("  " + RD(BOX.bl + BOX.h.repeat(iw) + BOX.br));

  return L.join("\n");
}


// ══════════════════════════════════════════════════════════════════════════════
//  STARTUP ANIMATION
// ══════════════════════════════════════════════════════════════════════════════

async function startupSequence() {
  const delay = (ms) => new Promise(r => setTimeout(r, ms));
  process.stdout.write("\x1b[2J\x1b[H"); // clear

  // Phase 1: scanning effect
  const phases = [
    "  " + RD("▪ INITIALIZING CORE..."),
    "  " + RD("▪ LOADING MODULES..."),
    "  " + R("▪ CONNECTING PROVIDER..."),
    "  " + RB("▪ SYSTEM ONLINE"),
  ];

  for (const phase of phases) {
    process.stdout.write(phase + "\r");
    await delay(200);
    process.stdout.write(" ".repeat(50) + "\r");
  }
  await delay(100);

  // Phase 2: render header with fade-in effect (just render it)
  process.stdout.write("\x1b[2J\x1b[H");
}

// ══════════════════════════════════════════════════════════════════════════════
//  EXECUTION VISUALIZATION (replaces "thinking...")
// ══════════════════════════════════════════════════════════════════════════════

class ExecutionVisualizer {
  constructor() {
    this.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    this.frameIdx = 0;
    this.interval = null;
    this.stage = "";
  }

  start(stage = "processing") {
    this.stage = stage;
    this.frameIdx = 0;
    this.interval = setInterval(() => {
      this.frameIdx = (this.frameIdx + 1) % this.frames.length;
      const frame = R(this.frames[this.frameIdx]);
      const stageText = D(this.stage);
      process.stdout.write(`\r  ${frame} ${stageText}${"".padEnd(20)}`);
    }, 80);
  }

  setStage(stage) {
    this.stage = stage;
  }

  stop() {
    if (this.interval) { clearInterval(this.interval); this.interval = null; }
    process.stdout.write("\r" + " ".repeat(50) + "\r");
  }
}

const viz = new ExecutionVisualizer();


// ══════════════════════════════════════════════════════════════════════════════
//  PROVIDER CONFIG (which need more than API key)
// ══════════════════════════════════════════════════════════════════════════════

const PROVIDERS = {
  "DigitalOcean": { base: "https://inference.do-ai.run/v1", fields: ["api_key"] },
  "OpenAI": { base: "https://api.openai.com/v1", fields: ["api_key"] },
  "Anthropic (via DO)": { base: "https://inference.do-ai.run/v1", fields: ["api_key"] },
  "Google Gemini": { base: "https://generativelanguage.googleapis.com/v1beta/openai", fields: ["api_key"] },
  "NVIDIA NIM": { base: "https://integrate.api.nvidia.com/v1", fields: ["api_key"] },
  "HuggingFace": { base: "https://router.huggingface.co/v1", fields: ["api_key"] },
  "OpenRouter": { base: "https://openrouter.ai/api/v1", fields: ["api_key"] },
  "GitHub Models": { base: "https://models.inference.ai.azure.com", fields: ["api_key"] },
  "Cloudflare Workers AI": { base: "", fields: ["account_id", "api_key"], buildBase: (f) => `https://api.cloudflare.com/client/v4/accounts/${f.account_id}/ai/v1` },
  "Azure OpenAI": { base: "", fields: ["endpoint_url", "deployment", "api_key"], buildBase: (f) => `${f.endpoint_url}/openai/deployments/${f.deployment}` },
  "Ollama (local)": { base: "http://localhost:11434/v1", fields: [] },
  "Custom": { base: "", fields: ["base_url", "api_key"] },
};

// ══════════════════════════════════════════════════════════════════════════════
//  INLINE WIZARD (runs inside dashboard)
// ══════════════════════════════════════════════════════════════════════════════

async function runInlineWizard() {
  console.log("");
  console.log("  " + RD(BOX.tl + "─── ") + R("CONFIGURATION") + RD(" " + "─".repeat(30) + BOX.tr));

  // Storage
  const { storageMode } = await inquirer.prompt([{ type: "list", name: "storageMode", message: "Storage:", choices: [{ name: "Local SQLite", value: "local" }, { name: "Cloudflare D1", value: "cloudflare" }], default: config.get("storage_mode") || "local", prefix: "  " + RD(BOX.v) }]);
  config.set("storage_mode", storageMode);

  if (storageMode === "cloudflare") {
    const cf = await inquirer.prompt([
      { type: "input", name: "a", message: "CF Account ID:", default: config.get("cf_account_id") || undefined, prefix: "  " + RD(BOX.v) },
      { type: "password", name: "t", message: "CF API Token:", mask: "•", prefix: "  " + RD(BOX.v) },
      { type: "input", name: "d", message: "D1 Database ID:", default: config.get("cf_d1_database_id") || undefined, prefix: "  " + RD(BOX.v) },
    ]);
    config.set("cf_account_id", cf.a); config.set("cf_api_token", cf.t); config.set("cf_d1_database_id", cf.d);
  }

  // Interface
  const { iface } = await inquirer.prompt([{ type: "list", name: "iface", message: "Interface:", choices: [{ name: "CLI only", value: "cli" }, { name: "Telegram", value: "telegram" }, { name: "WhatsApp", value: "whatsapp" }, { name: "Both", value: "both" }], default: config.get("interface_mode") || "cli", prefix: "  " + RD(BOX.v) }]);
  config.set("interface_mode", iface);

  if (iface === "telegram" || iface === "both") {
    const { tk } = await inquirer.prompt([{ type: "password", name: "tk", message: "Telegram Token:", mask: "•", prefix: "  " + RD(BOX.v), validate: v => v.length > 10 || (config.get("telegram_token") && !v) ? true : "Invalid" }]);
    if (tk) config.set("telegram_token", tk);
  }
  if (iface === "whatsapp" || iface === "both") {
    const wa = await inquirer.prompt([
      { type: "password", name: "tk", message: "WhatsApp Token:", mask: "•", prefix: "  " + RD(BOX.v) },
      { type: "input", name: "ph", message: "Phone ID:", default: config.get("whatsapp_phone_number_id") || undefined, prefix: "  " + RD(BOX.v) },
    ]);
    if (wa.tk) config.set("whatsapp_token", wa.tk);
    if (wa.ph) config.set("whatsapp_phone_number_id", wa.ph);
  }

  // Provider
  const { prov } = await inquirer.prompt([{ type: "list", name: "prov", message: "AI Provider:", choices: Object.keys(PROVIDERS), prefix: "  " + RD(BOX.v) }]);
  const pc = PROVIDERS[prov];
  const pf = {};

  for (const field of pc.fields) {
    const msg = field === "api_key" ? `${prov} API Key:` : field === "account_id" ? "Account ID:" : field === "endpoint_url" ? "Endpoint URL:" : field === "deployment" ? "Deployment:" : "Base URL:";
    const type = field === "api_key" ? "password" : "input";
    const { v } = await inquirer.prompt([{ type, name: "v", message: msg, mask: type === "password" ? "•" : undefined, prefix: "  " + RD(BOX.v), default: field === "account_id" ? config.get("cf_account_id") : field === "base_url" ? config.get("openai_api_base") : undefined }]);
    pf[field] = v;
    if (field === "api_key" && v) config.set("openai_api_key", v);
    if (field === "account_id" && v) config.set("cf_account_id", v);
  }

  let base = pc.base;
  if (pc.buildBase) base = pc.buildBase(pf);
  else if (pf.base_url) base = pf.base_url;
  if (base) config.set("openai_api_base", base);

  // Model — live fetch
  let models = ["llama3.3-70b-instruct", "Custom"];
  const key = pf.api_key || config.get("openai_api_key");
  if (key && base) {
    try {
      const r = await fetch(`${base}/models`, { headers: { Authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(8000) });
      if (r.ok) { const d = await r.json(); const items = (d.data || d.models || []).map(i => typeof i === "string" ? i : (i.id || i.name || "")).filter(Boolean).slice(0, 20); if (items.length) models = [...items, "Custom"]; }
    } catch {}
  }

  const { mdl } = await inquirer.prompt([{ type: "list", name: "mdl", message: "Model:", choices: models, default: config.get("default_model"), prefix: "  " + RD(BOX.v) }]);
  if (mdl === "Custom") { const { c } = await inquirer.prompt([{ type: "input", name: "c", message: "Model ID:", prefix: "  " + RD(BOX.v) }]); config.set("default_model", c); }
  else config.set("default_model", mdl);

  // Save
  try { writeFileSync(join(getProjectRoot(), ".env"), generateEnvContent()); console.log("  " + RD(BOX.v) + " " + G("✓") + " Saved"); } catch { console.log("  " + RD(BOX.v) + " " + Y("⚠") + " Could not write .env"); }

  console.log("  " + RD(BOX.bl + "─".repeat(46) + BOX.br));
  console.log("");
}


// ══════════════════════════════════════════════════════════════════════════════
//  CHAT ENGINE (with execution stages)
// ══════════════════════════════════════════════════════════════════════════════

const chatHistory = [];

async function sendMessage(message) {
  const apiKey = config.get("openai_api_key");
  const apiBase = config.get("openai_api_base");
  const model = config.get("default_model");

  if (!apiKey) {
    console.log("  " + Y("⚠") + " Not configured. Running setup...");
    await runInlineWizard();
    return;
  }

  chatHistory.push({ role: "user", content: message });

  // Execution visualization stages
  viz.start("understanding request");

  try {
    const msgs = [
      { role: "system", content: "You are SynthClaw, a personal AI. Be concise. Plain text." },
      ...chatHistory.slice(-8),
    ];

    viz.setStage("connecting to " + (config.get("default_model") || "model"));

    const resp = await fetch(`${apiBase}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ model, messages: msgs, temperature: 0.7, max_tokens: 2048 }),
    });

    viz.setStage("processing response");
    const data = await resp.json();
    viz.stop();

    if (!resp.ok) {
      console.log("  " + R("✗") + " " + (data.error?.message || `HTTP ${resp.status}`));
      chatHistory.pop();
      return;
    }

    const reply = (data.choices?.[0]?.message?.content || "").replace(/<think>[\s\S]*?<\/think>/g, "").trim();
    chatHistory.push({ role: "assistant", content: reply });

    // Render reply in box
    console.log("");
    console.log("  " + RD(BOX.tl + "─── ") + R("SYNTHCLAW") + RD(" " + "─".repeat(34) + BOX.tr));
    for (const line of (reply || "(empty)").split("\n")) {
      console.log("  " + RD(BOX.v) + " " + line);
    }
    console.log("  " + RD(BOX.bl + "─".repeat(46) + BOX.br));
    console.log("");

  } catch (err) {
    viz.stop();
    console.log("  " + R("✗") + " " + err.message);
    chatHistory.pop();
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  COMMANDS
// ══════════════════════════════════════════════════════════════════════════════

const COMMANDS = ["/setup", "/status", "/model", "/models", "/clear", "/help", "/run", "/quit"];

function autocomplete(line) {
  if (!line.startsWith("/")) return [[], line];
  return [COMMANDS.filter(c => c.startsWith(line)), line];
}

async function handleCommand(input) {
  const [cmd, ...args] = input.split(" ");
  const arg = args.join(" ");

  switch (cmd) {
    case "/setup": await runInlineWizard(); break;
    case "/status": process.stdout.write("\x1b[2J\x1b[H"); console.log(renderPanel()); break;
    case "/model":
      if (arg) { config.set("default_model", arg); console.log("  " + G("✓") + " " + D("Model →") + " " + RA(arg)); }
      else console.log("  " + D("Model:") + " " + RA(config.get("default_model") || "—"));
      break;
    case "/clear":
      chatHistory.length = 0;
      process.stdout.write("\x1b[2J\x1b[H"); console.log(renderPanel());
      console.log("  " + D("Chat cleared."));
      break;
    case "/run":
      if (!arg) { console.log("  " + D("Usage: /run <command>")); break; }
      try {
        const out = execSync(arg, { encoding: "utf-8", timeout: 30000, cwd: getProjectRoot() });
        console.log("  " + RD(BOX.tl + "── output " + "─".repeat(35) + BOX.tr));
        for (const line of out.trim().split("\n").slice(0, 20)) {
          console.log("  " + RD(BOX.v) + " " + D(line));
        }
        console.log("  " + RD(BOX.bl + "─".repeat(46) + BOX.br));
      } catch (err) { console.log("  " + R("✗") + " " + (err.stderr || err.message || "").slice(0, 200)); }
      break;
    case "/help":
      console.log("");
      console.log("  " + R("COMMANDS"));
      console.log("  " + D("/setup") + "     Configure provider & settings");
      console.log("  " + D("/status") + "    Refresh system panel");
      console.log("  " + D("/model") + "     Show or switch model");
      console.log("  " + D("/run") + "       Execute shell command");
      console.log("  " + D("/clear") + "     Reset conversation");
      console.log("  " + D("/quit") + "      Exit");
      console.log("  " + D("<text>") + "     Chat with SynthClaw AI");
      console.log("");
      break;
    case "/quit": case "/exit": process.exit(0);
    default: await sendMessage(input);
  }
}


// ══════════════════════════════════════════════════════════════════════════════
//  MAIN ENTRY — DASHBOARD
// ══════════════════════════════════════════════════════════════════════════════

export async function runDashboard() {
  await startupSequence();
  console.log(renderPanel());
  console.log("");

  // Auto-setup if unconfigured
  if (!config.get("openai_api_key")) {
    console.log("  " + R("●") + " " + D("First launch detected."));
    console.log("");
    await runInlineWizard();
    process.stdout.write("\x1b[2J\x1b[H");
    console.log(renderPanel());
    console.log("");
  }

  console.log("  " + D("Ready. Type a message or /help."));
  console.log("");

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: "  " + R("▸ "),
    completer: autocomplete,
  });

  rl.prompt();

  rl.on("line", async (line) => {
    const input = line.trim();
    if (!input) { rl.prompt(); return; }
    if (input.startsWith("/")) { await handleCommand(input); }
    else { await sendMessage(input); }
    rl.prompt();
  });

  rl.on("close", () => {
    console.log(D("\n  Session closed.\n"));
    process.exit(0);
  });

  // Background: refresh header every 30s
  setInterval(() => {
    process.stdout.write("\x1b7\x1b[H" + renderPanel() + "\n\x1b8");
  }, 30000);
}

export { runInlineWizard };
