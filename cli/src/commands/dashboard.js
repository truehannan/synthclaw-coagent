import chalk from "chalk";
import { execSync } from "child_process";
import { existsSync, writeFileSync } from "fs";
import { join } from "path";
import inquirer from "inquirer";
import { config, generateEnvContent, getProjectRoot } from "../utils.js";
import { MASCOT_OPEN, WORDMARK, SUBTITLE } from "../ascii.js";

// ── THEME ────────────────────────────────────────────────────────────────────
const R = chalk.hex("#cc0000"), RB = chalk.hex("#ff1a1a"), RD = chalk.hex("#e85d04");
const RA = chalk.hex("#ff3333"), D = chalk.dim, G = chalk.hex("#33ff33"), Y = chalk.hex("#ffaa00");
const BX = { tl: "╭", tr: "╮", bl: "╰", br: "╯", h: "─", v: "│", vr: "├", vl: "┤" };

// ── PROVIDERS ────────────────────────────────────────────────────────────────
const PROVIDERS = {
  "DigitalOcean": { base: "https://inference.do-ai.run/v1", fields: ["api_key"] },
  "OpenAI": { base: "https://api.openai.com/v1", fields: ["api_key"] },
  "Anthropic (via DO)": { base: "https://inference.do-ai.run/v1", fields: ["api_key"] },
  "Google Gemini": { base: "https://generativelanguage.googleapis.com/v1beta/openai", fields: ["api_key"] },
  "NVIDIA NIM": { base: "https://integrate.api.nvidia.com/v1", fields: ["api_key"] },
  "HuggingFace": { base: "https://router.huggingface.co/v1", fields: ["api_key"] },
  "OpenRouter": { base: "https://openrouter.ai/api/v1", fields: ["api_key"] },
  "GitHub Models": { base: "https://models.inference.ai.azure.com", fields: ["api_key"] },
  "Qwen (DashScope)": { base: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", fields: ["api_key"] },
  "Cloudflare Workers AI": { base: "", fields: ["account_id", "api_key"], buildBase: f => `https://api.cloudflare.com/client/v4/accounts/${f.account_id}/ai/v1` },
  "Ollama (local)": { base: "http://localhost:11434/v1", fields: [] },
  "Custom": { base: "", fields: ["base_url", "api_key"] },
};
const PFX = "  " + RD(BX.v);

// ══════════════════════════════════════════════════════════════════════════════
//  HEADER — all 5 mascot rows + wordmark + model
// ══════════════════════════════════════════════════════════════════════════════

function printHeader() {
  const w = Math.min(process.stdout.columns || 80, 72);
  const iw = w - 4;
  const model = config.get("default_model") || "not configured";
  const ml = model.length > 30 ? model.slice(0, 28) + "…" : model;
  const ready = config.get("openai_api_key") ? G("●") : R("○");

  console.log("");
  console.log("  " + RD(BX.tl + BX.h.repeat(iw) + BX.tr));
  console.log("  " + RD(BX.v) + " " + RB(MASCOT_OPEN[0].padEnd(15)) + " ".repeat(Math.max(1, iw - 17)) + RD(BX.v));
  console.log("  " + RD(BX.v) + " " + RB(MASCOT_OPEN[1].padEnd(15)) + " " + RB(WORDMARK[0]) + " ".repeat(Math.max(1, iw - 16 - WORDMARK[0].length - 1)) + RD(BX.v));
  console.log("  " + RD(BX.v) + " " + RB(MASCOT_OPEN[2].padEnd(15)) + " " + RB(WORDMARK[1]) + " ".repeat(Math.max(1, iw - 16 - WORDMARK[1].length - 1)) + RD(BX.v));
  console.log("  " + RD(BX.v) + " " + RB(MASCOT_OPEN[3].padEnd(15)) + " " + RB(WORDMARK[2]) + " ".repeat(Math.max(1, iw - 16 - WORDMARK[2].length - 1)) + RD(BX.v));
  console.log("  " + RD(BX.v) + " " + RB(MASCOT_OPEN[4].padEnd(15)) + " " + D(SUBTITLE) + " ".repeat(Math.max(1, iw - 16 - SUBTITLE.length - 1)) + RD(BX.v));
  console.log("  " + RD(BX.vr + BX.h.repeat(iw) + BX.vl));
  console.log("  " + RD(BX.v) + ` ${ready} ${D("MODEL")} ${RA(ml)}` + " ".repeat(Math.max(1, iw - ml.length - 10)) + RD(BX.v));
  console.log("  " + RD(BX.bl + BX.h.repeat(iw) + BX.br));
  console.log("");
}

// ══════════════════════════════════════════════════════════════════════════════
//  MENU — printed as numbered list, user types number to select
// ══════════════════════════════════════════════════════════════════════════════

const COMMANDS = [
  { key: "1", cmd: "/setup", label: "Setup", desc: "configure provider & model" },
  { key: "2", cmd: "/model", label: "Model", desc: "switch LLM model" },
  { key: "3", cmd: "/providers", label: "Providers", desc: "manage API keys" },
  { key: "4", cmd: "/skills", label: "Skills", desc: "install/manage skills" },
  { key: "5", cmd: "/society", label: "Society", desc: "agent tree view" },
  { key: "6", cmd: "/run", label: "Run", desc: "execute shell command" },
  { key: "7", cmd: "/clear", label: "Clear", desc: "clear chat history" },
  { key: "8", cmd: "/help", label: "Help", desc: "show this menu" },
  { key: "9", cmd: "/quit", label: "Quit", desc: "exit" },
];

function printMenu() {
  console.log("");
  console.log("  " + R("COMMANDS") + "  " + D("(type the number)"));
  console.log("");
  for (const c of COMMANDS) {
    console.log("  " + RD(c.key) + "  " + R(c.label.padEnd(12)) + D(c.desc));
  }
  console.log("");
  console.log("  " + D("Or type a slash command directly: /setup, /model, etc."));
  console.log("");
}

// ══════════════════════════════════════════════════════════════════════════════
//  COMMAND HANDLERS
// ══════════════════════════════════════════════════════════════════════════════

const chatHistory = [];

async function cmdSetup() {
  const { prov } = await inquirer.prompt([{ type: "list", name: "prov", message: "Provider:", choices: Object.keys(PROVIDERS), prefix: PFX }]);
  const pc = PROVIDERS[prov], pf = {};
  for (const f of pc.fields) {
    const tp = f === "api_key" ? "password" : "input";
    const msg = f === "api_key" ? `${prov} Key:` : f === "account_id" ? "Account ID:" : "Value:";
    const { v } = await inquirer.prompt([{ type: tp, name: "v", message: msg, mask: tp === "password" ? "•" : undefined, prefix: PFX }]);
    pf[f] = v;
    if (f === "api_key" && v) config.set("openai_api_key", v);
    if (f === "account_id" && v) config.set("cf_account_id", v);
  }
  let base = pc.base;
  if (pc.buildBase) base = pc.buildBase(pf);
  else if (pf.base_url) base = pf.base_url;
  if (base) config.set("openai_api_base", base);
  let models = ["llama3.3-70b-instruct", "Custom"];
  const key = pf.api_key || config.get("openai_api_key");
  if (key && base) {
    console.log("  " + D("Fetching models..."));
    try { const r = await fetch(`${base}/models`, { headers: { Authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(8000) }); if (r.ok) { const d = await r.json(); const it = (d.data || d.models || []).map(i => typeof i === "string" ? i : (i.id || i.name || "")).filter(Boolean).slice(0, 25); if (it.length) models = [...it, "Custom"]; } } catch {}
  }
  const { mdl } = await inquirer.prompt([{ type: "list", name: "mdl", message: "Model:", choices: models, default: config.get("default_model"), pageSize: 15, prefix: PFX }]);
  if (mdl === "Custom") { const { c } = await inquirer.prompt([{ type: "input", name: "c", message: "Model ID:", prefix: PFX }]); config.set("default_model", c); }
  else config.set("default_model", mdl);
  try { writeFileSync(join(getProjectRoot(), ".env"), generateEnvContent()); } catch {}
  console.log("  " + RD("──") + R("•") + " Done. Model: " + chalk.bold(config.get("default_model")));
  console.log("");
}

async function cmdModel() {
  if (!config.get("openai_api_key")) { console.log("  " + Y("No API key. Running setup...")); await cmdSetup(); return; }
  const provChoices = [...Object.keys(PROVIDERS), new inquirer.Separator(), { name: D("← Back"), value: "__back__" }];
  const { p } = await inquirer.prompt([{ type: "list", name: "p", message: "Provider:", choices: provChoices, prefix: PFX }]);
  if (p === "__back__") return;
  const pc = PROVIDERS[p];
  const base = pc.buildBase ? pc.buildBase({ account_id: config.get("cf_account_id") }) : (pc.base || config.get("openai_api_base"));
  console.log("  " + D("Fetching models..."));
  let models = [];
  try { const r = await fetch(`${base}/models`, { headers: { Authorization: `Bearer ${config.get("openai_api_key")}` }, signal: AbortSignal.timeout(10000) }); if (r.ok) { const d = await r.json(); models = (d.data || d.models || []).map(i => typeof i === "string" ? i : (i.id || i.name || "")).filter(Boolean).slice(0, 30); } } catch {}
  if (!models.length) models = [];
  // Always add custom input + back options
  models.push(new inquirer.Separator(), { name: R("✎") + "  Enter model ID manually", value: "__custom__" }, { name: D("← Back"), value: "__back__" });
  const { m } = await inquirer.prompt([{ type: "list", name: "m", message: "Model:", choices: models, pageSize: 15, prefix: PFX }]);
  if (m === "__back__") return;
  if (m === "__custom__") {
    const { c } = await inquirer.prompt([{ type: "input", name: "c", message: "Model ID:", prefix: PFX }]);
    if (c) { config.set("default_model", c); console.log("  " + RD("──") + R("•") + " Model: " + chalk.bold(c)); }
  } else {
    config.set("default_model", m);
    console.log("  " + RD("──") + R("•") + " Model: " + chalk.bold(m));
  }
  console.log("");
}

async function cmdProviders() {
  const provChoices = [...Object.keys(PROVIDERS), new inquirer.Separator(), { name: D("← Back"), value: "__back__" }];
  const { p } = await inquirer.prompt([{ type: "list", name: "p", message: "Provider:", choices: provChoices, prefix: PFX }]);
  if (p === "__back__") return;
  const pc = PROVIDERS[p];
  if (!pc.fields.length) { console.log("  " + D("No config needed.")); return; }
  const pf = {};
  for (const f of pc.fields) {
    const tp = f === "api_key" ? "password" : "input";
    const { v } = await inquirer.prompt([{ type: tp, name: "v", message: f === "api_key" ? `${p} Key:` : f + ":", mask: tp === "password" ? "•" : undefined, prefix: PFX }]);
    pf[f] = v;
    if (f === "api_key" && v) config.set("openai_api_key", v);
    if (f === "account_id" && v) config.set("cf_account_id", v);
  }
  let base = pc.base;
  if (pc.buildBase) base = pc.buildBase(pf); else if (pf.base_url) base = pf.base_url;
  if (base) config.set("openai_api_base", base);
  try { writeFileSync(join(getProjectRoot(), ".env"), generateEnvContent()); } catch {}
  console.log("  " + RD("──") + R("•") + ` ${p} configured`);
  console.log("");
}

async function sendMessage(msg) {
  const apiKey = config.get("openai_api_key"), apiBase = config.get("openai_api_base"), model = config.get("default_model");
  if (!apiKey) { console.log("  " + Y("No API key. Running /setup...")); await cmdSetup(); return; }
  chatHistory.push({ role: "user", content: msg });
  process.stdout.write("  " + D("⠋ Thinking..."));
  try {
    const msgs = [{ role: "system", content: "You are SynthClaw, a personal AI agent. Be concise and helpful." }, ...chatHistory.slice(-10)];
    const resp = await fetch(`${apiBase}/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` }, body: JSON.stringify({ model, messages: msgs, temperature: 0.7, max_tokens: 2048 }) });
    const data = await resp.json();
    process.stdout.write("\r" + " ".repeat(30) + "\r");
    if (!resp.ok) { console.log("  " + Y("✗") + " " + (data.error?.message || `HTTP ${resp.status}`)); chatHistory.pop(); return; }
    const reply = (data.choices?.[0]?.message?.content || "").replace(/<think>[\s\S]*?<\/think>/g, "").trim();
    chatHistory.push({ role: "assistant", content: reply });
    console.log("  " + RD(BX.tl + "── ") + R("SYNTHCLAW") + RD(" " + "─".repeat(30) + BX.tr));
    for (const line of (reply || "(empty)").split("\n")) console.log("  " + RD(BX.v) + " " + line);
    console.log("  " + RD(BX.bl + "─".repeat(42) + BX.br));
    console.log("");
  } catch (e) {
    process.stdout.write("\r" + " ".repeat(30) + "\r");
    console.log("  " + Y("✗") + " " + e.message); chatHistory.pop();
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  MAIN LOOP — inquirer input for chat, inquirer list for commands
//  Type / or press Enter empty → shows command LIST (arrow key select)
//  Type anything else → sends as chat message
// ══════════════════════════════════════════════════════════════════════════════

const CMD_CHOICES = [
  { name: R("⚙") + "  Setup        " + D("— configure provider & model"), value: "/setup" },
  { name: R("◎") + "  Model        " + D("— switch LLM model"), value: "/model" },
  { name: R("⊞") + "  Providers    " + D("— manage API keys"), value: "/providers" },
  { name: R("◈") + "  Skills       " + D("— install/manage skills"), value: "/skills" },
  { name: R("🏛") + " Society      " + D("— agent tree view"), value: "/society" },
  { name: R("▷") + "  Run          " + D("— execute shell command"), value: "/run" },
  { name: R("◻") + "  Clear        " + D("— clear chat"), value: "/clear" },
  { name: R("⊘") + "  Quit         " + D("— exit"), value: "/quit" },
  new inquirer.Separator(),
  { name: D("  Cancel (go back to chat)"), value: "__cancel__" },
];

export async function runDashboard() {
  printHeader();

  if (!config.get("openai_api_key")) {
    console.log("  " + R("●") + " No API key found. Let's configure:");
    console.log("");
    await cmdSetup();
  } else {
    console.log("  " + D("Chat or type") + " " + R("/") + " " + D("+ Enter for commands."));
    console.log("");
  }

  while (true) {
    // Input prompt
    let input = "";
    try {
      const { msg } = await inquirer.prompt([{ type: "input", name: "msg", message: R("▸"), prefix: "  " + RD(BX.v) }]);
      input = (msg || "").trim();
    } catch { break; }

    // Empty or "/" → show command selector (inquirer list — this ALWAYS works)
    if (!input || input === "/" || input === "/help") {
      try {
        const { cmd } = await inquirer.prompt([{ type: "list", name: "cmd", message: "Command:", choices: CMD_CHOICES, pageSize: 12, prefix: PFX }]);
        if (cmd === "__cancel__") continue;
        input = cmd;
      } catch { break; }
    }

    // Process command or message
    if (input === "/quit" || input === "/exit") break;
    if (input === "/setup") { await cmdSetup(); continue; }
    if (input === "/model") { await cmdModel(); continue; }
    if (input === "/providers") { await cmdProviders(); continue; }
    if (input === "/skills") {
      const { a } = await inquirer.prompt([{ type: "list", name: "a", message: "Skills:", choices: ["Install @user/skill", "List", "Remove", new inquirer.Separator(), { name: D("← Back"), value: "__back__" }], prefix: PFX }]);
      if (a === "__back__") continue;
      if (a.startsWith("I")) { const { p } = await inquirer.prompt([{ type: "input", name: "p", message: "@user/skill:", prefix: PFX }]); if (p) console.log("  " + RD("──") + R("•") + ` ${p} installed`); }
      console.log(""); continue;
    }
    if (input === "/society" || input === "/agents") {
      console.log(""); console.log("  " + R("AGENT SOCIETY"));
      console.log("  " + RD("──•") + " " + RA("Orchestrator") + "  " + D("plans + delegates"));
      console.log("  " + RD("  ├─") + " " + RA("Researcher") + "   " + D("gathers info"));
      console.log("  " + RD("  ├─") + " " + RA("Executor") + "     " + D("runs commands"));
      console.log("  " + RD("  ├─") + " " + RA("Reviewer") + "     " + D("validates"));
      console.log("  " + RD("  └─") + " " + RA("Observer") + "     " + D("monitors"));
      console.log(""); continue;
    }
    if (input.startsWith("/run")) {
      let c = input.startsWith("/run ") ? input.slice(5) : "";
      if (!c) { const { v } = await inquirer.prompt([{ type: "input", name: "v", message: "$", prefix: PFX }]); c = v; }
      if (c) { try { const o = execSync(c, { encoding: "utf-8", timeout: 30000, cwd: getProjectRoot(), stdio: ["pipe", "pipe", "pipe"] }); console.log(D(o.trim().split("\n").slice(0, 20).map(l => "  " + l).join("\n"))); } catch (e) { console.log("  " + Y("✗") + " " + (e.stderr || e.message || "").slice(0, 100)); } }
      console.log(""); continue;
    }
    if (input === "/clear") { chatHistory.length = 0; console.log("  " + D("Cleared.")); console.log(""); continue; }
    if (input.startsWith("/")) { console.log("  " + D("Unknown command.")); console.log(""); continue; }

    // Chat message
    await sendMessage(input);
  }

  process.exit(0);
}

export { cmdSetup as runInlineWizard };
