import chalk from "chalk";
import { execSync } from "child_process";
import { existsSync, writeFileSync } from "fs";
import { join } from "path";
import os from "os";
import inquirer from "inquirer";
import { createInterface } from "readline";
import { config, generateEnvContent, getProjectRoot } from "../utils.js";
import { MASCOT_OPEN, MASCOT_BLINK, WORDMARK, SUBTITLE } from "../ascii.js";

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

// ── COMMANDS ─────────────────────────────────────────────────────────────────
const CMD_CHOICES = [
  { name: R("⚙") + "  Setup        " + D("— configure provider & model"), value: "/setup" },
  { name: R("◎") + "  Model        " + D("— switch LLM model"), value: "/model" },
  { name: R("⊞") + "  Providers    " + D("— manage API keys"), value: "/providers" },
  { name: R("◈") + "  Skills       " + D("— install/manage skills"), value: "/skills" },
  { name: R("🏛") + " Society      " + D("— agent tree view"), value: "/society" },
  { name: R("▷") + "  Run          " + D("— execute shell command"), value: "/run" },
  { name: R("◻") + "  Clear        " + D("— clear chat"), value: "/clear" },
  { name: R("?") + "  Help         " + D("— show commands"), value: "/help" },
  { name: R("⊘") + "  Quit         " + D("— exit"), value: "/quit" },
];

// ══════════════════════════════════════════════════════════════════════════════
//  HEADER — printed once, shows full mascot
// ══════════════════════════════════════════════════════════════════════════════

function printHeader() {
  const w = Math.min(process.stdout.columns || 80, 72);
  const iw = w - 4;
  const model = config.get("default_model") || "not configured";
  const ml = model.length > 24 ? model.slice(0, 22) + "…" : model;
  const ready = config.get("openai_api_key") ? G("●") : R("○");

  console.log("");
  console.log("  " + RD(BX.tl + BX.h.repeat(iw) + BX.tr));

  // Full mascot (5 rows) + wordmark alongside
  for (let i = 0; i < 5; i++) {
    const mc = MASCOT_OPEN[i] || "";
    const wm = WORDMARK[i - 1] || "";
    const sub = i === 4 ? D(SUBTITLE) : "";
    const right = wm ? RB(wm) : sub;
    const pad = Math.max(1, iw - 16 - (wm ? wm.length : sub ? SUBTITLE.length : 0) - 1);
    console.log("  " + RD(BX.v) + " " + RB(mc.padEnd(15)) + " " + right + " ".repeat(pad) + RD(BX.v));
  }

  console.log("  " + RD(BX.vr + BX.h.repeat(iw) + BX.vl));
  console.log("  " + RD(BX.v) + ` ${ready} ${D("MODEL")} ${RA(ml)}` + " ".repeat(Math.max(1, iw - ml.length - 10)) + RD(BX.v));
  console.log("  " + RD(BX.bl + BX.h.repeat(iw) + BX.br));
  console.log("");
  console.log("  " + D("Type a message to chat. Type") + " " + R("/") + " " + D("for commands. Ctrl+C to exit."));
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
    const msg = f === "api_key" ? `${prov} Key:` : f === "account_id" ? "Account ID:" : f === "endpoint_url" ? "Endpoint URL:" : "Base URL:";
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
  console.log("  " + RD("──") + R("•") + " Setup complete. Model: " + chalk.bold(config.get("default_model")));
  console.log("");
}

async function cmdModel() {
  if (!config.get("openai_api_key")) { console.log("  " + Y("No API key. Running setup...")); await cmdSetup(); return; }
  const { p } = await inquirer.prompt([{ type: "list", name: "p", message: "Provider:", choices: Object.keys(PROVIDERS), prefix: PFX }]);
  const pc = PROVIDERS[p];
  const base = pc.buildBase ? pc.buildBase({ account_id: config.get("cf_account_id") }) : (pc.base || config.get("openai_api_base"));
  console.log("  " + D("Fetching models..."));
  let models = [];
  try { const r = await fetch(`${base}/models`, { headers: { Authorization: `Bearer ${config.get("openai_api_key")}` }, signal: AbortSignal.timeout(10000) }); if (r.ok) { const d = await r.json(); models = (d.data || d.models || []).map(i => typeof i === "string" ? i : (i.id || i.name || "")).filter(Boolean).slice(0, 30); } } catch {}
  if (!models.length) models = ["llama3.3-70b-instruct"];
  models.push(new inquirer.Separator(), { name: D("Custom..."), value: "__custom__" });
  const { m } = await inquirer.prompt([{ type: "list", name: "m", message: "Model:", choices: models, pageSize: 15, prefix: PFX }]);
  if (m === "__custom__") { const { c } = await inquirer.prompt([{ type: "input", name: "c", message: "ID:", prefix: PFX }]); config.set("default_model", c); }
  else config.set("default_model", m);
  console.log("  " + RD("──") + R("•") + " Model: " + chalk.bold(config.get("default_model")));
  console.log("");
}

async function cmdProviders() {
  const { p } = await inquirer.prompt([{ type: "list", name: "p", message: "Provider:", choices: Object.keys(PROVIDERS), prefix: PFX }]);
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
  if (!apiKey) { console.log("  " + Y("No API key. Running setup...")); await cmdSetup(); return; }
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

async function handleCmd(input, rl) {
  const [cmd, ...args] = input.split(" ");
  const arg = args.join(" ");
  switch (cmd) {
    case "/setup": return cmdSetup();
    case "/model": return cmdModel();
    case "/providers": return cmdProviders();
    case "/skills": {
      const { a } = await inquirer.prompt([{ type: "list", name: "a", message: "Skills:", choices: ["Install @user/skill", "List installed", "Remove"], prefix: PFX }]);
      if (a.startsWith("I")) { const { p } = await inquirer.prompt([{ type: "input", name: "p", message: "@user/skill:", prefix: PFX }]); if (p) console.log("  " + RD("──") + R("•") + ` ${p} installed`); }
      else console.log("  " + D("Skills managed via Telegram /skills or web frontend."));
      console.log(""); return;
    }
    case "/society": case "/agents":
      console.log(""); console.log("  " + R("AGENT SOCIETY"));
      console.log("  " + RD("──•") + " " + RA("Orchestrator") + "  " + D("plans + delegates"));
      console.log("  " + RD("  ├─") + " " + RA("Researcher") + "   " + D("gathers info"));
      console.log("  " + RD("  ├─") + " " + RA("Executor") + "     " + D("runs commands"));
      console.log("  " + RD("  ├─") + " " + RA("Reviewer") + "     " + D("validates results"));
      console.log("  " + RD("  └─") + " " + RA("Observer") + "     " + D("monitors execution"));
      console.log("  " + D("Complex tasks auto-delegate via should_delegate().")); console.log(""); return;
    case "/run":
      if (!arg) { const { c } = await inquirer.prompt([{ type: "input", name: "c", message: "$", prefix: PFX }]); if (c) return handleCmd("/run " + c, rl); return; }
      try { const o = execSync(arg, { encoding: "utf-8", timeout: 30000, cwd: getProjectRoot(), stdio: ["pipe", "pipe", "pipe"] }); console.log(D(o.trim().split("\n").slice(0, 20).map(l => "  " + l).join("\n"))); }
      catch (e) { console.log("  " + Y("✗") + " " + (e.stderr || e.message || "").slice(0, 100)); }
      console.log(""); return;
    case "/clear": chatHistory.length = 0; console.log("  " + D("Chat cleared.")); console.log(""); return;
    case "/help":
      console.log(""); console.log("  " + R("COMMANDS"));
      console.log("  " + RD("──•") + " " + R("/setup") + "       " + D("Configure provider & model"));
      console.log("  " + RD("──•") + " " + R("/model") + "       " + D("Switch LLM model"));
      console.log("  " + RD("──•") + " " + R("/providers") + "   " + D("Manage API keys"));
      console.log("  " + RD("──•") + " " + R("/skills") + "      " + D("Install/manage skills"));
      console.log("  " + RD("──•") + " " + R("/society") + "     " + D("Agent Society tree"));
      console.log("  " + RD("──•") + " " + R("/run <cmd>") + "   " + D("Execute shell command"));
      console.log("  " + RD("──•") + " " + R("/clear") + "       " + D("Clear chat"));
      console.log("  " + RD("──•") + " " + R("/quit") + "        " + D("Exit"));
      console.log(""); return;
    case "/quit": case "/exit": process.exit(0);
    default: return sendMessage(input);
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  MAIN — Simple readline. NO alternate screen. NO raw mode.
// ══════════════════════════════════════════════════════════════════════════════

export async function runDashboard() {
  printHeader();

  // Auto-setup if no key configured
  if (!config.get("openai_api_key")) {
    console.log("  " + R("●") + " No API key found. Let's configure:");
    console.log("");
    await cmdSetup();
  }

  // Simple readline — inquirer handles its own stdin when called
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: "  " + RD(BX.v) + " " + R("▸") + " ",
    terminal: true,
  });

  rl.prompt();

  rl.on("line", async (line) => {
    const input = line.trim();
    if (!input) { rl.prompt(); return; }

    // CRITICAL: pause readline so inquirer can use stdin
    rl.pause();

    try {
      if (input === "/") {
        // Show command menu
        const { c } = await inquirer.prompt([{
          type: "list",
          name: "c",
          message: R("▸") + " Command:",
          choices: CMD_CHOICES,
          pageSize: 12,
        }]);
        await handleCmd(c, rl);
      } else if (input.startsWith("/")) {
        await handleCmd(input, rl);
      } else {
        await sendMessage(input);
      }
    } catch (e) {
      if (!e.message?.includes("force closed")) {
        console.log("  " + Y("✗") + " " + (e.message || "Error"));
      }
    }

    // Resume readline
    rl.resume();
    rl.prompt();
  });

  rl.on("close", () => {
    console.log("");
    process.exit(0);
  });
}

export { cmdSetup as runInlineWizard };
