import chalk from "chalk";
import { execSync } from "child_process";
import { existsSync, writeFileSync } from "fs";
import { join } from "path";
import os from "os";
import inquirer from "inquirer";
import { config, generateEnvContent, getProjectRoot } from "../utils.js";
import { MASCOT_OPEN, MASCOT_BLINK, WORDMARK, SUBTITLE } from "../ascii.js";

// ── THEME ────────────────────────────────────────────────────────────────────
const R = chalk.hex("#cc0000"), RB = chalk.hex("#ff1a1a"), RD = chalk.hex("#e85d04");
const RA = chalk.hex("#ff3333"), D = chalk.dim, G = chalk.hex("#33ff33"), Y = chalk.hex("#ffaa00");
const BX = { tl: "╭", tr: "╮", bl: "╰", br: "╯", h: "─", v: "│", vr: "├", vl: "┤" };
const isWin = process.platform === "win32";

// ── TERMINAL ─────────────────────────────────────────────────────────────────
const ESC = "\x1b[";
const write = (s) => process.stdout.write(s);
const moveTo = (r, c) => write(`${ESC}${r};${c}H`);
const clearLine = () => write(ESC + "2K");
const clearScreen = () => write(ESC + "2J");
const hideCursor = () => write(ESC + "?25l");
const showCursor = () => write(ESC + "?25h");
const enterAlt = () => write("\x1b[?1049h");
const leaveAlt = () => write("\x1b[?1049l");
const tRows = () => process.stdout.rows || 24;
const tCols = () => process.stdout.columns || 80;

// ── LAYOUT (nano-like: fixed header, fixed input, chat scrolls in middle) ────
const HEADER_ROWS = 4;   // compact header: model + status only
const INPUT_ROWS = 1;    // single input line at very bottom
const BORDER_ROW = 1;    // border between chat and input

function chatStartRow() { return HEADER_ROWS + 1; }
function chatEndRow() { return tRows() - INPUT_ROWS - BORDER_ROW; }
function chatHeight() { return chatEndRow() - chatStartRow() + 1; }
function inputRow() { return tRows(); }
function borderRow() { return tRows() - INPUT_ROWS; }

// ── STATE ────────────────────────────────────────────────────────────────────
let chatLines = [];      // all output lines (scrollable)
let chatScroll = 0;      // how many lines scrolled up from bottom
let inputBuffer = "";
let inputCursor = 0;
let cmdMenuActive = false;
let cmdMenuIndex = 0;
let cmdFiltered = [];
let blinkState = 0, blinkTimer = null;
let processing = false;

// ── PROVIDERS ────────────────────────────────────────────────────────────────
const PROVIDERS = { "DigitalOcean": { base: "https://inference.do-ai.run/v1", fields: ["api_key"] }, "OpenAI": { base: "https://api.openai.com/v1", fields: ["api_key"] }, "Anthropic (via DO)": { base: "https://inference.do-ai.run/v1", fields: ["api_key"] }, "Google Gemini": { base: "https://generativelanguage.googleapis.com/v1beta/openai", fields: ["api_key"] }, "NVIDIA NIM": { base: "https://integrate.api.nvidia.com/v1", fields: ["api_key"] }, "HuggingFace": { base: "https://router.huggingface.co/v1", fields: ["api_key"] }, "OpenRouter": { base: "https://openrouter.ai/api/v1", fields: ["api_key"] }, "GitHub Models": { base: "https://models.inference.ai.azure.com", fields: ["api_key"] }, "Qwen (DashScope)": { base: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", fields: ["api_key"] }, "Cloudflare Workers AI": { base: "", fields: ["account_id", "api_key"], buildBase: f => `https://api.cloudflare.com/client/v4/accounts/${f.account_id}/ai/v1` }, "Ollama (local)": { base: "http://localhost:11434/v1", fields: [] }, "Custom": { base: "", fields: ["base_url", "api_key"] } };
const PFX = "  " + RD(BX.v);

// ── COMMANDS ─────────────────────────────────────────────────────────────────
const CMD_LIST = [
  { name: "Setup", value: "/setup", desc: "Configure provider & model" },
  { name: "Model", value: "/model", desc: "Switch LLM model" },
  { name: "Providers", value: "/providers", desc: "Manage API keys" },
  { name: "Skills", value: "/skills", desc: "Install/manage skills" },
  { name: "Society", value: "/society", desc: "Agent tree view" },
  { name: "Run", value: "/run", desc: "Shell command" },
  { name: "Clear", value: "/clear", desc: "Clear chat" },
  { name: "Quit", value: "/quit", desc: "Exit" },
];

// ══════════════════════════════════════════════════════════════════════════════
//  RENDER FUNCTIONS
// ══════════════════════════════════════════════════════════════════════════════

function renderHeader() {
  const w = Math.min(tCols(), 74);
  const model = config.get("default_model") || "not configured";
  const ml = model.length > 28 ? model.slice(0, 26) + "…" : model;
  const ready = config.get("openai_api_key") ? G("●") : R("○");

  // Row 1: top border
  moveTo(1, 1); clearLine();
  write("  " + RD(BX.tl + BX.h.repeat(w - 2) + BX.tr));

  // Row 2: brand + model
  moveTo(2, 1); clearLine();
  const mascot = blinkState === 0 ? MASCOT_OPEN[2] : MASCOT_BLINK[2];
  write("  " + RD(BX.v) + " " + RB(mascot) + "  " + ready + " " + RA(ml) + " ".repeat(Math.max(1, w - ml.length - 16)) + RD(BX.v));

  // Row 3: subtitle
  moveTo(3, 1); clearLine();
  write("  " + RD(BX.v) + " " + D("Agent Society") + "  " + D("/ for commands") + " ".repeat(Math.max(1, w - 32)) + RD(BX.v));

  // Row 4: bottom border
  moveTo(4, 1); clearLine();
  write("  " + RD(BX.bl + BX.h.repeat(w - 2) + BX.br));
}

function renderChat() {
  const h = chatHeight();
  const startIdx = Math.max(0, chatLines.length - h - chatScroll);
  const endIdx = startIdx + h;
  const visible = chatLines.slice(startIdx, endIdx);

  for (let i = 0; i < h; i++) {
    const row = chatStartRow() + i;
    moveTo(row, 1); clearLine();
    if (i < visible.length) {
      write("  " + visible[i]);
    }
  }
}

function renderBorder() {
  moveTo(borderRow(), 1); clearLine();
  const w = Math.min(tCols() - 4, 72);
  write("  " + RD(BX.h.repeat(w)));
}

function renderInput() {
  moveTo(inputRow(), 1); clearLine();
  const prefix = "  " + RD(BX.v) + " " + R("▸") + " ";
  write(prefix + inputBuffer);
  // Place cursor at correct position (prefix visible width = 7)
  moveTo(inputRow(), 7 + inputCursor + 1);
  showCursor();
}

function renderCmdMenu() {
  if (!cmdMenuActive) return;
  const menuH = cmdFiltered.length + 2; // +2 for borders
  const startRow = borderRow() - menuH;

  moveTo(startRow, 1); clearLine();
  write("  " + RD(BX.tl + BX.h.repeat(38) + BX.tr));

  for (let i = 0; i < cmdFiltered.length; i++) {
    const c = cmdFiltered[i];
    const sel = i === cmdMenuIndex;
    moveTo(startRow + 1 + i, 1); clearLine();
    const marker = sel ? R("▸") : " ";
    const nm = sel ? RA(c.name.padEnd(12)) : D(c.name.padEnd(12));
    write(`  ${RD(BX.v)} ${marker} ${nm} ${D(c.desc)}`);
  }

  moveTo(startRow + cmdFiltered.length + 1, 1); clearLine();
  write("  " + RD(BX.bl + BX.h.repeat(38) + BX.br));
}

function fullRender() {
  renderHeader();
  renderChat();
  renderBorder();
  if (cmdMenuActive) renderCmdMenu();
  renderInput();
}

// ══════════════════════════════════════════════════════════════════════════════
//  CHAT OUTPUT
// ══════════════════════════════════════════════════════════════════════════════

function addLine(text) {
  chatLines.push(text);
  chatScroll = 0; // auto-scroll to bottom
  renderChat();
  renderBorder();
  renderInput();
}

function addLines(lines) {
  chatLines.push(...lines);
  chatScroll = 0;
  renderChat();
  renderBorder();
  renderInput();
}

// ══════════════════════════════════════════════════════════════════════════════
//  COMMAND MENU
// ══════════════════════════════════════════════════════════════════════════════

function showMenu(filter = "") {
  cmdMenuActive = true;
  cmdMenuIndex = 0;
  const f = filter.replace("/", "").toLowerCase();
  cmdFiltered = f ? CMD_LIST.filter(c => c.name.toLowerCase().includes(f) || c.value.includes(f)) : [...CMD_LIST];
  if (!cmdFiltered.length) cmdFiltered = [...CMD_LIST];
  renderCmdMenu();
  renderInput();
}

function hideMenu() {
  if (!cmdMenuActive) return;
  cmdMenuActive = false;
  // Clear menu area by re-rendering chat over it
  renderChat();
  renderBorder();
  renderInput();
}

// ══════════════════════════════════════════════════════════════════════════════
//  COMMAND HANDLERS
// ══════════════════════════════════════════════════════════════════════════════

const chatHistory = [];

async function withInquirer(fn) {
  // Exit raw mode, run inquirer, re-enter raw mode
  if (process.stdin.isTTY) process.stdin.setRawMode(false);
  process.stdin.pause();
  try { await fn(); } finally {
    if (process.stdin.isTTY) process.stdin.setRawMode(true);
    process.stdin.resume();
    clearScreen();
    fullRender();
  }
}

async function cmdSetup() {
  await withInquirer(async () => {
    const { prov } = await inquirer.prompt([{ type: "list", name: "prov", message: "Provider:", choices: Object.keys(PROVIDERS), prefix: PFX }]);
    const pc = PROVIDERS[prov], pf = {};
    for (const f of pc.fields) { const tp = f === "api_key" ? "password" : "input"; const { v } = await inquirer.prompt([{ type: tp, name: "v", message: f === "api_key" ? `${prov} Key:` : f + ":", mask: tp === "password" ? "•" : undefined, prefix: PFX }]); pf[f] = v; if (f === "api_key" && v) config.set("openai_api_key", v); if (f === "account_id" && v) config.set("cf_account_id", v); }
    let base = pc.base; if (pc.buildBase) base = pc.buildBase(pf); else if (pf.base_url) base = pf.base_url; if (base) config.set("openai_api_base", base);
    // Fetch models
    let models = ["llama3.3-70b-instruct", "Custom"]; const key = pf.api_key || config.get("openai_api_key");
    if (key && base) { try { const r = await fetch(`${base}/models`, { headers: { Authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(8000) }); if (r.ok) { const d = await r.json(); const it = (d.data || d.models || []).map(i => typeof i === "string" ? i : (i.id || i.name || "")).filter(Boolean).slice(0, 20); if (it.length) models = [...it, "Custom"]; } } catch {} }
    const { mdl } = await inquirer.prompt([{ type: "list", name: "mdl", message: "Model:", choices: models, default: config.get("default_model"), prefix: PFX }]);
    if (mdl === "Custom") { const { c } = await inquirer.prompt([{ type: "input", name: "c", message: "Model ID:", prefix: PFX }]); config.set("default_model", c); } else config.set("default_model", mdl);
    try { writeFileSync(join(getProjectRoot(), ".env"), generateEnvContent()); } catch {}
    addLine(RD("──") + R("•") + " Setup complete. Model: " + config.get("default_model"));
  });
}

async function cmdModel() {
  await withInquirer(async () => {
    if (!config.get("openai_api_key")) { await cmdSetup(); return; }
    const { p } = await inquirer.prompt([{ type: "list", name: "p", message: "Provider:", choices: Object.keys(PROVIDERS), prefix: PFX }]);
    let models = []; const key = config.get("openai_api_key"), pc = PROVIDERS[p];
    const base = pc.buildBase ? pc.buildBase({ account_id: config.get("cf_account_id") }) : (pc.base || config.get("openai_api_base"));
    try { const r = await fetch(`${base}/models`, { headers: { Authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(10000) }); if (r.ok) { const d = await r.json(); models = (d.data || d.models || []).map(i => typeof i === "string" ? i : (i.id || i.name || "")).filter(Boolean).slice(0, 30); } } catch {}
    if (!models.length) models = ["llama3.3-70b-instruct"];
    const { m } = await inquirer.prompt([{ type: "list", name: "m", message: "Model:", choices: models, pageSize: 15, prefix: PFX }]);
    config.set("default_model", m);
    addLine(RD("──") + R("•") + " Model: " + m);
  });
}

async function cmdProviders() {
  await withInquirer(async () => {
    const { p } = await inquirer.prompt([{ type: "list", name: "p", message: "Provider:", choices: Object.keys(PROVIDERS), prefix: PFX }]);
    const pc = PROVIDERS[p]; if (!pc.fields.length) { addLine(D("  No config needed.")); return; }
    const pf = {};
    for (const f of pc.fields) { const tp = f === "api_key" ? "password" : "input"; const { v } = await inquirer.prompt([{ type: tp, name: "v", message: f === "api_key" ? `${p} Key:` : "Value:", mask: tp === "password" ? "•" : undefined, prefix: PFX }]); pf[f] = v; if (f === "api_key" && v) config.set("openai_api_key", v); if (f === "account_id" && v) config.set("cf_account_id", v); }
    let base = pc.base; if (pc.buildBase) base = pc.buildBase(pf); else if (pf.base_url) base = pf.base_url; if (base) config.set("openai_api_base", base);
    try { writeFileSync(join(getProjectRoot(), ".env"), generateEnvContent()); } catch {}
    addLine(RD("──") + R("•") + ` ${p} configured`);
  });
}

async function sendMessage(msg) {
  const apiKey = config.get("openai_api_key"), apiBase = config.get("openai_api_base"), model = config.get("default_model");
  if (!apiKey) { await cmdSetup(); return; }
  chatHistory.push({ role: "user", content: msg });
  addLine(D("  ⠋ Thinking..."));
  try {
    const msgs = [{ role: "system", content: "You are SynthClaw, a personal AI agent. Be concise." }, ...chatHistory.slice(-8)];
    const resp = await fetch(`${apiBase}/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` }, body: JSON.stringify({ model, messages: msgs, temperature: 0.7, max_tokens: 2048 }) });
    const data = await resp.json();
    // Remove thinking indicator
    chatLines.pop();
    if (!resp.ok) { addLine(Y("  ✗ ") + (data.error?.message || `HTTP ${resp.status}`)); chatHistory.pop(); return; }
    const reply = (data.choices?.[0]?.message?.content || "").replace(/<think>[\s\S]*?<\/think>/g, "").trim();
    chatHistory.push({ role: "assistant", content: reply });
    addLine(RD(BX.tl + "── ") + R("SYNTHCLAW") + RD(" " + "─".repeat(30) + BX.tr));
    for (const line of (reply || "(empty)").split("\n")) addLine(RD(BX.v) + " " + line);
    addLine(RD(BX.bl + "─".repeat(42) + BX.br));
    addLine("");
  } catch (e) { chatLines.pop(); addLine(Y("  ✗ ") + e.message); chatHistory.pop(); }
}

async function handleCmd(input) {
  const [cmd, ...args] = input.split(" "); const arg = args.join(" ");
  switch (cmd) {
    case "/setup": return cmdSetup();
    case "/model": return cmdModel();
    case "/providers": return cmdProviders();
    case "/skills": await withInquirer(async () => { const { a } = await inquirer.prompt([{ type: "list", name: "a", message: "Skills:", choices: ["Install @user/skill", "List", "Remove"], prefix: PFX }]); if (a.startsWith("I")) { const { p } = await inquirer.prompt([{ type: "input", name: "p", message: "@user/skill:", prefix: PFX }]); if (p) addLine(RD("──") + R("•") + ` ${p} installed`); } }); return;
    case "/society": case "/agents":
      addLine(R("  AGENT SOCIETY"));
      addLine(RD("  ──•") + " " + RA("Orchestrator") + " " + D("plans + delegates"));
      addLine(RD("    ├─") + " " + RA("Researcher") + "  " + D("gathers info"));
      addLine(RD("    ├─") + " " + RA("Executor") + "    " + D("runs commands"));
      addLine(RD("    ├─") + " " + RA("Reviewer") + "    " + D("validates"));
      addLine(RD("    └─") + " " + RA("Observer") + "    " + D("monitors")); addLine(""); return;
    case "/clear": chatHistory.length = 0; chatLines = []; chatScroll = 0; renderChat(); renderInput(); addLine(D("  Cleared.")); return;
    case "/run": if (!arg) { await withInquirer(async () => { const { c } = await inquirer.prompt([{ type: "input", name: "c", message: "$", prefix: PFX }]); if (c) { try { const o = execSync(c, { encoding: "utf-8", timeout: 30000, cwd: getProjectRoot(), stdio: ["pipe", "pipe", "pipe"] }); for (const l of o.trim().split("\n").slice(0, 15)) addLine(D("  " + l)); } catch (e) { addLine(Y("  ✗ ") + (e.stderr || e.message || "").slice(0, 80)); } } }); return; } try { const o = execSync(arg, { encoding: "utf-8", timeout: 30000, cwd: getProjectRoot(), stdio: ["pipe", "pipe", "pipe"] }); for (const l of o.trim().split("\n").slice(0, 15)) addLine(D("  " + l)); } catch (e) { addLine(Y("  ✗ ") + (e.stderr || e.message || "").slice(0, 80)); } return;
    case "/quit": case "/exit": cleanup(); process.exit(0);
    case "/help": addLine(""); CMD_LIST.forEach(c => addLine("  " + RD("──•") + " " + R(c.value.padEnd(12)) + D(c.desc))); addLine(""); return;
    default: return sendMessage(input);
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  MAIN
// ══════════════════════════════════════════════════════════════════════════════

function cleanup() {
  if (blinkTimer) clearInterval(blinkTimer);
  showCursor();
  leaveAlt();
  if (process.stdin.isTTY && process.stdin.isRaw) process.stdin.setRawMode(false);
}

export async function runDashboard() {
  enterAlt();
  clearScreen();

  process.on("exit", cleanup);
  process.on("SIGINT", () => { cleanup(); process.exit(0); });

  // Blink mascot in header
  blinkTimer = setInterval(() => { blinkState = 1; renderHeader(); setTimeout(() => { blinkState = 0; renderHeader(); renderInput(); }, 150); }, 4000);

  fullRender();

  // Auto-setup if no key
  if (!config.get("openai_api_key")) {
    addLine(R("●") + " First run — configuring...");
    await cmdSetup();
  } else {
    addLine(D("Ready. Type / for commands, or chat. Ctrl+C to exit."));
    addLine("");
  }

  // Handle resize
  process.stdout.on("resize", () => { clearScreen(); fullRender(); });

  // ── RAW STDIN ──────────────────────────────────────────────────────────────
  if (!process.stdin.isTTY) {
    const { createInterface } = await import("readline");
    const rl = createInterface({ input: process.stdin, output: process.stdout });
    rl.on("line", async (l) => { const t = l.trim(); if (t) { addLine(D("> ") + t); if (t.startsWith("/")) await handleCmd(t); else await sendMessage(t); } });
    rl.on("close", () => { cleanup(); process.exit(0); });
    return;
  }

  process.stdin.setRawMode(true);
  process.stdin.resume();
  process.stdin.setEncoding("utf8");

  process.stdin.on("data", async (key) => {
    if (processing) return;

    // Ctrl+C / Ctrl+D
    if (key === "\x03" || key === "\x04") { cleanup(); process.exit(0); }

    // Enter
    if (key === "\r" || key === "\n") {
      if (cmdMenuActive) {
        const sel = cmdFiltered[cmdMenuIndex];
        hideMenu();
        inputBuffer = ""; inputCursor = 0;
        if (sel) { processing = true; addLine(D("> ") + sel.value); await handleCmd(sel.value); processing = false; }
        renderInput();
        return;
      }
      const line = inputBuffer.trim();
      inputBuffer = ""; inputCursor = 0;
      if (line) {
        hideMenu();
        processing = true;
        addLine(D("> ") + line);
        if (line.startsWith("/")) await handleCmd(line); else await sendMessage(line);
        processing = false;
      }
      renderInput();
      return;
    }

    // Escape
    if (key === "\x1b") { if (cmdMenuActive) hideMenu(); return; }

    // Arrow Up
    if (key === "\x1b[A") { if (cmdMenuActive) { cmdMenuIndex = (cmdMenuIndex - 1 + cmdFiltered.length) % cmdFiltered.length; renderCmdMenu(); renderInput(); } else if (chatScroll < chatLines.length - chatHeight()) { chatScroll++; renderChat(); renderInput(); } return; }
    // Arrow Down
    if (key === "\x1b[B") { if (cmdMenuActive) { cmdMenuIndex = (cmdMenuIndex + 1) % cmdFiltered.length; renderCmdMenu(); renderInput(); } else if (chatScroll > 0) { chatScroll--; renderChat(); renderInput(); } return; }
    // Arrow Left
    if (key === "\x1b[D") { if (inputCursor > 0) inputCursor--; renderInput(); return; }
    // Arrow Right
    if (key === "\x1b[C") { if (inputCursor < inputBuffer.length) inputCursor++; renderInput(); return; }

    // Backspace
    if (key === "\x7f" || key === "\b") {
      if (inputCursor > 0) { inputBuffer = inputBuffer.slice(0, inputCursor - 1) + inputBuffer.slice(inputCursor); inputCursor--; }
      if (inputBuffer.startsWith("/") && inputBuffer.length > 0) showMenu(inputBuffer);
      else if (cmdMenuActive) hideMenu();
      renderInput(); return;
    }

    // Tab — select from menu
    if (key === "\t") {
      if (cmdMenuActive && cmdFiltered.length) { inputBuffer = cmdFiltered[cmdMenuIndex].value; inputCursor = inputBuffer.length; hideMenu(); }
      renderInput(); return;
    }

    // Printable char
    if (key.length === 1 && key.charCodeAt(0) >= 32) {
      inputBuffer = inputBuffer.slice(0, inputCursor) + key + inputBuffer.slice(inputCursor);
      inputCursor++;
      if (inputBuffer === "/") showMenu("/");
      else if (inputBuffer.startsWith("/") && !inputBuffer.includes(" ")) showMenu(inputBuffer);
      else if (cmdMenuActive && !inputBuffer.startsWith("/")) hideMenu();
      renderInput(); return;
    }
  });
}

export { cmdSetup as runInlineWizard };
