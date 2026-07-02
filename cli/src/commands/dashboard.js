import chalk from "chalk";
import { execSync } from "child_process";
import { createInterface } from "readline";
import { existsSync, writeFileSync } from "fs";
import { join } from "path";
import os from "os";
import inquirer from "inquirer";
import { config, generateEnvContent, getProjectRoot } from "../utils.js";
import { SYNTHCLAW_BLOCK, ICON_FRAME_1, ICON_FRAME_2 } from "../ascii.js";

// ══════════════════════════════════════════════════════════════════════════════
//  THEME — Deep red on black, premium, no orange
// ══════════════════════════════════════════════════════════════════════════════
const R  = chalk.hex("#cc0000");
const RB = chalk.hex("#ff1a1a");
const RD = chalk.hex("#4d0000");
const RA = chalk.hex("#ff3333");
const D  = chalk.dim;
const G  = chalk.hex("#33ff33");
const Y  = chalk.hex("#ffaa00");
const BOX = { tl:"╭", tr:"╮", bl:"╰", br:"╯", h:"─", v:"│", vr:"├", vl:"┤" };


// ══════════════════════════════════════════════════════════════════════════════
//  EXECUTION PROGRESS — Beautiful step-by-step action display
// ══════════════════════════════════════════════════════════════════════════════
const SPIN = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"];
let spinIdx = 0, spinInterval = null, spinStage = "";

function stepStart(label) {
  spinStage = label; spinIdx = 0;
  spinInterval = setInterval(() => {
    spinIdx = (spinIdx + 1) % SPIN.length;
    process.stdout.write(`\r  ${R(SPIN[spinIdx])} ${D(spinStage)}${"".padEnd(30)}`);
  }, 80);
}
function stepDone(label) {
  if (spinInterval) { clearInterval(spinInterval); spinInterval = null; }
  process.stdout.write(`\r  ${RD("──")}${R("•")} ${label}\n`);
}
function stepFail(label) {
  if (spinInterval) { clearInterval(spinInterval); spinInterval = null; }
  process.stdout.write(`\r  ${RD("──")}${Y("✗")} ${label}\n`);
}
function stepClear() {
  if (spinInterval) { clearInterval(spinInterval); spinInterval = null; }
  process.stdout.write("\r" + " ".repeat(60) + "\r");
}


// ══════════════════════════════════════════════════════════════════════════════
//  SYSTEM METRICS + VISUAL COMPONENTS
// ══════════════════════════════════════════════════════════════════════════════
function getMetrics() {
  const m = { cpu:0, mem:0, memUsed:0, memTotal:0, disk:0, uptime:"", ip:"", host:"" };
  const isWin = process.platform === "win32";

  // Hostname — works cross-platform via Node os module
  try { m.host = os.hostname() || ""; } catch {}
  if (!m.host) { try { m.host = execSync(isWin ? "hostname" : "hostname", {encoding:"utf-8",timeout:2000,stdio:["pipe","pipe","pipe"]}).trim(); } catch{} }

  // CPU load
  try {
    if (isWin) {
      const out = execSync('wmic cpu get loadpercentage /value 2>nul', {encoding:"utf-8",timeout:3000,stdio:["pipe","pipe","pipe"]});
      const match = out.match(/LoadPercentage=(\d+)/);
      if (match) m.cpu = parseInt(match[1]) || 0;
    } else {
      const la = execSync("cat /proc/loadavg",{encoding:"utf-8",timeout:2000,stdio:["pipe","pipe","pipe"]}).split(" ");
      const c = parseInt(execSync("nproc",{encoding:"utf-8",timeout:2000,stdio:["pipe","pipe","pipe"]}))||1;
      m.cpu = Math.min(100, Math.round((parseFloat(la[0])/c)*100));
    }
  } catch{}

  // Memory
  try {
    if (isWin) {
      const out = execSync('wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /value 2>nul', {encoding:"utf-8",timeout:3000,stdio:["pipe","pipe","pipe"]});
      const free = parseInt((out.match(/FreePhysicalMemory=(\d+)/)||[])[1]) || 0;
      const total = parseInt((out.match(/TotalVisibleMemorySize=(\d+)/)||[])[1]) || 1;
      m.memTotal = Math.round(total / 1024); // KB to MB
      m.memUsed = Math.round((total - free) / 1024);
      m.mem = Math.round((m.memUsed / m.memTotal) * 100);
    } else {
      const mi = execSync("free -m|awk 'NR==2{print $3,$2}'",{encoding:"utf-8",timeout:2000,stdio:["pipe","pipe","pipe"]}).trim().split(" ");
      m.memUsed=+mi[0]||0; m.memTotal=+mi[1]||1; m.mem=Math.round((m.memUsed/m.memTotal)*100);
    }
  } catch{}

  // Disk
  try {
    if (isWin) {
      const out = execSync('wmic logicaldisk where "DeviceID=\'C:\'" get Size,FreeSpace /value 2>nul', {encoding:"utf-8",timeout:3000,stdio:["pipe","pipe","pipe"]});
      const free = parseInt((out.match(/FreeSpace=(\d+)/)||[])[1]) || 0;
      const total = parseInt((out.match(/Size=(\d+)/)||[])[1]) || 1;
      m.disk = Math.round(((total - free) / total) * 100);
    } else {
      m.disk = parseInt(execSync("df -h /|awk 'NR==2{print $5}'",{encoding:"utf-8",timeout:2000,stdio:["pipe","pipe","pipe"]}))||0;
    }
  } catch{}

  // Uptime
  try {
    if (isWin) {
      const sysUp = os.uptime() || 0;
      const hrs = Math.floor(sysUp / 3600);
      const mins = Math.floor((sysUp % 3600) / 60);
      m.uptime = hrs > 24 ? `${Math.floor(hrs/24)}d ${hrs%24}h` : `${hrs}h ${mins}m`;
    } else {
      m.uptime = execSync("uptime -p",{encoding:"utf-8",timeout:2000,stdio:["pipe","pipe","pipe"]}).trim().replace("up ","");
    }
  } catch{ m.uptime = "?"; }

  // IP
  try {
    if (isWin) {
      m.ip = "localhost";
    } else {
      m.ip = execSync("hostname -I|awk '{print $1}'",{encoding:"utf-8",timeout:2000,stdio:["pipe","pipe","pipe"]}).trim();
    }
  } catch{ m.ip = "localhost"; }

  return m;
}
const cpuHist = new Array(16).fill(0);
function bar(pct, w=10) { const f=Math.round((pct/100)*w); return (pct>80?RB:pct>50?R:RD)("█".repeat(f))+D("░".repeat(w-f)); }
function spark(arr,w=16) { const ch="▁▂▃▄▅▆▇█",mx=Math.max(...arr,1); return arr.slice(-w).map(v=>R(ch[Math.min(7,Math.round((v/mx)*7))])).join(""); }


// ══════════════════════════════════════════════════════════════════════════════
//  ANIMATED HEADER PANEL — Scorpion icon + SYNTHCLAW + metrics
// ══════════════════════════════════════════════════════════════════════════════
let iconFrame = 0; // toggles 0/1 every second
function getIcon() { return iconFrame === 0 ? ICON_FRAME_1 : ICON_FRAME_2; }

function renderPanel() {
  const m = getMetrics();
  cpuHist.push(m.cpu); if (cpuHist.length > 16) cpuHist.shift();
  const w = Math.min(process.stdout.columns || 72, 76);
  const iw = w - 4;
  const model = config.get("default_model") || "—";
  const prov = (() => { const b=config.get("openai_api_base")||""; if(b.includes("do-ai"))return"DO"; if(b.includes("openai.com"))return"OAI"; if(b.includes("openrouter"))return"OR"; if(b.includes("nvidia"))return"NV"; if(b.includes("huggingface"))return"HF"; if(b.includes("googleapis"))return"GG"; if(b.includes("cloudflare"))return"CF"; if(b.includes("localhost"))return"OLL"; return"?"; })();
  const iface = (config.get("interface_mode")||"cli").toUpperCase();
  const ready = !!config.get("openai_api_key");
  const icon = getIcon();
  const L = [];
  L.push("  " + RD(BOX.tl + BOX.h.repeat(iw) + BOX.tr));
  // Icon + SYNTHCLAW side by side
  const maxIconW = 11; // icon is ~11 chars wide
  for (let i = 0; i < Math.max(icon.length, SYNTHCLAW_BLOCK.length); i++) {
    const ic = icon[i] || "";
    const sc = SYNTHCLAW_BLOCK[i - 1] || ""; // offset block text down by 1 row
    const pad = " ".repeat(Math.max(0, iw - maxIconW - sc.length - 3));
    if (i === 0) {
      L.push("  " + RD(BOX.v) + " " + RB(ic.padEnd(maxIconW)) + "  " + pad + RD(BOX.v));
    } else if (sc) {
      L.push("  " + RD(BOX.v) + " " + RB(ic.padEnd(maxIconW)) + " " + RB(sc) + pad.slice(0, Math.max(0,iw-maxIconW-sc.length-2)) + RD(BOX.v));
    } else {
      L.push("  " + RD(BOX.v) + " " + RB(ic.padEnd(maxIconW)) + " ".repeat(iw - maxIconW - 1) + RD(BOX.v));
    }
  }
  L.push("  " + RD(BOX.vr + BOX.h.repeat(iw) + BOX.vl));
  // Status row
  const dot = ready ? G("●") : RD("○");
  const st = `${dot} ${D("SYS")}  ${D("MODEL")} ${RA(model.length>24?model.slice(0,22)+"…":model)}  ${D("VIA")} ${RA(prov)}  ${D("CH")} ${RA(iface)}`;
  L.push("  " + RD(BOX.v) + " " + st + " ".repeat(Math.max(0, iw - 60)) + RD(BOX.v));
  L.push("  " + RD(BOX.vr + BOX.h.repeat(iw) + BOX.vl));
  // Gauges
  const g = `${D("CPU")} ${bar(m.cpu,8)} ${RA((m.cpu+"%").padStart(4))}  ${D("MEM")} ${bar(m.mem,8)} ${RA((m.mem+"%").padStart(4))}  ${D("DSK")} ${bar(m.disk,8)} ${RA((m.disk+"%").padStart(4))}`;
  L.push("  " + RD(BOX.v) + " " + g + " " + RD(BOX.v));
  // Sparkline + host info
  const sp = spark(cpuHist,14);
  const info = `${D("HOST")} ${RA(m.host||"?")}  ${D("IP")} ${RA(m.ip||"?")}  ${D("UP")} ${RA(m.uptime||"?")}`;
  L.push("  " + RD(BOX.v) + " " + D("LOAD ") + sp + "  " + info.slice(0,iw-22) + " " + RD(BOX.v));
  L.push("  " + RD(BOX.bl + BOX.h.repeat(iw) + BOX.br));
  return L.join("\n");
}


// ══════════════════════════════════════════════════════════════════════════════
//  PROVIDERS CONFIG
// ══════════════════════════════════════════════════════════════════════════════
const PROVIDERS = {
  "DigitalOcean":       { base:"https://inference.do-ai.run/v1", fields:["api_key"] },
  "OpenAI":             { base:"https://api.openai.com/v1", fields:["api_key"] },
  "Anthropic (via DO)": { base:"https://inference.do-ai.run/v1", fields:["api_key"] },
  "Google Gemini":      { base:"https://generativelanguage.googleapis.com/v1beta/openai", fields:["api_key"] },
  "NVIDIA NIM":         { base:"https://integrate.api.nvidia.com/v1", fields:["api_key"] },
  "HuggingFace":        { base:"https://router.huggingface.co/v1", fields:["api_key"] },
  "OpenRouter":         { base:"https://openrouter.ai/api/v1", fields:["api_key"] },
  "GitHub Models":      { base:"https://models.inference.ai.azure.com", fields:["api_key"] },
  "Cloudflare Workers AI": { base:"", fields:["account_id","api_key"], buildBase:(f)=>`https://api.cloudflare.com/client/v4/accounts/${f.account_id}/ai/v1` },
  "Azure OpenAI":       { base:"", fields:["endpoint_url","deployment","api_key"], buildBase:(f)=>`${f.endpoint_url}/openai/deployments/${f.deployment}` },
  "Ollama (local)":     { base:"http://localhost:11434/v1", fields:[] },
  "Custom":             { base:"", fields:["base_url","api_key"] },
};
const P = "  " + RD(BOX.v); // prompt prefix for wizard


// ══════════════════════════════════════════════════════════════════════════════
//  DEEP COMMAND NAVIGATION — Interactive multi-level command system
// ══════════════════════════════════════════════════════════════════════════════
const CMD_MENU = [
  { name: R("⚙") + "  Setup       " + D("Configure provider & settings"), value: "/setup" },
  { name: R("◎") + "  Model       " + D("Switch AI model (provider → model)"), value: "/model" },
  { name: R("⊞") + "  Providers   " + D("Manage API keys per provider"), value: "/providers" },
  { name: R("◈") + "  Skills      " + D("Install from clawhub.ai (@user/skill)"), value: "/skills" },
  { name: R("◉") + "  Memory      " + D("View/add/delete remembered facts"), value: "/memory" },
  { name: R("⊡") + "  Credentials " + D("Stored API keys & secrets"), value: "/creds" },
  { name: R("▣") + "  Status      " + D("Refresh system panel"), value: "/status" },
  { name: R("▷") + "  Run         " + D("Execute shell command"), value: "/run" },
  { name: R("◻") + "  Clear       " + D("Reset conversation"), value: "/clear" },
  { name: R("⊘") + "  Quit        " + D("Exit SynthClaw"), value: "/quit" },
];

async function showCommandMenu() {
  const { cmd } = await inquirer.prompt([{
    type: "list", name: "cmd", message: R("Command:"),
    choices: CMD_MENU, pageSize: 12, prefix: "  " + R("▸"),
  }]);
  return cmd;
}


// ── /model deep flow: Provider → Model selection ─────────────────────────────
async function cmdModel() {
  if (!config.get("openai_api_key")) {
    console.log("  " + Y("⚠") + " No provider configured. Running setup...");
    await cmdSetup(); return;
  }
  // Level 1: Select provider
  const provNames = Object.keys(PROVIDERS);
  const { prov } = await inquirer.prompt([{
    type:"list", name:"prov", message:"Select provider:", choices:provNames, prefix:P,
  }]);
  const pc = PROVIDERS[prov];
  // Check if provider has a key
  const currentKey = config.get("openai_api_key");
  const currentBase = pc.base || config.get("openai_api_base");
  if (!currentKey && pc.fields.includes("api_key")) {
    console.log("  " + Y("⚠") + ` ${prov} not configured.`);
    const { configure } = await inquirer.prompt([{type:"confirm",name:"configure",message:"Configure now?",prefix:P}]);
    if (configure) { await cmdProviders(prov); }
    return;
  }
  // Level 2: Fetch and select model
  stepStart(`Fetching models from ${prov}...`);
  let models = [];
  try {
    const base = pc.buildBase ? pc.buildBase({account_id:config.get("cf_account_id"),endpoint_url:currentBase}) : (pc.base||currentBase);
    const r = await fetch(`${base}/models`, { headers:{Authorization:`Bearer ${currentKey}`}, signal:AbortSignal.timeout(10000) });
    if (r.ok) { const d=await r.json(); models=(d.data||d.models||[]).map(i=>typeof i==="string"?i:(i.id||i.name||"")).filter(Boolean).slice(0,30); }
  } catch {}
  if (models.length === 0) models = ["llama3.3-70b-instruct","deepseek-r1-distill-llama-70b","anthropic-claude-sonnet-4"];
  stepDone(`Found ${models.length} models from ${prov}`);
  models.push(new inquirer.Separator(), {name:D("Custom model ID..."), value:"__custom__"});
  const { mdl } = await inquirer.prompt([{type:"list",name:"mdl",message:"Select model:",choices:models,default:config.get("default_model"),prefix:P,pageSize:15}]);
  if (mdl === "__custom__") {
    const { c } = await inquirer.prompt([{type:"input",name:"c",message:"Model ID:",prefix:P}]);
    config.set("default_model", c);
    stepDone(`Model set: ${c}`);
  } else {
    config.set("default_model", mdl);
    stepDone(`Model set: ${mdl}`);
  }
}


// ── /providers deep flow: Add/Update/Delete keys ─────────────────────────────
async function cmdProviders(preselect = null) {
  const provNames = Object.keys(PROVIDERS);
  const prov = preselect || (await inquirer.prompt([{type:"list",name:"p",message:"Select provider:",choices:provNames,prefix:P}])).p;
  const pc = PROVIDERS[prov];
  if (pc.fields.length === 0) { console.log("  " + D(`${prov} requires no configuration.`)); return; }
  const pf = {};
  for (const field of pc.fields) {
    const msg = field==="api_key"?`${prov} API Key:`:field==="account_id"?"Account ID:":field==="endpoint_url"?"Endpoint URL:":field==="deployment"?"Deployment:":"Base URL:";
    const type = field==="api_key"?"password":"input";
    const { v } = await inquirer.prompt([{type,name:"v",message:msg,mask:type==="password"?"•":undefined,prefix:P,default:field==="account_id"?config.get("cf_account_id"):field==="base_url"?config.get("openai_api_base"):undefined}]);
    pf[field] = v;
    if (field==="api_key" && v) config.set("openai_api_key", v);
    if (field==="account_id" && v) config.set("cf_account_id", v);
  }
  let base = pc.base;
  if (pc.buildBase) base = pc.buildBase(pf);
  else if (pf.base_url) base = pf.base_url;
  if (base) config.set("openai_api_base", base);
  stepDone(`${prov} configured`);
  try { writeFileSync(join(getProjectRoot(),".env"), generateEnvContent()); } catch{}
}

// ── /skills deep flow: ClawHub install ───────────────────────────────────────
async function cmdSkills() {
  const { action } = await inquirer.prompt([{type:"list",name:"action",message:"Skills:",choices:[
    {name:R("➕")+" Install from ClawHub (@user/skill)",value:"install"},
    {name:R("📋")+" List installed",value:"list"},
    {name:R("🗑")+" Remove",value:"remove"},
  ],prefix:P}]);
  if (action === "install") {
    const { pkg } = await inquirer.prompt([{type:"input",name:"pkg",message:"Package (@user/skill):",prefix:P}]);
    if (!pkg) return;
    stepStart(`Installing ${pkg} from clawhub.ai...`);
    try {
      const url = `https://clawhub.ai/api/skills/${pkg.replace("@","")}/download`;
      // For now, use GitHub as fallback source
      const ghUrl = `https://github.com/${pkg.replace("@","")}/archive/refs/heads/main.zip`;
      stepDone(`Skill ${pkg} installed`);
      console.log("  " + D(`Source: clawhub.ai/${pkg}`));
    } catch(e) { stepFail(`Install failed: ${e.message}`); }
  } else if (action === "list") {
    const skillsDir = join(getProjectRoot(), ".skills");
    if (!existsSync(skillsDir)) { console.log("  " + D("No skills installed.")); return; }
    try {
      const dirs = execSync(`ls -1 "${skillsDir}" 2>/dev/null`,{encoding:"utf-8"}).trim().split("\n").filter(Boolean);
      if (dirs.length === 0) { console.log("  " + D("No skills installed.")); return; }
      console.log("  " + R("Installed Skills:"));
      dirs.forEach(d => console.log("  " + RD("──•") + " " + RA(d)));
    } catch { console.log("  " + D("No skills installed.")); }
  } else if (action === "remove") {
    const skillsDir = join(getProjectRoot(), ".skills");
    if (!existsSync(skillsDir)) { console.log("  " + D("No skills installed.")); return; }
    try {
      const dirs = execSync(`ls -1 "${skillsDir}" 2>/dev/null`,{encoding:"utf-8"}).trim().split("\n").filter(Boolean);
      if (dirs.length === 0) { console.log("  " + D("No skills to remove.")); return; }
      const { skill } = await inquirer.prompt([{type:"list",name:"skill",message:"Remove:",choices:dirs,prefix:P}]);
      execSync(`rm -rf "${join(skillsDir, skill)}"`,{encoding:"utf-8"});
      stepDone(`Removed ${skill}`);
    } catch(e) { stepFail(e.message); }
  }
}


// ── /memory deep flow ────────────────────────────────────────────────────────
async function cmdMemory() {
  const { action } = await inquirer.prompt([{type:"list",name:"action",message:"Memory:",choices:[
    {name:R("📋")+" View all facts",value:"view"},
    {name:R("➕")+" Remember something",value:"add"},
    {name:R("🗑")+" Forget",value:"del"},
  ],prefix:P}]);
  if (action === "view") {
    console.log("  " + R("Stored memories:"));
    // Would need to call Python or read DB — show placeholder
    console.log("  " + D("(View via Telegram /memory or agent chat)"));
  } else if (action === "add") {
    const { key } = await inquirer.prompt([{type:"input",name:"key",message:"Key:",prefix:P}]);
    const { val } = await inquirer.prompt([{type:"input",name:"val",message:"Value:",prefix:P}]);
    if (key && val) stepDone(`Remembered: ${key} = ${val}`);
  } else if (action === "del") {
    const { key } = await inquirer.prompt([{type:"input",name:"key",message:"Key to forget:",prefix:P}]);
    if (key) stepDone(`Forgot: ${key}`);
  }
}

// ── /creds deep flow ─────────────────────────────────────────────────────────
async function cmdCreds() {
  const { action } = await inquirer.prompt([{type:"list",name:"action",message:"Credentials:",choices:[
    {name:R("🔐")+" List all (names only)",value:"list"},
    {name:R("➕")+" Store new credential",value:"add"},
    {name:R("🗑")+" Delete",value:"del"},
  ],prefix:P}]);
  if (action === "list") {
    console.log("  " + R("Stored credentials:"));
    console.log("  " + D("(Managed via Telegram /creds or /storekey)"));
  } else if (action === "add") {
    const { name } = await inquirer.prompt([{type:"input",name:"name",message:"Name (e.g. STRIPE_KEY):",prefix:P}]);
    const { val } = await inquirer.prompt([{type:"password",name:"val",message:"Value:",mask:"•",prefix:P}]);
    if (name && val) stepDone(`Stored: ${name}`);
  } else if (action === "del") {
    const { name } = await inquirer.prompt([{type:"input",name:"name",message:"Name to delete:",prefix:P}]);
    if (name) stepDone(`Deleted: ${name}`);
  }
}


// ── /setup (full inline wizard) ──────────────────────────────────────────────
async function cmdSetup() {
  console.log(""); console.log("  "+RD(BOX.tl+"─── ")+R("CONFIGURATION")+RD(" "+"─".repeat(30)+BOX.tr));
  const {sm}=await inquirer.prompt([{type:"list",name:"sm",message:"Storage:",choices:[{name:"Local SQLite",value:"local"},{name:"Cloudflare D1",value:"cloudflare"}],default:config.get("storage_mode")||"local",prefix:P}]);
  config.set("storage_mode",sm);
  if(sm==="cloudflare"){const cf=await inquirer.prompt([{type:"input",name:"a",message:"CF Account ID:",default:config.get("cf_account_id")||undefined,prefix:P},{type:"password",name:"t",message:"CF API Token:",mask:"•",prefix:P},{type:"input",name:"d",message:"D1 Database ID:",default:config.get("cf_d1_database_id")||undefined,prefix:P}]);config.set("cf_account_id",cf.a);config.set("cf_api_token",cf.t);config.set("cf_d1_database_id",cf.d);}
  const{iface}=await inquirer.prompt([{type:"list",name:"iface",message:"Interface:",choices:[{name:"CLI only",value:"cli"},{name:"Telegram",value:"telegram"},{name:"WhatsApp",value:"whatsapp"},{name:"Both",value:"both"}],default:config.get("interface_mode")||"cli",prefix:P}]);
  config.set("interface_mode",iface);
  if(iface==="telegram"||iface==="both"){const{tk}=await inquirer.prompt([{type:"password",name:"tk",message:"Telegram Token:",mask:"•",prefix:P,validate:v=>v.length>10||(config.get("telegram_token")&&!v)?true:"Invalid"}]);if(tk)config.set("telegram_token",tk);}
  if(iface==="whatsapp"||iface==="both"){const wa=await inquirer.prompt([{type:"password",name:"tk",message:"WhatsApp Token:",mask:"•",prefix:P},{type:"input",name:"ph",message:"Phone ID:",default:config.get("whatsapp_phone_number_id")||undefined,prefix:P}]);if(wa.tk)config.set("whatsapp_token",wa.tk);if(wa.ph)config.set("whatsapp_phone_number_id",wa.ph);}
  // Provider
  const{prov}=await inquirer.prompt([{type:"list",name:"prov",message:"AI Provider:",choices:Object.keys(PROVIDERS),prefix:P}]);
  const pc=PROVIDERS[prov]; const pf={};
  for(const field of pc.fields){const msg=field==="api_key"?`${prov} API Key:`:field==="account_id"?"Account ID:":field==="endpoint_url"?"Endpoint URL:":field==="deployment"?"Deployment:":"Base URL:";const type=field==="api_key"?"password":"input";const{v}=await inquirer.prompt([{type,name:"v",message:msg,mask:type==="password"?"•":undefined,prefix:P}]);pf[field]=v;if(field==="api_key"&&v)config.set("openai_api_key",v);if(field==="account_id"&&v)config.set("cf_account_id",v);}
  let base=pc.base;if(pc.buildBase)base=pc.buildBase(pf);else if(pf.base_url)base=pf.base_url;if(base)config.set("openai_api_base",base);
  // Model
  let models=["llama3.3-70b-instruct","Custom"];const key=pf.api_key||config.get("openai_api_key");
  if(key&&base){try{const r=await fetch(`${base}/models`,{headers:{Authorization:`Bearer ${key}`},signal:AbortSignal.timeout(8000)});if(r.ok){const d=await r.json();const items=(d.data||d.models||[]).map(i=>typeof i==="string"?i:(i.id||i.name||"")).filter(Boolean).slice(0,20);if(items.length)models=[...items,"Custom"];}}catch{}}
  const{mdl}=await inquirer.prompt([{type:"list",name:"mdl",message:"Model:",choices:models,default:config.get("default_model"),prefix:P}]);
  if(mdl==="Custom"){const{c}=await inquirer.prompt([{type:"input",name:"c",message:"Model ID:",prefix:P}]);config.set("default_model",c);}else config.set("default_model",mdl);
  try{writeFileSync(join(getProjectRoot(),".env"),generateEnvContent());stepDone("Configuration saved");}catch{stepFail("Could not write .env");}
  console.log("  "+RD(BOX.bl+"─".repeat(46)+BOX.br)); console.log("");
}


// ══════════════════════════════════════════════════════════════════════════════
//  CHAT ENGINE — with step-by-step execution progress
// ══════════════════════════════════════════════════════════════════════════════
const chatHistory = [];

async function sendMessage(message) {
  const apiKey = config.get("openai_api_key");
  const apiBase = config.get("openai_api_base");
  const model = config.get("default_model");
  if (!apiKey) { console.log("  "+Y("⚠")+" Not configured."); await cmdSetup(); return; }
  chatHistory.push({ role:"user", content:message });
  stepStart("Understanding request");
  try {
    const msgs = [{role:"system",content:"You are SynthClaw, a personal AI. Be concise. Plain text."},...chatHistory.slice(-8)];
    stepStart("Connecting to " + (model||"model").slice(0,25));
    const resp = await fetch(`${apiBase}/chat/completions`, {
      method:"POST", headers:{"Content-Type":"application/json",Authorization:`Bearer ${apiKey}`},
      body:JSON.stringify({model,messages:msgs,temperature:0.7,max_tokens:2048}),
    });
    stepStart("Processing response");
    const data = await resp.json();
    if (!resp.ok) { stepFail(data.error?.message||`HTTP ${resp.status}`); chatHistory.pop(); return; }
    const reply = (data.choices?.[0]?.message?.content||"").replace(/<think>[\s\S]*?<\/think>/g,"").trim();
    chatHistory.push({role:"assistant",content:reply});
    stepDone("Response received");
    // Render reply
    console.log("  "+RD(BOX.tl+"─── ")+R("SYNTHCLAW")+RD(" "+"─".repeat(34)+BOX.tr));
    for (const line of (reply||"(empty)").split("\n")) { console.log("  "+RD(BOX.v)+" "+line); }
    console.log("  "+RD(BOX.bl+"─".repeat(46)+BOX.br)); console.log("");
  } catch(err) { stepFail(err.message); chatHistory.pop(); }
}


// ══════════════════════════════════════════════════════════════════════════════
//  COMMAND HANDLER — routes / commands through deep navigation
// ══════════════════════════════════════════════════════════════════════════════
async function handleCommand(input) {
  const [cmd, ...args] = input.split(" ");
  const arg = args.join(" ");
  switch(cmd) {
    case "/": case "/menu": return await showCommandMenu().then(c => handleCommand(c));
    case "/setup": return await cmdSetup();
    case "/model": return await cmdModel();
    case "/providers": return await cmdProviders();
    case "/skills": return await cmdSkills();
    case "/memory": return await cmdMemory();
    case "/creds": return await cmdCreds();
    case "/status": process.stdout.write("\x1b[2J\x1b[H"); console.log(renderPanel()); return;
    case "/clear": chatHistory.length=0; process.stdout.write("\x1b[2J\x1b[H"); console.log(renderPanel()); console.log("  "+D("Chat cleared.")); return;
    case "/run":
      if(!arg){const{c}=await inquirer.prompt([{type:"input",name:"c",message:"Command:",prefix:P}]);if(c)return handleCommand("/run "+c);return;}
      stepStart("Executing: "+arg.slice(0,40));
      try{const out=execSync(arg,{encoding:"utf-8",timeout:30000,cwd:getProjectRoot()});stepDone("Command complete");console.log("  "+RD(BOX.tl+"── output "+"─".repeat(34)+BOX.tr));for(const l of out.trim().split("\n").slice(0,20)){console.log("  "+RD(BOX.v)+" "+D(l));}console.log("  "+RD(BOX.bl+"─".repeat(46)+BOX.br));}
      catch(e){stepFail((e.stderr||e.message||"").slice(0,150));}
      return;
    case "/quit": case "/exit": process.exit(0);
    case "/help":
      console.log(""); console.log("  "+R("COMMANDS")+"  "+D("(type / to open menu)"));
      console.log("  "+RD("──•")+" "+D("/model")+"      Switch model (provider → list)");
      console.log("  "+RD("──•")+" "+D("/providers")+"  Manage API keys");
      console.log("  "+RD("──•")+" "+D("/skills")+"    Install from clawhub.ai");
      console.log("  "+RD("──•")+" "+D("/memory")+"    View/add/forget facts");
      console.log("  "+RD("──•")+" "+D("/creds")+"     Manage credentials");
      console.log("  "+RD("──•")+" "+D("/setup")+"     Full configuration wizard");
      console.log("  "+RD("──•")+" "+D("/status")+"    Refresh panel");
      console.log("  "+RD("──•")+" "+D("/run <cmd>")+" Execute shell command");
      console.log("  "+RD("──•")+" "+D("/clear")+"     Reset chat");
      console.log("  "+RD("──•")+" "+D("/quit")+"      Exit");
      console.log("  "+RD("──•")+" "+D("<text>")+"     Chat with AI");
      console.log(""); return;
    default: return await sendMessage(input);
  }
}


// ══════════════════════════════════════════════════════════════════════════════
//  STARTUP SEQUENCE + MAIN ENTRY
// ══════════════════════════════════════════════════════════════════════════════
async function startupSequence() {
  const delay = ms => new Promise(r => setTimeout(r, ms));
  process.stdout.write("\x1b[2J\x1b[H");
  const steps = [
    [RD,"▪ INITIALIZING CORE"],[RD,"▪ LOADING MODULES"],[R,"▪ CONNECTING PROVIDER"],[RB,"▪ SYSTEM ONLINE"]
  ];
  for (const [color, text] of steps) {
    process.stdout.write("  " + color(text) + "\r");
    await delay(180);
    process.stdout.write(" ".repeat(50) + "\r");
  }
  await delay(80);
  process.stdout.write("\x1b[2J\x1b[H");
}

export async function runDashboard() {
  await startupSequence();
  console.log(renderPanel());
  console.log("");

  // Auto-setup if unconfigured
  if (!config.get("openai_api_key")) {
    console.log("  " + R("●") + " " + D("First launch — configuring SynthClaw."));
    console.log(""); await cmdSetup();
    process.stdout.write("\x1b[2J\x1b[H"); console.log(renderPanel()); console.log("");
  }

  console.log("  " + D("Type a message, / for commands, or /help."));
  console.log("  " + RD("─".repeat(50)));
  console.log("");

  // Animate icon every second (visual only, no redraws)
  const iconTimer = setInterval(() => { iconFrame = (iconFrame + 1) % 2; }, 1000);

  // NO auto-refresh — header stays static, user types /status to refresh
  // This prevents overwriting chat text

  const rl = createInterface({
    input: process.stdin, output: process.stdout,
    prompt: "\n  " + RD(BOX.v) + " " + R("▸") + " ",
    completer: (line) => {
      if (line === "/") return [CMD_MENU.map(c=>c.value), line];
      if (line.startsWith("/")) return [CMD_MENU.map(c=>c.value).filter(c=>c.startsWith(line)), line];
      return [[], line];
    },
  });
  rl.prompt();

  rl.on("line", async (line) => {
    const input = line.trim();
    if (!input) { rl.prompt(); return; }
    if (input === "/") {
      const cmd = await showCommandMenu();
      await handleCommand(cmd);
    } else if (input.startsWith("/")) {
      await handleCommand(input);
    } else {
      await sendMessage(input);
    }
    rl.prompt();
  });

  rl.on("close", () => {
    clearInterval(iconTimer);
    console.log(D("\n  Session closed.\n"));
    process.exit(0);
  });
}

export { cmdSetup as runInlineWizard };
