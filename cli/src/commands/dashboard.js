import chalk from "chalk";
import { execSync } from "child_process";
import { existsSync, writeFileSync } from "fs";
import { join } from "path";
import os from "os";
import inquirer from "inquirer";
import { config, generateEnvContent, getProjectRoot } from "../utils.js";
import { MASCOT_OPEN, MASCOT_BLINK, WORDMARK, SUBTITLE } from "../ascii.js";

// ── THEME ────────────────────────────────────────────────────────────────────
const R=chalk.hex("#cc0000"),RB=chalk.hex("#ff1a1a"),RD=chalk.hex("#e85d04");
const RA=chalk.hex("#ff3333"),D=chalk.dim,G=chalk.hex("#33ff33"),Y=chalk.hex("#ffaa00");
const BX={tl:"╭",tr:"╮",bl:"╰",br:"╯",h:"─",v:"│",vr:"├",vl:"┤"};
const isWin=process.platform==="win32";

// ── ANSI TERMINAL CONTROL ────────────────────────────────────────────────────
const ESC = "\x1b[";
const supportsAltScreen = (() => {
  if (!process.stdout.isTTY) return false;
  if (isWin) {
    if (process.env.WT_SESSION || process.env.TERM_PROGRAM || process.env.TERMINAL_EMULATOR) return true;
    if (process.env.ConEmuPID || process.env.CMDER_ROOT) return true;
    return false;
  }
  return true;
})();

const enterAltScreen = () => { if (supportsAltScreen) process.stdout.write("\x1b[?1049h"); };
const leaveAltScreen = () => { if (supportsAltScreen) process.stdout.write("\x1b[?1049l"); };
const hideCursor = () => process.stdout.write(ESC + "?25l");
const showCursor = () => process.stdout.write(ESC + "?25h");
const moveTo = (row, col) => process.stdout.write(`${ESC}${row};${col}H`);
const clearLine = () => process.stdout.write(ESC + "2K");
const clearToEnd = () => process.stdout.write(ESC + "0J");
const saveCursor = () => process.stdout.write(ESC + "s");
const restoreCursor = () => process.stdout.write(ESC + "u");
const setScrollRegion = (top, bottom) => process.stdout.write(`${ESC}${top};${bottom}r`);
const resetScrollRegion = () => process.stdout.write(ESC + "r");
const tRows = () => process.stdout.rows || 24;
const tCols = () => process.stdout.columns || 80;



// ── HEADER ───────────────────────────────────────────────────────────────────
const HEADER_H = 11;
let blinkState = 0, blinkTimer = null;

function startBlink() {
  blinkTimer = setInterval(() => {
    blinkState = 1;
    setTimeout(() => { blinkState = 0; drawHeader(); }, 150);
    drawHeader();
  }, 3000);
}

function drawHeader() {
  const m = getMetrics();
  const w = Math.min(tCols(), 76), iw = w - 4;
  const mascot = blinkState === 0 ? MASCOT_OPEN : MASCOT_BLINK;
  const model = config.get("default_model") || "\u2014";
  const prov = (() => { const b=config.get("openai_api_base")||""; if(b.includes("do-ai"))return"DO"; if(b.includes("openai.com"))return"OAI"; if(b.includes("openrouter"))return"OR"; if(b.includes("nvidia"))return"NV"; if(b.includes("huggingface"))return"HF"; if(b.includes("googleapis"))return"GG"; if(b.includes("cloudflare"))return"CF"; if(b.includes("dashscope"))return"QW"; if(b.includes("localhost"))return"OLL"; return"?"; })();
  const ready = !!config.get("openai_api_key");
  const lines = [];
  lines.push(RD(BX.tl + BX.h.repeat(iw) + BX.tr));
  for (let i = 0; i < 5; i++) {
    const mc = mascot[i] || "", wm = WORDMARK[i - 1] || "";
    const sub = i === 4 ? D(SUBTITLE) : "";
    const left = RB(mc.padEnd(14)) + "  " + (wm ? RB(wm) : sub);
    lines.push(RD(BX.v) + " " + left + " ".repeat(Math.max(1, iw - 14 - (wm?wm.length+2:sub?SUBTITLE.length+2:0) - 1)) + RD(BX.v));
  }
  lines.push(RD(BX.vr + BX.h.repeat(iw) + BX.vl));
  const dot = ready ? G("\u25cf") : R("\u25cb");
  const ml = model.length > 20 ? model.slice(0,18)+"\u2026" : model;
  lines.push(RD(BX.v) + ` ${dot} ${D("MODEL")} ${RA(ml)}  ${D("VIA")} ${RA(prov)}  ${D("CPU")} ${RA(m.cpu+"%")}  ${D("MEM")} ${RA(m.mem+"%")}  ${D("UP")} ${RA(m.uptime)}` + " ".repeat(Math.max(1,iw-62)) + RD(BX.v));
  lines.push(RD(BX.v) + ` ${D("SOCIETY")} ${RA("orchestrator")} ${D("\u2192")} ${RA("executor")} ${D("\u2192")} ${RA("reviewer")} ${D("\u2192")} ${RA("observer")}` + " ".repeat(Math.max(1,iw-58)) + RD(BX.v));
  lines.push(RD(BX.bl + BX.h.repeat(iw) + BX.br));
  moveTo(1, 1);
  for (let i = 0; i < lines.length; i++) { moveTo(i + 1, 1); clearLine(); process.stdout.write("  " + lines[i]); }
}



// ── METRICS ──────────────────────────────────────────────────────────────────
function getMetrics() {
  const m={cpu:0,mem:0,uptime:"?",host:"?"};
  try{m.host=os.hostname();}catch{}
  try{const s=os.uptime();const h=Math.floor(s/3600),mn=Math.floor((s%3600)/60);m.uptime=h>24?`${Math.floor(h/24)}d`:h>0?`${h}h ${mn}m`:`${mn}m`;}catch{}
  try{const t=os.totalmem(),f=os.freemem();m.mem=Math.round(((t-f)/t)*100);}catch{}
  try{if(!isWin){const la=execSync("cat /proc/loadavg",{encoding:"utf-8",timeout:2000,stdio:["pipe","pipe","pipe"]}).split(" ");const c=parseInt(execSync("nproc",{encoding:"utf-8",timeout:2000,stdio:["pipe","pipe","pipe"]}))||1;m.cpu=Math.min(100,Math.round((parseFloat(la[0])/c)*100));}else{const o=execSync('wmic cpu get loadpercentage /value',{encoding:"utf-8",timeout:3000,stdio:["pipe","pipe","pipe"]});m.cpu=parseInt((o.match(/LoadPercentage=(\d+)/)||[])[1])||0;}}catch{}
  return m;
}

// ── PROGRESS ─────────────────────────────────────────────────────────────────
const SP=["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"];
let _si=null,_sf=0;
function stepStart(l){stepStop();_sf=0;_si=setInterval(()=>{_sf=(_sf+1)%SP.length;outputLine(`  ${R(SP[_sf])} ${D(l)}`,true);},80);}
function stepStop(){if(_si){clearInterval(_si);_si=null;}}
function stepDone(l){stepStop();outputLine(`  ${RD("──")}${R("•")} ${l}`);}
function stepFail(l){stepStop();outputLine(`  ${RD("──")}${Y("✗")} ${l}`);}

// ── PROVIDERS ────────────────────────────────────────────────────────────────
const PROVIDERS={"DigitalOcean":{base:"https://inference.do-ai.run/v1",fields:["api_key"]},"OpenAI":{base:"https://api.openai.com/v1",fields:["api_key"]},"Anthropic (via DO)":{base:"https://inference.do-ai.run/v1",fields:["api_key"]},"Google Gemini":{base:"https://generativelanguage.googleapis.com/v1beta/openai",fields:["api_key"]},"NVIDIA NIM":{base:"https://integrate.api.nvidia.com/v1",fields:["api_key"]},"HuggingFace":{base:"https://router.huggingface.co/v1",fields:["api_key"]},"OpenRouter":{base:"https://openrouter.ai/api/v1",fields:["api_key"]},"GitHub Models":{base:"https://models.inference.ai.azure.com",fields:["api_key"]},"Qwen (DashScope)":{base:"https://dashscope-intl.aliyuncs.com/compatible-mode/v1",fields:["api_key"]},"Cloudflare Workers AI":{base:"",fields:["account_id","api_key"],buildBase:f=>`https://api.cloudflare.com/client/v4/accounts/${f.account_id}/ai/v1`},"Azure OpenAI":{base:"",fields:["endpoint_url","deployment","api_key"],buildBase:f=>`${f.endpoint_url}/openai/deployments/${f.deployment}`},"Ollama (local)":{base:"http://localhost:11434/v1",fields:[]},"Custom":{base:"",fields:["base_url","api_key"]}};
const PFX="  "+RD(BX.v);

// ── COMMANDS ─────────────────────────────────────────────────────────────────
const CMD_LIST=[
  {name:"Setup",value:"/setup",icon:"⚙",desc:"Configure provider & model"},
  {name:"Model",value:"/model",icon:"◎",desc:"Switch LLM model"},
  {name:"Providers",value:"/providers",icon:"⊞",desc:"Manage API providers"},
  {name:"Skills",value:"/skills",icon:"◈",desc:"Install/manage skills"},
  {name:"Society",value:"/society",icon:"🏛",desc:"Agent tree view"},
  {name:"Run",value:"/run",icon:"▷",desc:"Execute shell command"},
  {name:"Clear",value:"/clear",icon:"◻",desc:"Clear chat history"},
  {name:"Help",value:"/help",icon:"?",desc:"Show all commands"},
  {name:"Quit",value:"/quit",icon:"⊘",desc:"Exit dashboard"},
];



// ══════════════════════════════════════════════════════════════════════════════
//  RAW INPUT SYSTEM — Fixed-position input line at bottom
// ══════════════════════════════════════════════════════════════════════════════

const INPUT_ROW_OFFSET = 2; // input is 2 rows from bottom
const SCROLL_TOP = HEADER_H + 1;
let inputBuffer = "";
let inputCursor = 0;
let cmdMenuActive = false;
let cmdMenuIndex = 0;
let cmdFilteredList = [];
let outputRow = SCROLL_TOP; // next row to write output (scroll region)

function getInputRow() { return tRows() - 1; }
function getMenuTopRow() { return tRows() - 1 - CMD_LIST.length - 1; }

/** Write a line of output into the scroll region */
function outputLine(text, overwrite = false) {
  saveCursor();
  const scrollBottom = tRows() - INPUT_ROW_OFFSET - 1;
  // If output row exceeds scroll bottom, scroll up
  if (outputRow > scrollBottom) {
    // Scroll: move to bottom of scroll region and write, which auto-scrolls
    moveTo(scrollBottom, 1);
    process.stdout.write("\n");
    outputRow = scrollBottom;
  }
  moveTo(outputRow, 1);
  clearLine();
  process.stdout.write(text);
  if (!overwrite) outputRow++;
  restoreCursor();
  drawInput(); // always redraw input after output
}

/** Write multiple lines */
function outputLines(lines) {
  for (const l of lines) outputLine(l);
}

/** Draw the input line at fixed bottom position */
function drawInput() {
  const row = getInputRow();
  const prefix = "  " + RD(BX.v) + " " + R("\u25b8") + " ";
  const prefixLen = 7; // visible chars: "  │ ▸ "
  moveTo(row, 1);
  clearLine();
  process.stdout.write(prefix + inputBuffer);
  // Position cursor correctly
  moveTo(row, prefixLen + inputCursor + 1);
  showCursor();
}

/** Draw the input border line above input */
function drawInputBorder() {
  const row = getInputRow() - 1;
  moveTo(row, 1);
  clearLine();
  process.stdout.write("  " + RD(BX.h.repeat(Math.min(tCols()-4, 72))));
}



// ── COMMAND MENU (inline, rendered above input) ──────────────────────────────

function showCmdMenu(filter = "") {
  cmdMenuActive = true;
  cmdMenuIndex = 0;
  const f = filter.toLowerCase().slice(1); // remove leading /
  cmdFilteredList = f
    ? CMD_LIST.filter(c => c.name.toLowerCase().includes(f) || c.value.includes(f))
    : [...CMD_LIST];
  if (!cmdFilteredList.length) cmdFilteredList = [...CMD_LIST];
  renderCmdMenu();
}

function renderCmdMenu() {
  const startRow = getInputRow() - cmdFilteredList.length - 2;
  // Draw menu border
  moveTo(startRow, 1); clearLine();
  process.stdout.write("  " + RD(BX.tl + BX.h.repeat(40) + BX.tr));
  for (let i = 0; i < cmdFilteredList.length; i++) {
    const c = cmdFilteredList[i];
    const selected = i === cmdMenuIndex;
    const row = startRow + 1 + i;
    moveTo(row, 1); clearLine();
    const marker = selected ? R("\u25b8") : " ";
    const name = selected ? RA(c.name.padEnd(12)) : D(c.name.padEnd(12));
    const desc = D(c.desc || "");
    process.stdout.write(`  ${RD(BX.v)} ${marker} ${R(c.icon)} ${name} ${desc}`);
  }
  const bottomRow = startRow + cmdFilteredList.length + 1;
  moveTo(bottomRow, 1); clearLine();
  process.stdout.write("  " + RD(BX.bl + BX.h.repeat(40) + BX.br));
  drawInput();
}

function hideCmdMenu() {
  if (!cmdMenuActive) return;
  cmdMenuActive = false;
  // Clear menu area
  const startRow = getInputRow() - CMD_LIST.length - 3;
  for (let i = startRow; i < getInputRow() - 1; i++) {
    moveTo(i, 1); clearLine();
  }
  drawInputBorder();
  drawInput();
}

function cmdMenuUp() {
  if (!cmdMenuActive) return;
  cmdMenuIndex = (cmdMenuIndex - 1 + cmdFilteredList.length) % cmdFilteredList.length;
  renderCmdMenu();
}

function cmdMenuDown() {
  if (!cmdMenuActive) return;
  cmdMenuIndex = (cmdMenuIndex + 1) % cmdFilteredList.length;
  renderCmdMenu();
}

function cmdMenuSelect() {
  if (!cmdMenuActive || !cmdFilteredList.length) return null;
  const selected = cmdFilteredList[cmdMenuIndex];
  hideCmdMenu();
  return selected.value;
}



// ── COMMAND HANDLERS ─────────────────────────────────────────────────────────

async function withMenu(fn) {
  // For inquirer prompts: reset scroll, run, restore
  resetScrollRegion();
  showCursor();
  try { await fn(); } finally {
    setScrollRegion(SCROLL_TOP, tRows() - INPUT_ROW_OFFSET - 1);
    drawInput();
  }
}

async function cmdSetup(){outputLine("");await withMenu(async()=>{const{sm}=await inquirer.prompt([{type:"list",name:"sm",message:"Storage:",choices:["Local SQLite","Cloudflare D1"],prefix:PFX}]);config.set("storage_mode",sm.includes("D1")?"cloudflare":"local");if(sm.includes("D1")){const cf=await inquirer.prompt([{type:"input",name:"a",message:"CF Account ID:",prefix:PFX},{type:"password",name:"t",message:"CF Token:",mask:"•",prefix:PFX},{type:"input",name:"d",message:"D1 DB ID:",prefix:PFX}]);config.set("cf_account_id",cf.a);config.set("cf_api_token",cf.t);config.set("cf_d1_database_id",cf.d);}
const{iface}=await inquirer.prompt([{type:"list",name:"iface",message:"Interface:",choices:["CLI only","Telegram","WhatsApp","Both"],prefix:PFX}]);config.set("interface_mode",{C:"cli",T:"telegram",W:"whatsapp",B:"both"}[iface[0]]||"cli");
if(iface.includes("T")||iface==="Both"){const{t}=await inquirer.prompt([{type:"password",name:"t",message:"Telegram Token:",mask:"•",prefix:PFX}]);if(t)config.set("telegram_token",t);}
const{prov}=await inquirer.prompt([{type:"list",name:"prov",message:"Provider:",choices:Object.keys(PROVIDERS),prefix:PFX}]);const pc=PROVIDERS[prov],pf={};
for(const f of pc.fields){const msg=f==="api_key"?`${prov} Key:`:f==="account_id"?"Account ID:":f==="endpoint_url"?"Endpoint:":"Base URL:";const tp=f==="api_key"?"password":"input";const{v}=await inquirer.prompt([{type:tp,name:"v",message:msg,mask:tp==="password"?"•":undefined,prefix:PFX}]);pf[f]=v;if(f==="api_key"&&v)config.set("openai_api_key",v);if(f==="account_id"&&v)config.set("cf_account_id",v);}
let base=pc.base;if(pc.buildBase)base=pc.buildBase(pf);else if(pf.base_url)base=pf.base_url;if(base)config.set("openai_api_base",base);
let models=["llama3.3-70b-instruct","Custom"];const key=pf.api_key||config.get("openai_api_key");
if(key&&base){try{const r=await fetch(`${base}/models`,{headers:{Authorization:`Bearer ${key}`},signal:AbortSignal.timeout(8000)});if(r.ok){const d=await r.json();const it=(d.data||d.models||[]).map(i=>typeof i==="string"?i:(i.id||i.name||"")).filter(Boolean).slice(0,20);if(it.length)models=[...it,"Custom"];}}catch{}}
const{mdl}=await inquirer.prompt([{type:"list",name:"mdl",message:"Model:",choices:models,default:config.get("default_model"),prefix:PFX}]);
if(mdl==="Custom"){const{c}=await inquirer.prompt([{type:"input",name:"c",message:"Model ID:",prefix:PFX}]);config.set("default_model",c);}else config.set("default_model",mdl);
try{writeFileSync(join(getProjectRoot(),".env"),generateEnvContent());stepDone("Saved");}catch{stepFail("Write failed");}});}

async function cmdModel(){await withMenu(async()=>{if(!config.get("openai_api_key")){await cmdSetup();return;}const{p}=await inquirer.prompt([{type:"list",name:"p",message:"Provider:",choices:Object.keys(PROVIDERS),prefix:PFX}]);stepStart("Fetching models");let models=[];const key=config.get("openai_api_key"),pc=PROVIDERS[p];const base=pc.buildBase?pc.buildBase({account_id:config.get("cf_account_id")}):(pc.base||config.get("openai_api_base"));try{const r=await fetch(`${base}/models`,{headers:{Authorization:`Bearer ${key}`},signal:AbortSignal.timeout(10000)});if(r.ok){const d=await r.json();models=(d.data||d.models||[]).map(i=>typeof i==="string"?i:(i.id||i.name||"")).filter(Boolean).slice(0,30);}}catch{}if(!models.length)models=["llama3.3-70b-instruct"];stepDone(`${models.length} models`);models.push(new inquirer.Separator(),{name:D("Custom..."),value:"__c__"});const{m}=await inquirer.prompt([{type:"list",name:"m",message:"Model:",choices:models,pageSize:15,prefix:PFX}]);if(m==="__c__"){const{c}=await inquirer.prompt([{type:"input",name:"c",message:"ID:",prefix:PFX}]);config.set("default_model",c);}else config.set("default_model",m);stepDone(config.get("default_model"));drawHeader();});}
async function cmdProviders(){await withMenu(async()=>{const{p}=await inquirer.prompt([{type:"list",name:"p",message:"Provider:",choices:Object.keys(PROVIDERS),prefix:PFX}]);const pc=PROVIDERS[p];if(!pc.fields.length){outputLine("  "+D("No config needed."));return;}const pf={};for(const f of pc.fields){const tp=f==="api_key"?"password":"input";const{v}=await inquirer.prompt([{type:tp,name:"v",message:f==="api_key"?`${p} Key:`:"Value:",mask:tp==="password"?"•":undefined,prefix:PFX}]);pf[f]=v;if(f==="api_key"&&v)config.set("openai_api_key",v);if(f==="account_id"&&v)config.set("cf_account_id",v);}let base=pc.base;if(pc.buildBase)base=pc.buildBase(pf);else if(pf.base_url)base=pf.base_url;if(base)config.set("openai_api_base",base);stepDone(`${p} configured`);try{writeFileSync(join(getProjectRoot(),".env"),generateEnvContent());}catch{}});}
async function cmdSkills(){await withMenu(async()=>{const{a}=await inquirer.prompt([{type:"list",name:"a",message:"Skills:",choices:["Install @user/skill","List","Remove"],prefix:PFX}]);if(a.startsWith("I")){const{p}=await inquirer.prompt([{type:"input",name:"p",message:"@user/skill:",prefix:PFX}]);if(p)stepDone(`${p} installed`);}else outputLine("  "+D("Managed via Telegram /skills"));});}



// ── CHAT ─────────────────────────────────────────────────────────────────────
const chatHistory=[];
async function sendMessage(msg){
  const apiKey=config.get("openai_api_key"),apiBase=config.get("openai_api_base"),model=config.get("default_model");
  if(!apiKey){await cmdSetup();return;}
  chatHistory.push({role:"user",content:msg});
  stepStart("Connecting");
  try{
    const msgs=[{role:"system",content:"You are SynthClaw, a personal AI. Be concise. Plain text."},...chatHistory.slice(-8)];
    const resp=await fetch(`${apiBase}/chat/completions`,{method:"POST",headers:{"Content-Type":"application/json",Authorization:`Bearer ${apiKey}`},body:JSON.stringify({model,messages:msgs,temperature:0.7,max_tokens:2048})});
    const data=await resp.json();
    stepStop();
    if(!resp.ok){stepFail(data.error?.message||`HTTP ${resp.status}`);chatHistory.pop();return;}
    const reply=(data.choices?.[0]?.message?.content||"").replace(/<think>[\s\S]*?<\/think>/g,"").trim();
    chatHistory.push({role:"assistant",content:reply});
    stepDone("Done");
    outputLine("  "+RD(BX.tl+"── ")+R("SYNTHCLAW")+RD(" "+"─".repeat(36)+BX.tr));
    for(const line of(reply||"(empty)").split("\n"))outputLine("  "+RD(BX.v)+" "+line);
    outputLine("  "+RD(BX.bl+"─".repeat(48)+BX.br));
    outputLine("");
  }catch(e){stepStop();stepFail(e.message);chatHistory.pop();}
}

// ── COMMAND DISPATCH ─────────────────────────────────────────────────────────
async function handleCmd(input){
  const[cmd,...args]=input.split(" ");const arg=args.join(" ");
  switch(cmd){
    case"/setup":return cmdSetup();
    case"/model":return cmdModel();
    case"/providers":return cmdProviders();
    case"/skills":return cmdSkills();
    case"/society":case"/agents":
      outputLine("  "+R("AGENT SOCIETY"));
      outputLine("  "+RD("──•")+" "+RA("Orchestrator")+" "+D("[idle]")+"  plans + delegates");
      outputLine("  "+RD("  ├─")+" "+RA("Researcher")+"  "+D("[idle]")+"  gathers info");
      outputLine("  "+RD("  ├─")+" "+RA("Executor")+"    "+D("[idle]")+"  runs commands");
      outputLine("  "+RD("  ├─")+" "+RA("Reviewer")+"    "+D("[idle]")+"  validates results");
      outputLine("  "+RD("  └─")+" "+RA("Observer")+"    "+D("[idle]")+"  monitors execution");
      outputLine("  "+D("Complex tasks auto-delegate. Use /delegate <task> to force."));
      outputLine(""); return;
    case"/delegate":if(!arg)return sendMessage(input);return sendMessage("[DELEGATE] "+arg);
    case"/direct":if(!arg)return sendMessage(input);return sendMessage(arg);
    case"/memory":outputLine("  "+D("Use Telegram /memory for full access."));return;
    case"/creds":outputLine("  "+D("Use Telegram /creds for full access."));return;
    case"/status":drawHeader();return;
    case"/clear":chatHistory.length=0;outputLine("  "+D("Cleared."));return;
    case"/run":if(!arg){await withMenu(async()=>{const{c}=await inquirer.prompt([{type:"input",name:"c",message:"$",prefix:PFX}]);if(c)await handleCmd("/run "+c);});return;}try{const o=execSync(arg,{encoding:"utf-8",timeout:30000,cwd:getProjectRoot(),stdio:["pipe","pipe","pipe"]});for(const l of o.trim().split("\n").slice(0,20))outputLine("  "+D(l));}catch(e){stepFail((e.stderr||e.message||"").slice(0,120));}return;
    case"/quit":case"/exit":cleanup();process.exit(0);
    case"/help":
      outputLine("\n  "+R("COMMANDS")+" "+D("(type / for menu, arrows to navigate)"));
      CMD_LIST.forEach(c=>outputLine("  "+RD("──•")+" "+D(c.value.padEnd(14))+R(c.icon)+" "+c.name));
      outputLine("");return;
    default:return sendMessage(input);
  }
}



// ══════════════════════════════════════════════════════════════════════════════
//  MAIN — Raw stdin keypress handling
// ══════════════════════════════════════════════════════════════════════════════

function cleanup() {
  if (blinkTimer) clearInterval(blinkTimer);
  resetScrollRegion();
  showCursor();
  leaveAltScreen();
  if (process.stdin.isTTY) process.stdin.setRawMode(false);
}

let processingCommand = false;

async function processInput(line) {
  if (processingCommand) return;
  processingCommand = true;
  try {
    if (line.startsWith("/")) {
      await handleCmd(line);
    } else {
      await sendMessage(line);
    }
  } catch(e) {
    stepFail(e.message || "Error");
  }
  processingCommand = false;
  drawInput();
}

export async function runDashboard() {
  enterAltScreen();
  process.stdout.write(ESC + "2J");

  process.on("exit", cleanup);
  process.on("SIGINT", () => { cleanup(); process.exit(0); });

  drawHeader();
  startBlink();

  // Set scroll region between header and input area
  setScrollRegion(SCROLL_TOP, tRows() - INPUT_ROW_OFFSET - 1);
  outputRow = SCROLL_TOP;

  // Draw input border and input line
  drawInputBorder();
  drawInput();

  // Auto-setup if unconfigured
  if (!config.get("openai_api_key")) {
    outputLine("  " + R("\u25cf") + " First run. Configuring...");
    outputLine("");
    await cmdSetup();
    drawHeader();
  }

  outputLine("  " + D("Ready. Type / for commands (arrow keys to navigate), or chat."));
  outputLine("");
  drawInput();

  // ── Raw mode stdin ─────────────────────────────────────────────────────────
  if (!process.stdin.isTTY) {
    // Non-TTY fallback: use readline
    const { createInterface } = await import("readline");
    const rl = createInterface({ input: process.stdin, output: process.stdout });
    rl.on("line", async (line) => { await processInput(line.trim()); });
    rl.on("close", () => { cleanup(); process.exit(0); });
    return;
  }

  process.stdin.setRawMode(true);
  process.stdin.resume();
  process.stdin.setEncoding("utf8");

  process.stdin.on("data", async (key) => {
    if (processingCommand) return; // ignore keys while command is running

    // Ctrl+C
    if (key === "\x03") { cleanup(); process.exit(0); }
    // Ctrl+D
    if (key === "\x04") { cleanup(); process.exit(0); }

    // Enter
    if (key === "\r" || key === "\n") {
      if (cmdMenuActive) {
        const selected = cmdMenuSelect();
        if (selected) {
          inputBuffer = "";
          inputCursor = 0;
          drawInput();
          await processInput(selected);
        }
        return;
      }
      const line = inputBuffer.trim();
      inputBuffer = "";
      inputCursor = 0;
      if (line) {
        outputLine("  " + D(">") + " " + line);
        await processInput(line);
      }
      drawInput();
      return;
    }

    // Escape — close menu
    if (key === "\x1b" && !key.startsWith("\x1b[")) {
      if (cmdMenuActive) { hideCmdMenu(); return; }
      return;
    }

    // Arrow keys
    if (key === "\x1b[A") { // Up
      if (cmdMenuActive) { cmdMenuUp(); return; }
      return;
    }
    if (key === "\x1b[B") { // Down
      if (cmdMenuActive) { cmdMenuDown(); return; }
      return;
    }
    if (key === "\x1b[D") { // Left
      if (inputCursor > 0) inputCursor--;
      drawInput(); return;
    }
    if (key === "\x1b[C") { // Right
      if (inputCursor < inputBuffer.length) inputCursor++;
      drawInput(); return;
    }

    // Home / End
    if (key === "\x1b[H") { inputCursor = 0; drawInput(); return; }
    if (key === "\x1b[F") { inputCursor = inputBuffer.length; drawInput(); return; }

    // Backspace
    if (key === "\x7f" || key === "\b") {
      if (inputCursor > 0) {
        inputBuffer = inputBuffer.slice(0, inputCursor - 1) + inputBuffer.slice(inputCursor);
        inputCursor--;
      }
      // Update menu filter if active
      if (inputBuffer.startsWith("/") && inputBuffer.length > 0) {
        showCmdMenu(inputBuffer);
      } else if (cmdMenuActive) {
        hideCmdMenu();
      }
      drawInput();
      return;
    }

    // Delete
    if (key === "\x1b[3~") {
      if (inputCursor < inputBuffer.length) {
        inputBuffer = inputBuffer.slice(0, inputCursor) + inputBuffer.slice(inputCursor + 1);
      }
      drawInput(); return;
    }

    // Tab — autocomplete from menu
    if (key === "\t") {
      if (cmdMenuActive && cmdFilteredList.length) {
        const selected = cmdMenuSelect();
        if (selected) {
          inputBuffer = selected;
          inputCursor = inputBuffer.length;
        }
      }
      drawInput(); return;
    }

    // Normal printable characters
    if (key.length === 1 && key.charCodeAt(0) >= 32) {
      inputBuffer = inputBuffer.slice(0, inputCursor) + key + inputBuffer.slice(inputCursor);
      inputCursor++;

      // Show command menu when typing /
      if (inputBuffer === "/") {
        showCmdMenu("/");
      } else if (inputBuffer.startsWith("/") && !inputBuffer.includes(" ")) {
        showCmdMenu(inputBuffer);
      } else if (cmdMenuActive && !inputBuffer.startsWith("/")) {
        hideCmdMenu();
      }

      drawInput();
      return;
    }
  });
}

export { cmdSetup as runInlineWizard };
