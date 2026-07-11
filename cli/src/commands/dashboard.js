import chalk from "chalk";
import { execSync } from "child_process";
import { createInterface } from "readline";
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

// Detect if the terminal supports alternate screen (modern terminal emulators)
// Old cmd.exe on Windows doesn't support it; ConPTY-based terminals (Windows Terminal, VS Code) do.
const supportsAltScreen = (() => {
  if (!process.stdout.isTTY) return false;
  // Windows: check if running inside Windows Terminal or modern ConPTY
  if (isWin) {
    // WT_SESSION is set by Windows Terminal, TERM_PROGRAM by VS Code terminal
    if (process.env.WT_SESSION || process.env.TERM_PROGRAM || process.env.TERMINAL_EMULATOR) return true;
    // ConEmu, cmder, etc.
    if (process.env.ConEmuPID || process.env.CMDER_ROOT) return true;
    // Legacy cmd.exe or PowerShell without ConPTY — no alt screen
    return false;
  }
  // Unix/macOS: virtually all terminals support alt screen
  return true;
})();

const enterAltScreen = () => { if (supportsAltScreen) process.stdout.write("\x1b[?1049h"); };
const leaveAltScreen = () => { if (supportsAltScreen) process.stdout.write("\x1b[?1049l"); };
const hideCursor = () => process.stdout.write(ESC + "?25l");
const showCursor = () => process.stdout.write(ESC + "?25h");
const moveTo = (row, col) => { if (supportsAltScreen) process.stdout.write(`${ESC}${row};${col}H`); };
const clearLine = () => process.stdout.write(ESC + "2K");
const setScrollRegion = (top, bottom) => { if (supportsAltScreen) process.stdout.write(`${ESC}${top};${bottom}r`); };
const resetScrollRegion = () => { if (supportsAltScreen) process.stdout.write(ESC + "r"); };
const rows = () => process.stdout.rows || 24;
const cols = () => process.stdout.columns || 80;


// ── HEADER PANEL (fixed at top rows 1-HEADER_H) ─────────────────────────────
const HEADER_H = 11; // number of rows the header occupies
let blinkState = 0; // 0=open, 1=blink (millisecond blink)
let blinkTimer = null;

function startBlink() {
  // Blink every 3 seconds for 150ms
  blinkTimer = setInterval(() => {
    blinkState = 1;
    setTimeout(() => { blinkState = 0; drawHeader(); }, 150);
    drawHeader();
  }, 3000);
}

function drawHeader() {
  const m = getMetrics();
  const w = Math.min(cols(), 76);
  const iw = w - 4;
  const mascot = blinkState === 0 ? MASCOT_OPEN : MASCOT_BLINK;
  const model = config.get("default_model") || "\u2014";
  const prov = (() => { const b=config.get("openai_api_base")||""; if(b.includes("do-ai"))return"DO"; if(b.includes("openai.com"))return"OAI"; if(b.includes("openrouter"))return"OR"; if(b.includes("nvidia"))return"NV"; if(b.includes("huggingface"))return"HF"; if(b.includes("googleapis"))return"GG"; if(b.includes("cloudflare"))return"CF"; if(b.includes("localhost"))return"OLL"; return"?"; })();
  const ready = !!config.get("openai_api_key");

  const lines = [];
  lines.push(RD(BX.tl + BX.h.repeat(iw) + BX.tr));

  // Mascot (5 rows) + Wordmark (3 rows offset) + Subtitle
  for (let i = 0; i < 5; i++) {
    const mc = mascot[i] || "";
    const wm = WORDMARK[i - 1] || "";
    const sub = i === 4 ? D(SUBTITLE) : "";
    const left = RB(mc.padEnd(14)) + "  " + (wm ? RB(wm) : sub);
    lines.push(RD(BX.v) + " " + left + " ".repeat(Math.max(1, iw - 14 - (wm?wm.length+2:sub?SUBTITLE.length+2:0) - 1)) + RD(BX.v));
  }

  lines.push(RD(BX.vr + BX.h.repeat(iw) + BX.vl));

  // Status line
  const dot = ready ? G("\u25cf") : R("\u25cb");
  const ml = model.length > 20 ? model.slice(0,18)+"\u2026" : model;
  lines.push(RD(BX.v) + ` ${dot} ${D("MODEL")} ${RA(ml)}  ${D("VIA")} ${RA(prov)}  ${D("CPU")} ${RA(m.cpu+"%")}  ${D("MEM")} ${RA(m.mem+"%")}  ${D("UP")} ${RA(m.uptime)}` + " ".repeat(Math.max(1,iw-62)) + RD(BX.v));

  // Agent Society status line
  lines.push(RD(BX.v) + ` ${D("SOCIETY")} ${RA("orchestrator")} ${D("\u2192")} ${RA("executor")} ${D("\u2192")} ${RA("reviewer")} ${D("\u2192")} ${RA("observer")}` + " ".repeat(Math.max(1,iw-58)) + RD(BX.v));

  lines.push(RD(BX.bl + BX.h.repeat(iw) + BX.br));

  if (supportsAltScreen) {
    // Draw at top of screen (alternate screen mode)
    moveTo(1, 1);
    for (let i = 0; i < lines.length; i++) {
      moveTo(i + 1, 1);
      clearLine();
      process.stdout.write("  " + lines[i]);
    }
  } else {
    // Fallback: just print the header (no cursor positioning)
    console.clear();
    for (const line of lines) {
      console.log("  " + line);
    }
  }
}

// ── INPUT LINE (fixed at bottom) ────────────────────────────────────────────
const INPUT_H = 2; // rows reserved at bottom for input

function drawInputBorder() {
  const r = rows();
  moveTo(r - 1, 1); clearLine();
  process.stdout.write("  " + RD(BX.h.repeat(Math.min(cols()-4, 72))));
}


// ── METRICS (silent, cross-platform) ─────────────────────────────────────────
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
function stepStart(l){stepStop();_sf=0;_si=setInterval(()=>{_sf=(_sf+1)%SP.length;process.stdout.write(`\r  ${R(SP[_sf])} ${D(l)}${"".padEnd(20)}`);},80);}
function stepStop(){if(_si){clearInterval(_si);_si=null;}process.stdout.write("\r"+" ".repeat(60)+"\r");}
function stepDone(l){stepStop();console.log(`  ${RD("──")}${R("•")} ${l}`);}
function stepFail(l){stepStop();console.log(`  ${RD("──")}${Y("✗")} ${l}`);}


// ── PROVIDERS ────────────────────────────────────────────────────────────────
const PROVIDERS={"DigitalOcean":{base:"https://inference.do-ai.run/v1",fields:["api_key"]},"OpenAI":{base:"https://api.openai.com/v1",fields:["api_key"]},"Anthropic (via DO)":{base:"https://inference.do-ai.run/v1",fields:["api_key"]},"Google Gemini":{base:"https://generativelanguage.googleapis.com/v1beta/openai",fields:["api_key"]},"NVIDIA NIM":{base:"https://integrate.api.nvidia.com/v1",fields:["api_key"]},"HuggingFace":{base:"https://router.huggingface.co/v1",fields:["api_key"]},"OpenRouter":{base:"https://openrouter.ai/api/v1",fields:["api_key"]},"GitHub Models":{base:"https://models.inference.ai.azure.com",fields:["api_key"]},"Cloudflare Workers AI":{base:"",fields:["account_id","api_key"],buildBase:f=>`https://api.cloudflare.com/client/v4/accounts/${f.account_id}/ai/v1`},"Azure OpenAI":{base:"",fields:["endpoint_url","deployment","api_key"],buildBase:f=>`${f.endpoint_url}/openai/deployments/${f.deployment}`},"Ollama (local)":{base:"http://localhost:11434/v1",fields:[]},"Custom":{base:"",fields:["base_url","api_key"]}};
const PFX="  "+RD(BX.v);

// ── COMMANDS ─────────────────────────────────────────────────────────────────
const CMD_LIST=[{name:R("⚙")+" Setup",value:"/setup"},{name:R("◎")+" Model",value:"/model"},{name:R("⊞")+" Providers",value:"/providers"},{name:R("◈")+" Skills",value:"/skills"},{name:R("🏛")+" Society",value:"/society"},{name:R("▷")+" Run",value:"/run"},{name:R("◻")+" Clear",value:"/clear"},{name:R("⊘")+" Quit",value:"/quit"}];

// Helper: reset scroll region before interactive prompts, restore after
async function withMenu(fn) {
  resetScrollRegion();
  try { await fn(); } finally {
    setScrollRegion(HEADER_H + 1, rows() - INPUT_H);
  }
}

async function cmdSetup(){console.log("");const{sm}=await inquirer.prompt([{type:"list",name:"sm",message:"Storage:",choices:["Local SQLite","Cloudflare D1"],prefix:PFX}]);config.set("storage_mode",sm.includes("D1")?"cloudflare":"local");if(sm.includes("D1")){const cf=await inquirer.prompt([{type:"input",name:"a",message:"CF Account ID:",prefix:PFX},{type:"password",name:"t",message:"CF Token:",mask:"•",prefix:PFX},{type:"input",name:"d",message:"D1 DB ID:",prefix:PFX}]);config.set("cf_account_id",cf.a);config.set("cf_api_token",cf.t);config.set("cf_d1_database_id",cf.d);}
const{iface}=await inquirer.prompt([{type:"list",name:"iface",message:"Interface:",choices:["CLI only","Telegram","WhatsApp","Both"],prefix:PFX}]);config.set("interface_mode",{C:"cli",T:"telegram",W:"whatsapp",B:"both"}[iface[0]]||"cli");
if(iface.includes("T")||iface==="Both"){const{t}=await inquirer.prompt([{type:"password",name:"t",message:"Telegram Token:",mask:"•",prefix:PFX}]);if(t)config.set("telegram_token",t);}
const{prov}=await inquirer.prompt([{type:"list",name:"prov",message:"Provider:",choices:Object.keys(PROVIDERS),prefix:PFX}]);const pc=PROVIDERS[prov],pf={};
for(const f of pc.fields){const msg=f==="api_key"?`${prov} Key:`:f==="account_id"?"Account ID:":f==="endpoint_url"?"Endpoint:":"Base URL:";const tp=f==="api_key"?"password":"input";const{v}=await inquirer.prompt([{type:tp,name:"v",message:msg,mask:tp==="password"?"•":undefined,prefix:PFX}]);pf[f]=v;if(f==="api_key"&&v)config.set("openai_api_key",v);if(f==="account_id"&&v)config.set("cf_account_id",v);}
let base=pc.base;if(pc.buildBase)base=pc.buildBase(pf);else if(pf.base_url)base=pf.base_url;if(base)config.set("openai_api_base",base);
let models=["llama3.3-70b-instruct","Custom"];const key=pf.api_key||config.get("openai_api_key");
if(key&&base){try{const r=await fetch(`${base}/models`,{headers:{Authorization:`Bearer ${key}`},signal:AbortSignal.timeout(8000)});if(r.ok){const d=await r.json();const it=(d.data||d.models||[]).map(i=>typeof i==="string"?i:(i.id||i.name||"")).filter(Boolean).slice(0,20);if(it.length)models=[...it,"Custom"];}}catch{}}
const{mdl}=await inquirer.prompt([{type:"list",name:"mdl",message:"Model:",choices:models,default:config.get("default_model"),prefix:PFX}]);
if(mdl==="Custom"){const{c}=await inquirer.prompt([{type:"input",name:"c",message:"Model ID:",prefix:PFX}]);config.set("default_model",c);}else config.set("default_model",mdl);
try{writeFileSync(join(getProjectRoot(),".env"),generateEnvContent());stepDone("Saved");}catch{stepFail("Write failed");}console.log("");}

async function cmdModel(){await withMenu(async()=>{if(!config.get("openai_api_key")){await cmdSetup();return;}const{p}=await inquirer.prompt([{type:"list",name:"p",message:"Provider:",choices:Object.keys(PROVIDERS),prefix:PFX}]);stepStart("Fetching models");let models=[];const key=config.get("openai_api_key"),pc=PROVIDERS[p];const base=pc.buildBase?pc.buildBase({account_id:config.get("cf_account_id")}):(pc.base||config.get("openai_api_base"));try{const r=await fetch(`${base}/models`,{headers:{Authorization:`Bearer ${key}`},signal:AbortSignal.timeout(10000)});if(r.ok){const d=await r.json();models=(d.data||d.models||[]).map(i=>typeof i==="string"?i:(i.id||i.name||"")).filter(Boolean).slice(0,30);}}catch{}if(!models.length)models=["llama3.3-70b-instruct"];stepDone(`${models.length} models`);models.push(new inquirer.Separator(),{name:D("Custom..."),value:"__c__"});const{m}=await inquirer.prompt([{type:"list",name:"m",message:"Model:",choices:models,pageSize:15,prefix:PFX}]);if(m==="__c__"){const{c}=await inquirer.prompt([{type:"input",name:"c",message:"ID:",prefix:PFX}]);config.set("default_model",c);}else config.set("default_model",m);stepDone(config.get("default_model"));});}
async function cmdProviders(){await withMenu(async()=>{const{p}=await inquirer.prompt([{type:"list",name:"p",message:"Provider:",choices:Object.keys(PROVIDERS),prefix:PFX}]);const pc=PROVIDERS[p];if(!pc.fields.length){console.log("  "+D("No config needed."));return;}const pf={};for(const f of pc.fields){const tp=f==="api_key"?"password":"input";const{v}=await inquirer.prompt([{type:tp,name:"v",message:f==="api_key"?`${p} Key:`:"Value:",mask:tp==="password"?"•":undefined,prefix:PFX}]);pf[f]=v;if(f==="api_key"&&v)config.set("openai_api_key",v);if(f==="account_id"&&v)config.set("cf_account_id",v);}let base=pc.base;if(pc.buildBase)base=pc.buildBase(pf);else if(pf.base_url)base=pf.base_url;if(base)config.set("openai_api_base",base);stepDone(`${p} configured`);try{writeFileSync(join(getProjectRoot(),".env"),generateEnvContent());}catch{}});}
async function cmdSkills(){await withMenu(async()=>{const{a}=await inquirer.prompt([{type:"list",name:"a",message:"Skills:",choices:["Install @user/skill","List","Remove"],prefix:PFX}]);if(a.startsWith("I")){const{p}=await inquirer.prompt([{type:"input",name:"p",message:"@user/skill:",prefix:PFX}]);if(p)stepDone(`${p} installed`);}else console.log("  "+D("Managed via Telegram /skills"));});}
async function cmdMemory(){console.log("  "+D("Use Telegram /memory for full access."));}
async function cmdCreds(){console.log("  "+D("Use Telegram /creds for full access."));}


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
    console.log("  "+RD(BX.tl+"── ")+R("SYNTHCLAW")+RD(" "+"─".repeat(36)+BX.tr));
    for(const line of(reply||"(empty)").split("\n"))console.log("  "+RD(BX.v)+" "+line);
    console.log("  "+RD(BX.bl+"─".repeat(48)+BX.br));console.log("");
  }catch(e){stepStop();stepFail(e.message);chatHistory.pop();}
}

// ── COMMAND HANDLER ──────────────────────────────────────────────────────────
async function handleCmd(input){
  const[cmd,...args]=input.split(" ");const arg=args.join(" ");
  switch(cmd){
    case"/":{const{c}=await inquirer.prompt([{type:"list",name:"c",message:R("▸"),choices:CMD_LIST,pageSize:12}]);return handleCmd(c);}
    case"/setup":return cmdSetup();
    case"/model":return cmdModel();
    case"/providers":return cmdProviders();
    case"/skills":return cmdSkills();
    case"/society":case"/agents":
      console.log("  "+R("AGENT SOCIETY"));
      console.log("  "+RD("──•")+" "+RA("Orchestrator")+" "+D("[idle]")+"  plans + delegates");
      console.log("  "+RD("  ├─")+" "+RA("Researcher")+"  "+D("[idle]")+"  gathers info");
      console.log("  "+RD("  ├─")+" "+RA("Executor")+"    "+D("[idle]")+"  runs commands");
      console.log("  "+RD("  ├─")+" "+RA("Reviewer")+"    "+D("[idle]")+"  validates results");
      console.log("  "+RD("  └─")+" "+RA("Observer")+"    "+D("[idle]")+"  monitors execution");
      console.log("  "+D("Complex tasks auto-delegate. Use /delegate <task> to force."));
      console.log(""); return;
    case"/delegate":if(!arg)return sendMessage(input);return sendMessage("[DELEGATE] "+arg);
    case"/direct":if(!arg)return sendMessage(input);return sendMessage(arg);
    case"/memory":return cmdMemory();
    case"/creds":return cmdCreds();
    case"/status":drawHeader();return;
    case"/clear":chatHistory.length=0;console.log("  "+D("Cleared."));return;
    case"/run":if(!arg){const{c}=await inquirer.prompt([{type:"input",name:"c",message:"$",prefix:PFX}]);if(c)return handleCmd("/run "+c);return;}try{const o=execSync(arg,{encoding:"utf-8",timeout:30000,cwd:getProjectRoot(),stdio:["pipe","pipe","pipe"]});console.log(D(o.trim().split("\n").slice(0,20).map(l=>"  "+l).join("\n")));}catch(e){stepFail((e.stderr||e.message||"").slice(0,120));}return;
    case"/quit":case"/exit":cleanup();process.exit(0);
    case"/help":console.log("\n  "+R("COMMANDS")+" "+D("(type / for menu)"));CMD_LIST.forEach(c=>console.log("  "+RD("──•")+" "+D(c.value.padEnd(12))+c.name.replace(/\x1b\[[0-9;]*m/g,"").trim()));console.log("");return;
    default:return sendMessage(input);
  }
}


// ── MAIN (alternate screen + scroll region) ──────────────────────────────────
function cleanup() {
  resetScrollRegion();
  showCursor();
  leaveAltScreen();
}

export async function runDashboard() {
  // Enter alternate screen buffer (like nano/vim — previous CLI hidden)
  enterAltScreen();
  if (supportsAltScreen) {
    process.stdout.write(ESC + "2J"); // clear alternate screen
  }

  // Handle exit gracefully
  process.on("exit", cleanup);
  process.on("SIGINT", () => { cleanup(); process.exit(0); });

  // Draw header at top
  drawHeader();
  startBlink(); // mascot blinks every 3s for 150ms

  if (supportsAltScreen) {
    // Set scroll region: between header and input area
    const scrollTop = HEADER_H + 1;
    const scrollBottom = rows() - INPUT_H;
    setScrollRegion(scrollTop, scrollBottom);

    // Move cursor into scroll region
    moveTo(scrollTop, 1);
    showCursor();
    // Draw input border at bottom
    drawInputBorder();
  } else {
    // Fallback: just show cursor, no scroll regions
    showCursor();
    console.log("");
  }

  // Auto-setup if unconfigured
  if (supportsAltScreen) moveTo(HEADER_H + 1, 1);
  if (!config.get("openai_api_key")) {
    console.log("  " + R("\u25cf") + " First run. Configuring...");
    console.log("");
    await cmdSetup();
    // Redraw header with new config
    drawHeader();
  }

  console.log("  " + D("Ready. Type / for commands, or chat."));
  console.log("");

  // Readline for input — prompt with visible orange border
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: "  " + RD(BX.v) + " " + R("\u25b8") + " ",
    terminal: true,
  });
  rl.prompt();

  rl.on("line", async (line) => {
    const input = line.trim();
    if (!input) { rl.prompt(); return; }
    if (input === "/") {
      // Reset scroll region so inquirer renders correctly
      resetScrollRegion();
      const { c } = await inquirer.prompt([{
        type: "list", name: "c", message: R("\u25b8") + " " + D("Command:"),
        choices: CMD_LIST, pageSize: 11,
      }]);
      // Restore scroll region after menu closes
      setScrollRegion(HEADER_H + 1, rows() - INPUT_H);
      await handleCmd(c);
    } else if (input.startsWith("/")) {
      await handleCmd(input);
    } else {
      await sendMessage(input);
    }
    rl.prompt();
  });

  rl.on("close", () => {
    if (blinkTimer) clearInterval(blinkTimer);
    cleanup();
    process.exit(0);
  });
}

export { cmdSetup as runInlineWizard };
