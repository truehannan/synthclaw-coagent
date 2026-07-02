import chalk from "chalk";
import { execSync } from "child_process";
import { createInterface } from "readline";
import { existsSync, writeFileSync } from "fs";
import { join } from "path";
import os from "os";
import inquirer from "inquirer";
import { config, generateEnvContent, getProjectRoot } from "../utils.js";
import { SYNTHCLAW_BLOCK, ICON_FRAME_1, ICON_FRAME_2 } from "../ascii.js";

// ── THEME ────────────────────────────────────────────────────────────────────
const R=chalk.hex("#cc0000"),RB=chalk.hex("#ff1a1a"),RD=chalk.hex("#4d0000");
const RA=chalk.hex("#ff3333"),D=chalk.dim,G=chalk.hex("#33ff33"),Y=chalk.hex("#ffaa00");
const BX={tl:"╭",tr:"╮",bl:"╰",br:"╯",h:"─",v:"│",vr:"├",vl:"┤"};
const isWin=process.platform==="win32";

// ── ANSI TERMINAL CONTROL ────────────────────────────────────────────────────
const ESC = "\x1b[";
const enterAltScreen = () => process.stdout.write("\x1b[?1049h");
const leaveAltScreen = () => process.stdout.write("\x1b[?1049l");
const hideCursor = () => process.stdout.write(ESC + "?25l");
const showCursor = () => process.stdout.write(ESC + "?25h");
const moveTo = (row, col) => process.stdout.write(`${ESC}${row};${col}H`);
const clearLine = () => process.stdout.write(ESC + "2K");
const setScrollRegion = (top, bottom) => process.stdout.write(`${ESC}${top};${bottom}r`);
const resetScrollRegion = () => process.stdout.write(ESC + "r");
const rows = () => process.stdout.rows || 24;
const cols = () => process.stdout.columns || 80;


// ── HEADER PANEL (fixed at top rows 1-HEADER_H) ─────────────────────────────
const HEADER_H = 9; // number of rows the header occupies
let iconFrame = 0;

function drawHeader() {
  const m = getMetrics();
  const w = Math.min(cols(), 76);
  const iw = w - 4;
  const icon = iconFrame === 0 ? ICON_FRAME_1 : ICON_FRAME_2;
  const model = config.get("default_model") || "—";
  const prov = (() => { const b=config.get("openai_api_base")||""; if(b.includes("do-ai"))return"DO"; if(b.includes("openai.com"))return"OAI"; if(b.includes("openrouter"))return"OR"; if(b.includes("nvidia"))return"NV"; if(b.includes("huggingface"))return"HF"; if(b.includes("googleapis"))return"GG"; if(b.includes("cloudflare"))return"CF"; if(b.includes("localhost"))return"OLL"; return"?"; })();
  const ready = !!config.get("openai_api_key");

  const lines = [];
  lines.push(RD(BX.tl + BX.h.repeat(iw) + BX.tr));
  // Icon + wordmark (icon is 5 rows, wordmark is 3 rows — offset wordmark down 1)
  for (let i = 0; i < 5; i++) {
    const ic = icon[i] || "";
    const sc = SYNTHCLAW_BLOCK[i - 1] || "";
    // Gap of 3 spaces between icon and text
    const left = RB(ic.padEnd(17)) + (sc ? "   " + RB(sc) : "");
    lines.push(RD(BX.v) + " " + left + " ".repeat(Math.max(1, iw - 17 - (sc ? sc.length + 3 : 0) - 1)) + RD(BX.v));
  }
  lines.push(RD(BX.vr + BX.h.repeat(iw) + BX.vl));
  // Status
  const dot = ready ? G("●") : R("○");
  const ml = model.length > 24 ? model.slice(0,22)+"…" : model;
  lines.push(RD(BX.v) + ` ${dot} ${D("MODEL")} ${RA(ml)}  ${D("VIA")} ${RA(prov)}  ${D("CPU")} ${RA(m.cpu+"%")}  ${D("MEM")} ${RA(m.mem+"%")}  ${D("UP")} ${RA(m.uptime)}` + " ".repeat(Math.max(1,iw-65)) + RD(BX.v));
  lines.push(RD(BX.bl + BX.h.repeat(iw) + BX.br));

  // Draw at top of screen
  moveTo(1, 1);
  for (let i = 0; i < lines.length; i++) {
    moveTo(i + 1, 1);
    clearLine();
    process.stdout.write("  " + lines[i]);
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
const CMD_LIST=[{name:R("⚙")+" Setup",value:"/setup"},{name:R("◎")+" Model",value:"/model"},{name:R("⊞")+" Providers",value:"/providers"},{name:R("◈")+" Skills",value:"/skills"},{name:R("◉")+" Memory",value:"/memory"},{name:R("⊡")+" Creds",value:"/creds"},{name:R("▣")+" Status",value:"/status"},{name:R("▷")+" Run",value:"/run"},{name:R("◻")+" Clear",value:"/clear"},{name:R("⊘")+" Quit",value:"/quit"}];

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

async function cmdModel(){if(!config.get("openai_api_key")){await cmdSetup();return;}const{p}=await inquirer.prompt([{type:"list",name:"p",message:"Provider:",choices:Object.keys(PROVIDERS),prefix:PFX}]);stepStart("Fetching models");let models=[];const key=config.get("openai_api_key"),pc=PROVIDERS[p];const base=pc.buildBase?pc.buildBase({account_id:config.get("cf_account_id")}):(pc.base||config.get("openai_api_base"));try{const r=await fetch(`${base}/models`,{headers:{Authorization:`Bearer ${key}`},signal:AbortSignal.timeout(10000)});if(r.ok){const d=await r.json();models=(d.data||d.models||[]).map(i=>typeof i==="string"?i:(i.id||i.name||"")).filter(Boolean).slice(0,30);}}catch{}if(!models.length)models=["llama3.3-70b-instruct"];stepDone(`${models.length} models`);models.push(new inquirer.Separator(),{name:D("Custom..."),value:"__c__"});const{m}=await inquirer.prompt([{type:"list",name:"m",message:"Model:",choices:models,pageSize:15,prefix:PFX}]);if(m==="__c__"){const{c}=await inquirer.prompt([{type:"input",name:"c",message:"ID:",prefix:PFX}]);config.set("default_model",c);}else config.set("default_model",m);stepDone(config.get("default_model"));}
async function cmdProviders(){const{p}=await inquirer.prompt([{type:"list",name:"p",message:"Provider:",choices:Object.keys(PROVIDERS),prefix:PFX}]);const pc=PROVIDERS[p];if(!pc.fields.length){console.log("  "+D("No config needed."));return;}const pf={};for(const f of pc.fields){const tp=f==="api_key"?"password":"input";const{v}=await inquirer.prompt([{type:tp,name:"v",message:f==="api_key"?`${p} Key:`:"Value:",mask:tp==="password"?"•":undefined,prefix:PFX}]);pf[f]=v;if(f==="api_key"&&v)config.set("openai_api_key",v);if(f==="account_id"&&v)config.set("cf_account_id",v);}let base=pc.base;if(pc.buildBase)base=pc.buildBase(pf);else if(pf.base_url)base=pf.base_url;if(base)config.set("openai_api_base",base);stepDone(`${p} configured`);try{writeFileSync(join(getProjectRoot(),".env"),generateEnvContent());}catch{}}
async function cmdSkills(){const{a}=await inquirer.prompt([{type:"list",name:"a",message:"Skills:",choices:["Install @user/skill","List","Remove"],prefix:PFX}]);if(a.startsWith("I")){const{p}=await inquirer.prompt([{type:"input",name:"p",message:"@user/skill:",prefix:PFX}]);if(p)stepDone(`${p} installed`);}else console.log("  "+D("Managed via Telegram /skills"));}
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
  process.stdout.write(ESC + "2J"); // clear alternate screen

  // Handle exit gracefully
  process.on("exit", cleanup);
  process.on("SIGINT", () => { cleanup(); process.exit(0); });

  // Draw header at top
  drawHeader();

  // Set scroll region: between header and input area
  const scrollTop = HEADER_H + 1;
  const scrollBottom = rows() - INPUT_H;
  setScrollRegion(scrollTop, scrollBottom);

  // Move cursor into scroll region
  moveTo(scrollTop, 1);
  showCursor();

  // Animate icon tail every second (redraws header in-place)
  const iconTimer = setInterval(() => {
    iconFrame = (iconFrame + 1) % 2;
    // Save cursor, draw header, restore cursor
    process.stdout.write("\x1b7"); // save
    drawHeader();
    process.stdout.write("\x1b8"); // restore
  }, 1000);

  // Draw input border at bottom
  drawInputBorder();

  // Auto-setup if unconfigured
  moveTo(scrollTop, 1);
  if (!config.get("openai_api_key")) {
    console.log("  " + R("●") + " First run. Configuring...");
    console.log("");
    await cmdSetup();
    // Redraw header with new config
    drawHeader();
  }

  console.log("  " + D("Ready. Type / for commands, or chat."));
  console.log("");

  // Readline for input
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: "  " + RD(BX.v) + " " + R("\u25b8") + " ",
  });
  rl.prompt();

  rl.on("line", async (line) => {
    const input = line.trim();
    if (!input) { rl.prompt(); return; }
    if (input === "/") {
      const { c } = await inquirer.prompt([{
        type: "list", name: "c", message: R("\u25b8"),
        choices: CMD_LIST, pageSize: 11,
      }]);
      await handleCmd(c);
    } else if (input.startsWith("/")) {
      await handleCmd(input);
    } else {
      await sendMessage(input);
    }
    rl.prompt();
  });

  rl.on("close", () => {
    clearInterval(iconTimer);
    cleanup();
    process.exit(0);
  });
}

export { cmdSetup as runInlineWizard };
