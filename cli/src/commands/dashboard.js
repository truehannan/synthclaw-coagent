import chalk from "chalk";
import { execSync } from "child_process";
import { createInterface } from "readline";
import { config, getProjectRoot, printError } from "../utils.js";
import { SYNTHCLAW_ASCII } from "../ascii.js";

const RED = chalk.hex("#e85d04");
const DIM_RED = chalk.hex("#8b3a00");
const BRIGHT_RED = chalk.hex("#ff4500");

// ── System metrics ──────────────────────────────────────────────────────────

function getSystemMetrics() {
  const metrics = { cpu: 0, mem: 0, disk: 0, uptime: "", ip: "", location: "" };
  try {
    const loadavg = execSync("cat /proc/loadavg 2>/dev/null", { encoding: "utf-8" }).trim();
    const cores = parseInt(execSync("nproc 2>/dev/null", { encoding: "utf-8" }).trim()) || 1;
    metrics.cpu = Math.min(100, Math.round((parseFloat(loadavg.split(" ")[0]) / cores) * 100));
  } catch {}
  try {
    const memInfo = execSync("free -m 2>/dev/null | awk 'NR==2{print $3,$2}'", { encoding: "utf-8" }).trim();
    const [used, total] = memInfo.split(" ").map(Number);
    metrics.mem = Math.round((used / total) * 100);
  } catch {}
  try {
    const diskInfo = execSync("df -h / 2>/dev/null | awk 'NR==2{print $5}'", { encoding: "utf-8" }).trim();
    metrics.disk = parseInt(diskInfo) || 0;
  } catch {}
  try {
    metrics.uptime = execSync("uptime -p 2>/dev/null", { encoding: "utf-8" }).trim().replace("up ", "");
  } catch {}
  try {
    metrics.ip = execSync("curl -s --max-time 3 ifconfig.me 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}'", { encoding: "utf-8" }).trim();
  } catch {}
  return metrics;
}

function renderGauge(label, percent, width = 20) {
  const filled = Math.round((percent / 100) * width);
  const empty = width - filled;
  const color = percent > 80 ? BRIGHT_RED : percent > 50 ? RED : DIM_RED;
  const bar = color("█".repeat(filled)) + chalk.dim("░".repeat(empty));
  return `  ${chalk.dim(label.padEnd(6))} ${bar} ${color(percent + "%")}`;
}

function renderHeader() {
  const metrics = getSystemMetrics();
  const lines = [];
  lines.push("");
  lines.push(RED("  ╔══════════════════════════════════════════════════════╗"));
  lines.push(RED("  ║") + BRIGHT_RED("  SYNTHCLAW COAGENT") + chalk.dim(" — Command Interface") + RED("       ║"));
  lines.push(RED("  ╚══════════════════════════════════════════════════════╝"));
  lines.push("");
  lines.push(renderGauge("CPU", metrics.cpu));
  lines.push(renderGauge("MEM", metrics.mem));
  lines.push(renderGauge("DISK", metrics.disk));
  lines.push("");
  lines.push(chalk.dim(`  IP: ${metrics.ip || "unknown"}  •  Uptime: ${metrics.uptime || "?"}`));
  lines.push(chalk.dim(`  Model: ${config.get("default_model")}  •  Provider: ${config.get("openai_api_base")}`));
  lines.push("");
  lines.push(RED("  ─".repeat(27)));
  lines.push(chalk.dim("  Type a message or /command. Tab for autocomplete. Ctrl+C to exit."));
  lines.push("");
  return lines.join("\n");
}

// ── Slash command autocomplete ──────────────────────────────────────────────

const COMMANDS = [
  "/help", "/clear", "/model", "/models", "/status", "/memory",
  "/creds", "/run", "/plan", "/agent", "/ping", "/skills",
  "/apis", "/connectors", "/mcp", "/usage", "/stop", "/task",
];

function autocomplete(line) {
  if (!line.startsWith("/")) return [[], line];
  const hits = COMMANDS.filter((c) => c.startsWith(line));
  return [hits, line];
}

// ── Chat send function ──────────────────────────────────────────────────────

async function sendMessage(message) {
  const apiKey = config.get("openai_api_key");
  const apiBase = config.get("openai_api_base");
  const model = config.get("default_model");

  if (!apiKey) {
    printError("No API key. Run: synthclaw setup");
    return;
  }

  process.stdout.write(chalk.dim("\n  thinking...\r"));

  try {
    const response = await fetch(`${apiBase}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: "You are SynthClaw, a personal AI assistant. Be concise and direct." },
          { role: "user", content: message },
        ],
        temperature: 0.7,
        max_tokens: 2048,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      console.log(BRIGHT_RED(`\n  ✗ ${data.error?.message || "API error"}\n`));
      return;
    }

    const reply = data.choices?.[0]?.message?.content || "(empty)";
    // Clean and display
    const cleaned = reply.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
    console.log("");
    console.log(RED("  ┌─ SynthClaw"));
    for (const line of cleaned.split("\n")) {
      console.log(RED("  │ ") + line);
    }
    console.log(RED("  └─"));
    console.log("");
  } catch (err) {
    console.log(BRIGHT_RED(`\n  ✗ ${err.message}\n`));
  }
}

// ── Handle slash commands locally ───────────────────────────────────────────

async function handleCommand(input) {
  const [cmd, ...args] = input.split(" ");
  const arg = args.join(" ");

  switch (cmd) {
    case "/help":
      console.log(chalk.dim("\n  Available commands:"));
      for (const c of COMMANDS) {
        console.log(`    ${RED(c)}`);
      }
      console.log("");
      break;
    case "/model":
      if (arg) {
        config.set("default_model", arg);
        console.log(chalk.dim(`\n  Switched to: ${RED(arg)}\n`));
      } else {
        console.log(chalk.dim(`\n  Current: ${RED(config.get("default_model"))}\n`));
      }
      break;
    case "/status":
      console.log(renderHeader());
      break;
    case "/clear":
      console.clear();
      console.log(renderHeader());
      break;
    default:
      // Send as message to LLM (the /command is part of the message)
      await sendMessage(input);
  }
}

// ── Main dashboard loop ─────────────────────────────────────────────────────

export async function runDashboard() {
  // Startup animation
  console.clear();
  console.log(RED(SYNTHCLAW_ASCII));
  await new Promise((r) => setTimeout(r, 800));
  console.clear();
  console.log(renderHeader());

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: RED("  ▶ "),
    completer: autocomplete,
  });

  rl.prompt();

  rl.on("line", async (line) => {
    const input = line.trim();
    if (!input) {
      rl.prompt();
      return;
    }

    if (input.startsWith("/")) {
      await handleCommand(input);
    } else {
      await sendMessage(input);
    }

    rl.prompt();
  });

  rl.on("close", () => {
    console.log(chalk.dim("\n  Goodbye.\n"));
    process.exit(0);
  });
}
