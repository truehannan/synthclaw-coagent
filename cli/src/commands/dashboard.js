import chalk from "chalk";
import { execSync } from "child_process";
import { createInterface } from "readline";
import { config, getProjectRoot, printError } from "../utils.js";

const RED = chalk.hex("#e85d04");
const DIM_RED = chalk.hex("#8b3a00");
const BRIGHT_RED = chalk.hex("#ff4500");
const BG_RED = chalk.bgHex("#1a0800");

// ── Block character "SYNTHCLAW" header ──────────────────────────────────────
// Small compact header using █ block characters in red

const BLOCK_HEADER = [
  "█▀▀ █▄█ █▄ █ ▀█▀ █ █ █▀▀ █   █▀█ █ █ █",
  "▀▀█  █  █ ▀█  █  █▀█ █   █   █▀█ █▄█ █",
  "▀▀▀  ▀  ▀  ▀  ▀  ▀ ▀ ▀▀▀ ▀▀▀ ▀ ▀  ▀  ▀",
];

// ── System metrics ──────────────────────────────────────────────────────────

function getSystemMetrics() {
  const metrics = { cpu: 0, mem: 0, memUsed: 0, memTotal: 0, disk: 0, diskUsed: "", diskTotal: "", uptime: "", ip: "" };
  try {
    const loadavg = execSync("cat /proc/loadavg 2>/dev/null", { encoding: "utf-8" }).trim();
    const cores = parseInt(execSync("nproc 2>/dev/null", { encoding: "utf-8" }).trim()) || 1;
    metrics.cpu = Math.min(100, Math.round((parseFloat(loadavg.split(" ")[0]) / cores) * 100));
  } catch {}
  try {
    const memInfo = execSync("free -m 2>/dev/null | awk 'NR==2{print $3,$2}'", { encoding: "utf-8" }).trim();
    const [used, total] = memInfo.split(" ").map(Number);
    metrics.mem = Math.round((used / total) * 100);
    metrics.memUsed = used;
    metrics.memTotal = total;
  } catch {}
  try {
    const diskInfo = execSync("df -h / 2>/dev/null | awk 'NR==2{print $5,$3,$2}'", { encoding: "utf-8" }).trim();
    const parts = diskInfo.split(" ");
    metrics.disk = parseInt(parts[0]) || 0;
    metrics.diskUsed = parts[1] || "?";
    metrics.diskTotal = parts[2] || "?";
  } catch {}
  try {
    metrics.uptime = execSync("uptime -p 2>/dev/null", { encoding: "utf-8" }).trim().replace("up ", "");
  } catch {}
  try {
    metrics.ip = execSync("hostname -I 2>/dev/null | awk '{print $1}'", { encoding: "utf-8" }).trim();
    if (!metrics.ip) {
      metrics.ip = execSync("curl -s --max-time 2 ifconfig.me 2>/dev/null", { encoding: "utf-8" }).trim();
    }
  } catch {}
  return metrics;
}

function renderGauge(label, percent, width = 22) {
  const filled = Math.round((percent / 100) * width);
  const empty = width - filled;
  const color = percent > 85 ? BRIGHT_RED : percent > 60 ? RED : DIM_RED;
  const bar = color("█".repeat(filled)) + chalk.dim("░".repeat(empty));
  const pct = (percent + "%").padStart(4);
  return `  ${chalk.dim(label.padEnd(5))} ${bar} ${color(pct)}`;
}

// ── Fixed header renderer ───────────────────────────────────────────────────

function getTermWidth() {
  try {
    return process.stdout.columns || 60;
  } catch {
    return 60;
  }
}

function renderFixedHeader() {
  const metrics = getSystemMetrics();
  const w = getTermWidth();
  const divider = RED("─".repeat(Math.min(w - 2, 56)));
  const lines = [];

  // Block header
  lines.push("");
  for (const row of BLOCK_HEADER) {
    lines.push("  " + BRIGHT_RED(row));
  }
  lines.push("");

  // Divider
  lines.push("  " + divider);

  // System gauges — compact row
  lines.push(renderGauge("CPU", metrics.cpu));
  lines.push(renderGauge("MEM", metrics.mem) + chalk.dim(` ${metrics.memUsed}/${metrics.memTotal}M`));
  lines.push(renderGauge("DISK", metrics.disk) + chalk.dim(` ${metrics.diskUsed}/${metrics.diskTotal}`));

  // Info line
  lines.push("");
  const model = config.get("default_model") || "llama3.3-70b-instruct";
  const shortModel = model.length > 30 ? model.slice(0, 27) + "..." : model;
  lines.push(
    "  " + chalk.dim("IP ") + RED(metrics.ip || "local") +
    chalk.dim("  •  Up ") + RED(metrics.uptime || "?") +
    chalk.dim("  •  Model ") + RED(shortModel)
  );

  // Bottom divider
  lines.push("  " + divider);
  lines.push(chalk.dim("  Type a message or /command • Ctrl+C to exit"));
  lines.push("");

  return lines.join("\n");
}

function getHeaderHeight() {
  // Block header (3) + blank + divider + 3 gauges + blank + info + divider + hint + blank = 12
  return 12;
}

// ── Screen management (fixed header + scrollable area) ──────────────────────

function clearAndRenderHeader() {
  // Clear screen, move cursor to top, render header
  process.stdout.write("\x1b[2J\x1b[H");
  process.stdout.write(renderFixedHeader());
}

function scrollToBottom() {
  // Move cursor after the fixed header for chat content
  const headerH = getHeaderHeight();
  process.stdout.write(`\x1b[${headerH + 1};1H`);
}

// ── Slash command autocomplete ──────────────────────────────────────────────

const COMMANDS = [
  "/help", "/clear", "/model", "/models", "/status", "/memory",
  "/creds", "/run", "/plan", "/agent", "/ping", "/skills",
  "/apis", "/connectors", "/mcp", "/usage", "/stop", "/task",
  "/providers", "/update",
];

function autocomplete(line) {
  if (!line.startsWith("/")) return [[], line];
  const hits = COMMANDS.filter((c) => c.startsWith(line));
  return [hits, line];
}

// ── Chat send function ──────────────────────────────────────────────────────

const chatHistory = [];

function printChatBubble(role, text) {
  const prefix = role === "user" ? chalk.dim("  you ▸ ") : RED("  ◂ ");
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    if (i === 0) {
      console.log(prefix + lines[i]);
    } else {
      console.log("        " + lines[i]);
    }
  }
}

async function sendMessage(message) {
  const apiKey = config.get("openai_api_key");
  const apiBase = config.get("openai_api_base");
  const model = config.get("default_model");

  if (!apiKey) {
    printError("No API key. Run: synthclaw setup");
    return;
  }

  // Show user message
  printChatBubble("user", message);
  chatHistory.push({ role: "user", content: message });

  process.stdout.write(chalk.dim("  ⏳ thinking...\r"));

  try {
    const msgs = [
      { role: "system", content: "You are SynthClaw, a personal AI assistant with server access. Be concise and direct. Use plain text, no markdown." },
      ...chatHistory.slice(-10),
    ];

    const response = await fetch(`${apiBase}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        model,
        messages: msgs,
        temperature: 0.7,
        max_tokens: 2048,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      process.stdout.write("                    \r"); // Clear spinner
      console.log(BRIGHT_RED(`  ✗ ${data.error?.message || `API error ${response.status}`}`));
      return;
    }

    const reply = data.choices?.[0]?.message?.content || "(empty)";
    const cleaned = reply.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

    process.stdout.write("                    \r"); // Clear spinner
    printChatBubble("assistant", cleaned);
    chatHistory.push({ role: "assistant", content: cleaned });
    console.log("");
  } catch (err) {
    process.stdout.write("                    \r");
    console.log(BRIGHT_RED(`  ✗ ${err.message}`));
  }
}

// ── Handle slash commands locally ───────────────────────────────────────────

async function handleCommand(input) {
  const [cmd, ...args] = input.split(" ");
  const arg = args.join(" ");

  switch (cmd) {
    case "/help":
      console.log("");
      console.log(RED("  Commands:"));
      for (const c of COMMANDS) {
        console.log(chalk.dim(`    ${c}`));
      }
      console.log("");
      break;
    case "/model":
      if (arg) {
        config.set("default_model", arg);
        console.log(chalk.dim(`  Switched to: `) + RED(arg));
      } else {
        console.log(chalk.dim(`  Current: `) + RED(config.get("default_model")));
      }
      console.log("");
      break;
    case "/status":
      clearAndRenderHeader();
      break;
    case "/clear":
      chatHistory.length = 0;
      clearAndRenderHeader();
      break;
    case "/run":
      if (!arg) {
        console.log(chalk.dim("  Usage: /run <command>"));
        break;
      }
      try {
        const output = execSync(arg, { encoding: "utf-8", timeout: 30000, cwd: getProjectRoot() });
        console.log(DIM_RED("  ┌─ output"));
        for (const line of output.trim().split("\n").slice(0, 30)) {
          console.log(DIM_RED("  │ ") + line);
        }
        console.log(DIM_RED("  └─"));
      } catch (err) {
        console.log(BRIGHT_RED(`  ✗ ${(err.stderr || err.message || "").slice(0, 200)}`));
      }
      console.log("");
      break;
    default:
      // Send as message to LLM (including the /command)
      await sendMessage(input);
  }
}

// ── Main dashboard loop ─────────────────────────────────────────────────────

export async function runDashboard() {
  // Initial render
  clearAndRenderHeader();

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

  // Refresh header every 30s (update system metrics)
  setInterval(() => {
    // Save cursor, go to top, re-render header, restore cursor
    const saved = "\x1b[s";
    const restored = "\x1b[u";
    process.stdout.write(saved + "\x1b[H" + renderFixedHeader() + restored);
  }, 30000);
}
