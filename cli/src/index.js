import chalk from "chalk";
import { MASCOT_OPEN, WORDMARK, SUBTITLE } from "./ascii.js";

const args = process.argv.slice(2);
const command = args[0] || ""; // Default: launch dashboard (no command)

const RED = chalk.hex("#cc0000");

function showBanner() {
  for (const row of MASCOT_OPEN) {
    console.log("  " + RED(row));
  }
  for (const row of WORDMARK) {
    console.log("  " + RED(row));
  }
  console.log("  " + chalk.dim(SUBTITLE));
  console.log("");
}

async function main() {
  switch (command) {
    case "":
    case "agent":
      // No args or `agent` with no task = launch interactive dashboard
      if (command === "agent" && args.length > 1) {
        const { runAgent } = await import("./commands/agent.js");
        await runAgent(args.slice(1));
      } else {
        const { runDashboard } = await import("./commands/dashboard.js");
        await runDashboard();
      }
      break;

    case "setup":
      // Setup now launches dashboard which auto-triggers wizard if unconfigured
      // But also support standalone for backward compat
      const { runSetup } = await import("./commands/setup.js");
      showBanner();
      await runSetup();
      break;

    case "start":
      // If remote host configured, start the agent service remotely
      // Otherwise launch the interactive dashboard
      const { config: startConfig } = await import("./utils.js");
      if (startConfig.get("remote_host") || startConfig.get("interface_mode") === "telegram" || startConfig.get("interface_mode") === "whatsapp" || startConfig.get("interface_mode") === "both") {
        const { runStart } = await import("./commands/start.js");
        await runStart(args.slice(1));
      } else {
        // CLI-only mode: launch dashboard
        const { runDashboard: runDash } = await import("./commands/dashboard.js");
        await runDash();
      }
      break;

    case "stop":
      const { runStop } = await import("./commands/stop.js");
      await runStop();
      break;

    case "status":
      const { runStatus } = await import("./commands/status.js");
      await runStatus();
      break;

    case "logs":
      const { runLogs } = await import("./commands/logs.js");
      await runLogs(args.slice(1));
      break;

    case "model":
      const { runModel } = await import("./commands/model.js");
      await runModel(args.slice(1));
      break;

    case "models":
      const { runModels } = await import("./commands/models.js");
      await runModels();
      break;

    case "run":
      const { runRun } = await import("./commands/run.js");
      await runRun(args.slice(1));
      break;

    case "plan":
      const { runPlan } = await import("./commands/plan.js");
      await runPlan(args.slice(1));
      break;

    case "memory":
      const { runMemory } = await import("./commands/memory.js");
      await runMemory(args.slice(1));
      break;

    case "creds":
      const { runCreds } = await import("./commands/creds.js");
      await runCreds(args.slice(1));
      break;

    case "clear":
      const { runClear } = await import("./commands/clear.js");
      await runClear();
      break;

    case "ping":
      const { runPing } = await import("./commands/ping.js");
      await runPing();
      break;

    case "deploy":
      const { runDeploy } = await import("./commands/deploy.js");
      await runDeploy(args.slice(1));
      break;

    case "session":
    case "sessions":
      // Session management
      const sessionArg = args[1] || "";
      if (sessionArg) {
        // Switch to session by ID
        const { config: sessConfig } = await import("./utils.js");
        sessConfig.set("active_session", sessionArg);
        console.log(chalk.green("✓") + " Switched to session: " + sessionArg);
      } else {
        // List sessions
        console.log(chalk.bold("\n  Sessions:"));
        const { config: sc } = await import("./utils.js");
        const active = sc.get("active_session") || "default";
        console.log("  " + chalk.green("●") + " " + active + chalk.dim(" (active)"));
        console.log(chalk.dim("\n  Usage: synthclaw session <id> — switch session"));
        console.log(chalk.dim("  Sessions are managed in the web frontend.\n"));
      }
      break;

    case "update":
      const { runUpdate } = await import("./commands/update.js");
      await runUpdate();
      break;

    case "import":
      const { runImport } = await import("./commands/skillhub.js");
      await runImport(args.slice(1));
      break;

    case "search":
      const { runSearch } = await import("./commands/skillhub.js");
      await runSearch(args.slice(1));
      break;

    case "help":
    case "--help":
    case "-h":
      showBanner();
      showHelp();
      break;

    case "--version":
    case "-v":
      console.log("synthclaw agent-society v3.0.0");
      break;

    default:
      console.log(chalk.red(`Unknown command: ${command}`));
      console.log(`Run ${chalk.cyan("synthclaw help")} to see available commands.\n`);
      process.exit(1);
  }
}

function showHelp() {
  const c = chalk.cyan;
  const d = chalk.dim;
  console.log(chalk.bold("USAGE"));
  console.log(`  ${c("synthclaw")} ${chalk.yellow("[command]")} [options]\n`);
  console.log(d("  No command = launch interactive dashboard (setup + chat + AI)\n"));

  console.log(chalk.bold("MAIN"));
  console.log(`  ${c("(no command)")}   Launch JARVIS-like dashboard (auto-setup if needed)`);
  console.log(`  ${c("start")}          Start agent service (Telegram/WhatsApp) or dashboard (CLI mode)`);
  console.log(`  ${c("setup")}          Run setup wizard standalone`);
  console.log("");

  console.log(chalk.bold("LIFECYCLE"));
  console.log(`  ${c("stop")}           Stop the running agent`);
  console.log(`  ${c("status")}         Show agent & service status`);
  console.log(`  ${c("logs")}           Tail agent logs`);
  console.log(`  ${c("update")}         Hard-reset to latest upstream + rebuild`);
  console.log(`  ${c("deploy")}         Deploy agent to a remote VPS`);
  console.log("");

  console.log(chalk.bold("AI & EXECUTION"));
  console.log(`  ${c("agent")} <task>    Autonomous execution (with task arg)`);
  console.log(`  ${c("run")} <cmd>       Execute a shell command`);
  console.log(`  ${c("plan")} <task>     Break task into steps (no execution)`);
  console.log(`  ${c("models")}         List all available models (live fetch)`);
  console.log(`  ${c("model")} <name>    Switch LLM model`);
  console.log("");

  console.log(chalk.bold("DATA"));
  console.log(`  ${c("memory")}         Show remembered facts`);
  console.log(`  ${c("creds")}          List stored credentials`);
  console.log(`  ${c("clear")}          Wipe conversation history`);
  console.log(`  ${c("ping")}           Check if agent is alive`);
  console.log("");

  console.log(d("Examples:"));
  console.log(d("  synthclaw                     # Launch dashboard"));
  console.log(d("  synthclaw start               # Start service or dashboard"));
  console.log(d('  synthclaw agent "deploy nginx" # Autonomous task'));
  console.log("");
}

main().catch((err) => {
  console.error(chalk.red("Error:"), err.message);
  process.exit(1);
});
