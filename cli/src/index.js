import chalk from "chalk";
import { SYNTHCLAW_ASCII } from "./ascii.js";

const args = process.argv.slice(2);
const command = args[0] || "help";

function showBanner() {
  console.log(chalk.hex("#e85d04")(SYNTHCLAW_ASCII));
  console.log(
    chalk.dim("  Personal AI Agent — Telegram & WhatsApp — Self-hosted\n")
  );
}

async function main() {
  switch (command) {
    case "setup":
      showBanner();
      const { runSetup } = await import("./commands/setup.js");
      await runSetup();
      break;

    case "start":
      const { runStart } = await import("./commands/start.js");
      await runStart(args.slice(1));
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

    case "agent":
      if (args.length <= 1) {
        // No args — launch interactive dashboard
        const { runDashboard } = await import("./commands/dashboard.js");
        await runDashboard();
      } else {
        const { runAgent } = await import("./commands/agent.js");
        await runAgent(args.slice(1));
      }
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

    case "help":
    case "--help":
    case "-h":
      showBanner();
      showHelp();
      break;

    case "--version":
    case "-v":
      console.log("synthclaw v1.0.0");
      break;

    default:
      console.log(chalk.red(`Unknown command: ${command}`));
      console.log(`Run ${chalk.cyan("synthclaw help")} to see available commands.\n`);
      process.exit(1);
  }
}

function showHelp() {
  console.log(chalk.bold("USAGE"));
  console.log(`  ${chalk.cyan("synthclaw")} ${chalk.yellow("<command>")} [options]\n`);

  console.log(chalk.bold("SETUP & LIFECYCLE"));
  console.log(
    `  ${chalk.cyan("setup")}          Interactive wizard — configure credentials & settings`
  );
  console.log(
    `  ${chalk.cyan("deploy")}         Deploy agent to a remote VPS (scp + setup)`
  );
  console.log(
    `  ${chalk.cyan("start")}          Start the agent (runs persistently in background)`
  );
  console.log(`  ${chalk.cyan("stop")}           Stop the running agent`);
  console.log(`  ${chalk.cyan("status")}         Show agent & service status`);
  console.log(`  ${chalk.cyan("logs")}           Tail agent logs`);
  console.log("");

  console.log(chalk.bold("AI & EXECUTION"));
  console.log(
    `  ${chalk.cyan("run")} <cmd>       Execute a shell command on the agent server`
  );
  console.log(
    `  ${chalk.cyan("plan")} <task>     Break a task into steps (no execution)`
  );
  console.log(
    `  ${chalk.cyan("agent")}          Interactive dashboard (no args) or autonomous exec (with task)`
  );
  console.log("");

  console.log(chalk.bold("MEMORY & CREDENTIALS"));
  console.log(`  ${chalk.cyan("memory")}         Show all remembered facts`);
  console.log(
    `  ${chalk.cyan("memory set")} <k> <v>  Remember a fact`
  );
  console.log(`  ${chalk.cyan("creds")}          List stored credentials`);
  console.log(
    `  ${chalk.cyan("creds set")} <name> <value>  Store an encrypted credential`
  );
  console.log("");

  console.log(chalk.bold("MODEL MANAGEMENT"));
  console.log(`  ${chalk.cyan("model")}          Show current model`);
  console.log(`  ${chalk.cyan("model")} <name>   Switch LLM model`);
  console.log(`  ${chalk.cyan("models")}         List all available models`);
  console.log("");

  console.log(chalk.bold("UTILITY"));
  console.log(`  ${chalk.cyan("ping")}           Check if the agent is alive`);
  console.log(`  ${chalk.cyan("clear")}          Wipe conversation history`);
  console.log(`  ${chalk.cyan("help")}           Show this help message`);
  console.log("");

  console.log(chalk.dim("Examples:"));
  console.log(chalk.dim("  synthclaw setup"));
  console.log(chalk.dim("  synthclaw start"));
  console.log(chalk.dim('  synthclaw agent "deploy a node server on port 3000"'));
  console.log(chalk.dim('  synthclaw run "systemctl status nginx"'));
  console.log("");
}

main().catch((err) => {
  console.error(chalk.red("Error:"), err.message);
  process.exit(1);
});
