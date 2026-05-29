import chalk from "chalk";
import ora from "ora";
import { config, remoteExec, isAgentRunning, printSuccess, printError, printInfo } from "../utils.js";

export async function runStart(args) {
  const foreground = args.includes("--foreground") || args.includes("-f");

  if (isAgentRunning()) {
    printInfo("Agent is already running.");
    console.log(chalk.dim("  Use: synthclaw stop   to stop it first"));
    console.log(chalk.dim("  Use: synthclaw logs   to view output"));
    return;
  }

  const spinner = ora("Starting SynthClaw agent...").start();

  try {
    if (foreground) {
      spinner.info("Starting in foreground mode (Ctrl+C to stop)...");
      const baseDir = config.get("base_dir");
      // Run directly — will block until interrupted
      remoteExec(
        `cd ${baseDir} && source venv/bin/activate && python main.py`,
        { stdio: "inherit", timeout: 0 }
      );
    } else {
      // Start via systemd for persistent background operation
      remoteExec("systemctl start agent");
      
      // Wait a moment and verify
      await new Promise((r) => setTimeout(r, 2000));
      
      if (isAgentRunning()) {
        spinner.succeed("Agent started successfully (running as systemd service)");
        printInfo("The agent will persist until the machine is stopped.");
        console.log(chalk.dim("  synthclaw logs     — view output"));
        console.log(chalk.dim("  synthclaw status   — check status"));
        console.log(chalk.dim("  synthclaw stop     — stop the agent"));
      } else {
        spinner.fail("Agent failed to start");
        printError("Check logs: synthclaw logs");
        try {
          const output = remoteExec("journalctl -u agent --no-pager -n 20");
          console.log(chalk.dim(output));
        } catch {}
      }
    }
  } catch (err) {
    spinner.fail("Failed to start agent");
    printError(err.message);
  }
}
