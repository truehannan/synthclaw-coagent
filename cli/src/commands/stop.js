import chalk from "chalk";
import ora from "ora";
import { remoteExec, isAgentRunning, printSuccess, printError, printInfo } from "../utils.js";

export async function runStop() {
  if (!isAgentRunning()) {
    printInfo("Agent is not running.");
    return;
  }

  const spinner = ora("Stopping SynthClaw agent...").start();

  try {
    remoteExec("systemctl stop agent");
    await new Promise((r) => setTimeout(r, 1000));

    if (!isAgentRunning()) {
      spinner.succeed("Agent stopped.");
    } else {
      spinner.warn("Agent may still be stopping...");
      remoteExec("systemctl kill agent");
      printInfo("Force killed.");
    }
  } catch (err) {
    spinner.fail("Failed to stop agent");
    printError(err.message);
  }
}
