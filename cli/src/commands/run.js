import chalk from "chalk";
import { remoteExec, printError } from "../utils.js";

export async function runRun(args) {
  const cmd = args.join(" ").trim();

  if (!cmd) {
    console.log(chalk.bold("\n  Usage: ") + "conclave run <shell command>");
    console.log(chalk.dim("  Example: conclave run ls -la /opt/agent\n"));
    return;
  }

  try {
    const output = remoteExec(cmd, { timeout: 30000 });
    if (output.trim()) {
      console.log(output);
    } else {
      console.log(chalk.dim("(no output)"));
    }
  } catch (err) {
    if (err.stdout) console.log(err.stdout);
    if (err.stderr) console.error(chalk.red(err.stderr));
    printError(`Command exited with code ${err.status || 1}`);
  }
}
