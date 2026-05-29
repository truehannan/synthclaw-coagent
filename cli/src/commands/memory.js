import chalk from "chalk";
import { remoteExec, config, printSuccess, printError, printInfo } from "../utils.js";

export async function runMemory(args) {
  const subcommand = args[0];

  if (subcommand === "set" && args.length >= 3) {
    const key = args[1];
    const value = args.slice(2).join(" ");
    try {
      const baseDir = config.get("base_dir");
      remoteExec(
        `cd ${baseDir} && source venv/bin/activate && python -c "import memory; memory.init_db(); memory.set_memory('${key.replace(/'/g, "\\'")}', '${value.replace(/'/g, "\\'")}')"`,
        { timeout: 10000 }
      );
      printSuccess(`Remembered: ${key} = ${value}`);
    } catch (err) {
      printError("Failed to store memory: " + err.message);
    }
    return;
  }

  if (subcommand === "get" && args[1]) {
    const key = args[1];
    try {
      const baseDir = config.get("base_dir");
      const output = remoteExec(
        `cd ${baseDir} && source venv/bin/activate && python -c "import memory; memory.init_db(); v=memory.get_memory('${key.replace(/'/g, "\\'")}'); print(v if v else '(not found)')"`,
        { timeout: 10000 }
      );
      console.log(chalk.bold(`  ${key}: `) + output.trim());
    } catch (err) {
      printError("Failed to retrieve memory: " + err.message);
    }
    return;
  }

  // Default: show all memories
  try {
    const baseDir = config.get("base_dir");
    const output = remoteExec(
      `cd ${baseDir} && source venv/bin/activate && python -c "
import memory, json
memory.init_db()
m = memory.get_all_memory()
if m:
    for k, v in m.items():
        print(f'  {k}: {v}')
else:
    print('  (no memories stored)')
"`,
      { timeout: 10000 }
    );
    console.log(chalk.bold("\n  Stored Memories\n"));
    console.log(output);
  } catch (err) {
    printError("Failed to retrieve memories: " + err.message);
    printInfo("Make sure the agent is deployed and accessible.");
  }
}
