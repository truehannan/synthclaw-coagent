import chalk from "chalk";
import { remoteExec, config, printSuccess, printError, printInfo } from "../utils.js";

export async function runCreds(args) {
  const subcommand = args[0];

  if (subcommand === "set" && args.length >= 3) {
    const name = args[1];
    const value = args.slice(2).join(" ");
    try {
      const baseDir = config.get("base_dir");
      remoteExec(
        `cd ${baseDir} && source venv/bin/activate && python -c "import memory; memory.init_db(); memory.store_credential('${name.replace(/'/g, "\\'")}', '${value.replace(/'/g, "\\'")}')"`,
        { timeout: 10000 }
      );
      printSuccess(`Stored credential: ${name}`);
    } catch (err) {
      printError("Failed to store credential: " + err.message);
    }
    return;
  }

  if (subcommand === "get" && args[1]) {
    const name = args[1];
    try {
      const baseDir = config.get("base_dir");
      const output = remoteExec(
        `cd ${baseDir} && source venv/bin/activate && python -c "import memory; memory.init_db(); v=memory.get_credential('${name.replace(/'/g, "\\'")}'); print(v if v else '(not found)')"`,
        { timeout: 10000 }
      );
      console.log(chalk.bold(`  ${name}: `) + output.trim());
    } catch (err) {
      printError("Failed to retrieve credential: " + err.message);
    }
    return;
  }

  // Default: list all credentials (values hidden)
  try {
    const baseDir = config.get("base_dir");
    const output = remoteExec(
      `cd ${baseDir} && source venv/bin/activate && python -c "
import memory
memory.init_db()
creds = memory.list_credentials()
if creds:
    for c in creds:
        print(f'  {c[\"name\"]:30s} {c.get(\"description\", \"\")}')
else:
    print('  (no credentials stored)')
"`,
      { timeout: 10000 }
    );
    console.log(chalk.bold("\n  Stored Credentials") + chalk.dim(" (values encrypted)\n"));
    console.log(output);
  } catch (err) {
    printError("Failed to list credentials: " + err.message);
    printInfo("Make sure the agent is deployed and accessible.");
  }
}
