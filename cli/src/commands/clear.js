import chalk from "chalk";
import inquirer from "inquirer";
import { remoteExec, config, printSuccess, printError } from "../utils.js";

export async function runClear() {
  const { confirm } = await inquirer.prompt([
    {
      type: "confirm",
      name: "confirm",
      message: "Clear all conversation history? This cannot be undone.",
      default: false,
    },
  ]);

  if (!confirm) {
    console.log(chalk.dim("  Cancelled."));
    return;
  }

  try {
    const baseDir = config.get("base_dir");
    remoteExec(
      `cd ${baseDir} && source venv/bin/activate && python -c "
import memory
memory.init_db()
import sqlite3
from config import DB_PATH
conn = sqlite3.connect(DB_PATH)
conn.execute('DELETE FROM messages')
conn.commit()
conn.close()
print('done')
"`,
      { timeout: 10000 }
    );
    printSuccess("Conversation history cleared.");
  } catch (err) {
    printError("Failed to clear history: " + err.message);
  }
}
