import chalk from "chalk";
import ora from "ora";
import { execSync } from "child_process";
import { getProjectRoot, printSuccess, printError, printInfo } from "../utils.js";

export async function runUpdate() {
  const root = getProjectRoot();
  printInfo(`Project: ${root}\n`);

  // Step 1: git fetch
  const s1 = ora("Fetching latest from origin...").start();
  try {
    execSync("git fetch origin", { encoding: "utf-8", cwd: root, timeout: 30000, stdio: ["pipe", "pipe", "pipe"] });
    s1.succeed("Fetched origin");
  } catch (err) {
    s1.fail("git fetch failed");
    printError((err.stderr || err.message || "").slice(0, 150));
    return;
  }

  // Step 2: git pull (merge)
  const s2 = ora("Merging latest changes...").start();
  try {
    const output = execSync("git pull origin main 2>&1", { encoding: "utf-8", cwd: root, timeout: 15000 });
    if (output.includes("Already up to date")) {
      s2.succeed("Already up to date");
    } else {
      s2.succeed("Updated to latest");
    }
  } catch {
    // Try master branch
    try {
      execSync("git pull origin master 2>&1", { encoding: "utf-8", cwd: root, timeout: 15000 });
      s2.succeed("Updated to latest (master)");
    } catch (err2) {
      s2.fail("Pull failed");
      printError((err2.stderr || err2.message || "").slice(0, 150));
      printInfo("Try manually: git pull origin main");
      return;
    }
  }

  console.log("");
  printSuccess("Update complete!");
  printInfo("Now run: npm run setup");
  console.log("");
}
