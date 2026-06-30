import chalk from "chalk";
import ora from "ora";
import { execSync } from "child_process";
import { existsSync } from "fs";
import { getProjectRoot, printSuccess, printError, printInfo } from "../utils.js";

export async function runUpdate() {
  const root = getProjectRoot();
  printInfo(`Project: ${root}\n`);

  // Step 1: git fetch origin
  const s1 = ora("Fetching latest from origin...").start();
  try {
    execSync("git fetch origin", {
      encoding: "utf-8", cwd: root, timeout: 30000,
    });
    s1.succeed("Fetched origin");
  } catch (err) {
    s1.fail("git fetch failed: " + (err.stderr || err.message).slice(0, 150));
    printError("Check your internet connection and git remote.");
    return;
  }

  // Step 2: git reset --hard origin/main (hard reset to upstream)
  const s2 = ora("Resetting to origin/main...").start();
  try {
    const output = execSync("git reset --hard origin/main 2>&1", {
      encoding: "utf-8", cwd: root, timeout: 15000,
    });
    s2.succeed("Reset to latest upstream");
  } catch (err) {
    // Try origin/master as fallback
    try {
      execSync("git reset --hard origin/master 2>&1", {
        encoding: "utf-8", cwd: root, timeout: 15000,
      });
      s2.succeed("Reset to latest upstream (master)");
    } catch (err2) {
      s2.fail("git reset failed: " + (err2.stderr || err2.message).slice(0, 150));
      printError("Could not reset to upstream. Check branch name.");
      return;
    }
  }

  // Step 3: npm install (CLI deps)
  const s3 = ora("Installing CLI dependencies...").start();
  try {
    execSync("npm install --prefix cli", { encoding: "utf-8", cwd: root, timeout: 60000 });
    s3.succeed("CLI deps installed");
  } catch (err) {
    s3.fail("npm install failed");
    printError((err.stderr || err.message).slice(0, 150));
    return;
  }

  // Step 4: npm run build (CLI)
  const s4 = ora("Building CLI...").start();
  try {
    execSync("npm run build --prefix cli", { encoding: "utf-8", cwd: root, timeout: 30000 });
    s4.succeed("CLI built");
  } catch (err) {
    s4.fail("Build failed");
    printError((err.stderr || err.message).slice(0, 150));
    return;
  }

  // Step 5: npm link (make synthclaw command available globally)
  const s5 = ora("Linking synthclaw command...").start();
  try {
    execSync("npm link --prefix cli", { encoding: "utf-8", cwd: root, timeout: 15000 });
    s5.succeed("synthclaw command linked");
  } catch (err) {
    s5.warn("Link may have failed — try: cd cli && npm link");
  }

  // Step 6: Update Python deps if venv exists
  const venvPip = root + "/venv/bin/pip";
  const reqFile = root + "/requirements.txt";
  if (existsSync(venvPip) && existsSync(reqFile)) {
    const s6 = ora("Updating Python dependencies...").start();
    try {
      execSync(`${venvPip} install -r ${reqFile} -q`, { encoding: "utf-8", cwd: root, timeout: 180000 });
      s6.succeed("Python deps updated");
    } catch (err) {
      s6.warn("Python deps update failed — run: venv/bin/pip install -r requirements.txt");
    }
  } else if (existsSync(reqFile)) {
    printInfo("No venv found. Create one: python3 -m venv venv && venv/bin/pip install -r requirements.txt");
  }

  console.log("");
  printSuccess("Update complete! Hard-reset to latest upstream + full rebuild.");
  printInfo("Run: synthclaw start");
  console.log("");
}
