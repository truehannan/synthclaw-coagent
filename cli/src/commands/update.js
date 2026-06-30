import chalk from "chalk";
import ora from "ora";
import { execSync } from "child_process";
import { existsSync, join } from "fs";
import { getProjectRoot, printSuccess, printError, printInfo } from "../utils.js";

export async function runUpdate() {
  const root = getProjectRoot();
  printInfo(`Project: ${root}\n`);

  // Step 1: git pull
  const s1 = ora("Pulling latest from GitHub...").start();
  try {
    const output = execSync("git pull origin main --rebase 2>&1 || git pull origin main 2>&1", {
      encoding: "utf-8", cwd: root, timeout: 30000,
    });
    if (output.includes("Already up to date")) {
      s1.succeed("Already up to date");
    } else {
      s1.succeed("Updated from GitHub");
    }
  } catch (err) {
    s1.fail("git pull failed: " + (err.stderr || err.message).slice(0, 100));
    printInfo("Try manually: cd " + root + " && git pull");
    return;
  }

  // Step 2: npm install + build CLI
  const s2 = ora("Installing CLI dependencies...").start();
  try {
    execSync("npm install --prefix cli", { encoding: "utf-8", cwd: root, timeout: 60000 });
    s2.succeed("CLI deps installed");
  } catch (err) {
    s2.fail("npm install failed");
    printError((err.stderr || err.message).slice(0, 150));
    return;
  }

  const s3 = ora("Building CLI...").start();
  try {
    execSync("npm run build --prefix cli", { encoding: "utf-8", cwd: root, timeout: 30000 });
    s3.succeed("CLI built");
  } catch (err) {
    s3.fail("Build failed");
    printError((err.stderr || err.message).slice(0, 150));
    return;
  }

  // Step 3: npm link
  const s4 = ora("Linking synthclaw command...").start();
  try {
    execSync("npm link --prefix cli", { encoding: "utf-8", cwd: root, timeout: 15000 });
    s4.succeed("synthclaw command linked");
  } catch (err) {
    s4.warn("Link may have failed — try: cd cli && npm link");
  }

  // Step 4: Update Python deps if venv exists
  const venvPip = root + "/venv/bin/pip";
  const reqFile = root + "/requirements.txt";
  if (existsSync(venvPip) && existsSync(reqFile)) {
    const s5 = ora("Updating Python dependencies...").start();
    try {
      execSync(`${venvPip} install -r ${reqFile} -q`, { encoding: "utf-8", cwd: root, timeout: 120000 });
      s5.succeed("Python deps updated");
    } catch {
      s5.warn("Python deps update skipped (run: venv/bin/pip install -r requirements.txt)");
    }
  }

  console.log("");
  printSuccess("Update complete. Run: synthclaw start");
  console.log("");
}
