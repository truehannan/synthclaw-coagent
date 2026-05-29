import chalk from "chalk";
import ora from "ora";
import inquirer from "inquirer";
import { execSync } from "child_process";
import { config, getProjectRoot, generateEnvContent, printSuccess, printError, printInfo } from "../utils.js";

export async function runDeploy(args) {
  let host = config.get("remote_host");
  let user = config.get("remote_user") || "root";

  if (!host) {
    const answers = await inquirer.prompt([
      {
        type: "input",
        name: "host",
        message: "Remote server IP/hostname:",
        validate: (v) => (v.length > 0 ? true : "Required"),
      },
      {
        type: "input",
        name: "user",
        message: "SSH user:",
        default: "root",
      },
    ]);
    host = answers.host;
    user = answers.user;
    config.set("remote_host", host);
    config.set("remote_user", user);
  }

  const baseDir = config.get("base_dir");
  const root = getProjectRoot();

  console.log(chalk.bold("\n  Deploying SynthClaw to ") + chalk.cyan(`${user}@${host}:${baseDir}`));
  console.log("");

  // Step 1: Upload files
  const spinner1 = ora("Uploading agent files...").start();
  try {
    execSync(
      `ssh ${user}@${host} 'mkdir -p ${baseDir}'`,
      { encoding: "utf-8", timeout: 10000 }
    );
    // Upload Python files + requirements
    execSync(
      `scp -r ${root}/main.py ${root}/agent.py ${root}/whatsapp_bot.py ${root}/tools.py ${root}/memory.py ${root}/config.py ${root}/requirements.txt ${root}/setup_server.sh ${user}@${host}:${baseDir}/`,
      { encoding: "utf-8", timeout: 60000 }
    );
    spinner1.succeed("Files uploaded");
  } catch (err) {
    spinner1.fail("Upload failed");
    printError(err.message);
    return;
  }

  // Step 2: Write .env on remote
  const spinner2 = ora("Writing .env configuration...").start();
  try {
    const envContent = generateEnvContent();
    const escaped = envContent.replace(/'/g, "'\\''");
    execSync(
      `ssh ${user}@${host} 'cat > ${baseDir}/.env << '"'"'ENVEOF'"'"'\n${escaped}\nENVEOF'`,
      { encoding: "utf-8", timeout: 10000 }
    );
    execSync(`ssh ${user}@${host} 'chmod 600 ${baseDir}/.env'`, {
      encoding: "utf-8",
      timeout: 5000,
    });
    spinner2.succeed(".env written");
  } catch (err) {
    spinner2.fail(".env write failed");
    printError(err.message);
    return;
  }

  // Step 3: Run setup script
  const spinner3 = ora("Running server setup (installing deps)...").start();
  try {
    execSync(
      `ssh ${user}@${host} 'bash ${baseDir}/setup_server.sh'`,
      { encoding: "utf-8", timeout: 300000 }
    );
    spinner3.succeed("Server setup complete");
  } catch (err) {
    spinner3.fail("Server setup had issues");
    printError(err.stderr || err.message);
    printInfo("You may need to SSH in and check.");
  }

  // Step 4: Start agent
  const spinner4 = ora("Starting agent...").start();
  try {
    execSync(`ssh ${user}@${host} 'systemctl restart agent'`, {
      encoding: "utf-8",
      timeout: 10000,
    });
    await new Promise((r) => setTimeout(r, 2000));
    const status = execSync(
      `ssh ${user}@${host} 'systemctl is-active agent'`,
      { encoding: "utf-8", timeout: 5000 }
    ).trim();

    if (status === "active") {
      spinner4.succeed("Agent is running!");
    } else {
      spinner4.warn("Agent started but status: " + status);
    }
  } catch (err) {
    spinner4.fail("Could not start agent");
    printError(err.message);
  }

  console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
  console.log(chalk.bold("  ✓ DEPLOYMENT COMPLETE"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));
  console.log("");
  printSuccess(`Agent deployed to ${user}@${host}`);
  printInfo("Open Telegram and send /start to your bot!");
  console.log("");
  console.log(chalk.dim("  synthclaw status   — check status"));
  console.log(chalk.dim("  synthclaw logs     — view logs"));
  console.log(chalk.dim("  synthclaw stop     — stop the agent"));
  console.log("");
}
