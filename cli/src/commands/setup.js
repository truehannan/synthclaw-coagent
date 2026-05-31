import chalk from "chalk";
import ora from "ora";
import inquirer from "inquirer";
import { writeFileSync, existsSync } from "fs";
import { join } from "path";
import { execSync } from "child_process";
import { randomBytes } from "crypto";
import { config, generateEnvContent, getProjectRoot, printSuccess, printError, printInfo } from "../utils.js";

export async function runSetup() {
  console.log(
    chalk.bold("  Setup Wizard") +
      chalk.dim(" — Configure your SynthClaw agent\n")
  );

  // Step 1: Interface mode
  console.log(chalk.hex("#e85d04")("━".repeat(50)));
  console.log(chalk.bold("  1. INTERFACE MODE"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));

  const { interfaceMode } = await inquirer.prompt([
    {
      type: "list",
      name: "interfaceMode",
      message: "Which messaging interface(s)?",
      choices: [
        { name: "Telegram (polling)", value: "telegram" },
        { name: "WhatsApp (webhooks via Meta Cloud API)", value: "whatsapp" },
        { name: "Both (Telegram + WhatsApp)", value: "both" },
      ],
      default: config.get("interface_mode"),
    },
  ]);
  config.set("interface_mode", interfaceMode);

  // Step 2: Telegram config
  if (interfaceMode === "telegram" || interfaceMode === "both") {
    console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
    console.log(chalk.bold("  2. TELEGRAM CONFIGURATION"));
    console.log(chalk.hex("#e85d04")("━".repeat(50)));
    console.log(
      chalk.dim("  Get your bot token from @BotFather on Telegram.\n")
    );

    const { telegramToken } = await inquirer.prompt([
      {
        type: "password",
        name: "telegramToken",
        message: "Telegram Bot Token:",
        mask: "*",
        validate: (v) => (v.length > 10 ? true : "Token seems too short"),
        default: config.get("telegram_token") || undefined,
      },
    ]);
    config.set("telegram_token", telegramToken);
  }

  // Step 3: WhatsApp config
  if (interfaceMode === "whatsapp" || interfaceMode === "both") {
    console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
    console.log(chalk.bold("  3. WHATSAPP CONFIGURATION"));
    console.log(chalk.hex("#e85d04")("━".repeat(50)));
    console.log(
      chalk.dim(
        "  Meta WhatsApp Cloud API — developers.facebook.com/docs/whatsapp\n"
      )
    );

    const waAnswers = await inquirer.prompt([
      {
        type: "password",
        name: "whatsappToken",
        message: "WhatsApp API Access Token:",
        mask: "*",
        validate: (v) => (v.length > 10 ? true : "Token seems too short"),
        default: config.get("whatsapp_token") || undefined,
      },
      {
        type: "input",
        name: "whatsappPhoneId",
        message: "WhatsApp Phone Number ID:",
        validate: (v) => (v.length > 5 ? true : "ID seems too short"),
        default: config.get("whatsapp_phone_number_id") || undefined,
      },
      {
        type: "input",
        name: "whatsappVerifyToken",
        message: "Webhook Verify Token (auto-generated):",
        default:
          config.get("whatsapp_verify_token") ||
          randomBytes(16).toString("hex"),
      },
      {
        type: "input",
        name: "whatsappPort",
        message: "Webhook Port:",
        default: config.get("whatsapp_port") || "8443",
      },
    ]);
    config.set("whatsapp_token", waAnswers.whatsappToken);
    config.set("whatsapp_phone_number_id", waAnswers.whatsappPhoneId);
    config.set("whatsapp_verify_token", waAnswers.whatsappVerifyToken);
    config.set("whatsapp_port", waAnswers.whatsappPort);
  }

  // Step 4: LLM Provider
  console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
  console.log(chalk.bold("  4. LLM PROVIDER"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));
  console.log(
    chalk.dim(
      "  Any OpenAI-compatible API (OpenAI, DigitalOcean AI, Ollama, Groq)\n"
    )
  );

  const llmAnswers = await inquirer.prompt([
    {
      type: "password",
      name: "apiKey",
      message: "API Key:",
      mask: "*",
      validate: (v) => (v.length > 5 ? true : "Key seems too short"),
      default: config.get("openai_api_key") || undefined,
    },
    {
      type: "input",
      name: "apiBase",
      message: "API Base URL:",
      default:
        config.get("openai_api_base") || "https://inference.do-ai.run/v1",
    },
    {
      type: "list",
      name: "defaultModel",
      message: "Default Model:",
      choices: [
        "llama3.3-70b-instruct",
        "mistral-nemo-instruct-2407",
        "deepseek-r1-distill-llama-70b",
        "anthropic-claude-sonnet-4",
        "openai-gpt-4o",
        "openai-gpt-4o-mini",
        new inquirer.Separator(),
        { name: "Custom (type your own)", value: "__custom__" },
      ],
      default: config.get("default_model"),
    },
  ]);

  let model = llmAnswers.defaultModel;
  if (model === "__custom__") {
    const { customModel } = await inquirer.prompt([
      {
        type: "input",
        name: "customModel",
        message: "Custom model name:",
        validate: (v) => (v.length > 1 ? true : "Enter a model name"),
      },
    ]);
    model = customModel;
  }

  config.set("openai_api_key", llmAnswers.apiKey);
  config.set("openai_api_base", llmAnswers.apiBase);
  config.set("default_model", model);

  // Step 5: Server settings
  console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
  console.log(chalk.bold("  5. SERVER SETTINGS"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));

  const serverAnswers = await inquirer.prompt([
    {
      type: "input",
      name: "baseDir",
      message: "Installation directory on server:",
      default: config.get("base_dir") || "/opt/agent",
    },
    {
      type: "input",
      name: "remoteHost",
      message: "Remote server IP/hostname (leave blank for local):",
      default: config.get("remote_host") || "",
    },
    {
      type: "input",
      name: "remoteUser",
      message: "SSH user:",
      default: config.get("remote_user") || "root",
      when: (answers) => answers.remoteHost !== "",
    },
    {
      type: "input",
      name: "maxIterations",
      message: "Max tool iterations per message:",
      default: config.get("max_tool_iterations") || "10",
    },
    {
      type: "input",
      name: "maxHistory",
      message: "Max conversation history messages:",
      default: config.get("max_history_messages") || "20",
    },
  ]);

  config.set("base_dir", serverAnswers.baseDir);
  config.set("remote_host", serverAnswers.remoteHost || "");
  config.set("remote_user", serverAnswers.remoteUser || "root");
  config.set("max_tool_iterations", serverAnswers.maxIterations);
  config.set("max_history_messages", serverAnswers.maxHistory);

  // Step 6: Write .env file
  const spinner1 = ora("Writing .env configuration...").start();
  const root = getProjectRoot();
  try {
    const envContent = generateEnvContent();
    const envPath = join(root, ".env");
    writeFileSync(envPath, envContent);
    spinner1.succeed(`Configuration saved to ${envPath}`);
  } catch (err) {
    spinner1.warn("Could not write local .env (will use stored config)");
  }

  // Step 7: Install Python dependencies locally
  console.log("");
  const { installDeps } = await inquirer.prompt([
    {
      type: "confirm",
      name: "installDeps",
      message: "Install Python dependencies now? (creates venv + pip install)",
      default: true,
    },
  ]);

  if (installDeps) {
    const venvPath = join(root, "venv");
    const requirementsPath = join(root, "requirements.txt");

    if (!existsSync(requirementsPath)) {
      printError(`requirements.txt not found at ${requirementsPath}`);
    } else {
      // Detect python3 binary
      let pythonBin = "python3";
      try {
        execSync(`${pythonBin} --version`, { encoding: "utf-8" });
      } catch {
        pythonBin = "python";
        try {
          execSync(`${pythonBin} --version`, { encoding: "utf-8" });
        } catch {
          printError("Python 3 not found. Please install Python 3.10+ first.");
          pythonBin = null;
        }
      }

      if (pythonBin) {
        // Create venv
        const spinnerVenv = ora("Creating Python virtual environment...").start();
        try {
          if (!existsSync(venvPath)) {
            execSync(`${pythonBin} -m venv "${venvPath}"`, {
              encoding: "utf-8",
              timeout: 30000,
              cwd: root,
            });
          }
          spinnerVenv.succeed("Virtual environment ready");
        } catch (err) {
          spinnerVenv.fail("Could not create venv: " + err.message);
        }

        // Install dependencies
        if (existsSync(venvPath)) {
          const spinnerPip = ora("Installing Python dependencies (this may take a minute)...").start();
          try {
            const pipBin = join(venvPath, "bin", "pip");
            execSync(`"${pipBin}" install --upgrade pip -q`, {
              encoding: "utf-8",
              timeout: 60000,
              cwd: root,
            });
            execSync(`"${pipBin}" install -r "${requirementsPath}" -q`, {
              encoding: "utf-8",
              timeout: 180000,
              cwd: root,
            });
            spinnerPip.succeed("Python dependencies installed");
          } catch (err) {
            spinnerPip.fail("pip install failed: " + (err.stderr || err.message).slice(0, 200));
            printInfo("You can retry manually: venv/bin/pip install -r requirements.txt");
          }
        }
      }
    }

    // Create workspace directory
    const workspaceDir = join(root, "workspace");
    try {
      execSync(`mkdir -p "${workspaceDir}"`, { encoding: "utf-8" });
    } catch {}
  }

  // Summary
  console.log(chalk.hex("#e85d04")("\n━".repeat(50)));
  console.log(chalk.bold("  ✓ SETUP COMPLETE"));
  console.log(chalk.hex("#e85d04")("━".repeat(50)));
  console.log("");
  printSuccess(`Interface: ${config.get("interface_mode")}`);
  printSuccess(`Model: ${config.get("default_model")}`);
  printSuccess(`API: ${config.get("openai_api_base")}`);
  if (config.get("remote_host")) {
    printSuccess(
      `Remote: ${config.get("remote_user")}@${config.get("remote_host")}`
    );
  } else {
    printSuccess("Mode: Local (agent runs on this machine)");
  }
  console.log("");
  printInfo("Next steps:");
  if (config.get("remote_host")) {
    console.log(chalk.dim("  synthclaw deploy    # deploy to your VPS"));
    console.log(chalk.dim("  synthclaw start     # start the agent (on VPS)"));
  } else {
    console.log(chalk.dim("  synthclaw start     # start the agent locally"));
  }
  console.log(chalk.dim("  synthclaw status    # check if running"));
  console.log(chalk.dim("  synthclaw logs      # view agent output"));
  console.log("");
}
