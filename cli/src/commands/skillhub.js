import chalk from "chalk";
import ora from "ora";
import { mkdirSync, writeFileSync } from "fs";
import { join } from "path";
import { getProjectRoot, printSuccess, printInfo } from "../utils.js";

const CLAWHUB_API = "https://clawhub.ai/api/v1";

export async function runImport(args) {
  if (!args[0] || args[0] !== "skill" || !args[1]) {
    console.log(chalk.bold("\n  Usage:"));
    console.log(`    ${chalk.cyan("synthclaw import skill")} ${chalk.yellow("@owner/name")}`);
    console.log(`    ${chalk.cyan("synthclaw import skill")} ${chalk.yellow("name")}`);
    console.log(chalk.dim("\n  Imports from ClawHub registry (clawhub.ai)\n"));
    return;
  }

  const skillRef = args[1];
  const root = getProjectRoot();
  const skillsDir = join(root, ".skills");
  mkdirSync(skillsDir, { recursive: true });

  let owner = "", name = skillRef;
  if (skillRef.startsWith("@")) {
    const parts = skillRef.slice(1).split("/");
    owner = parts[0] || "";
    name = parts[1] || parts[0];
  }

  const spinner = ora(`Fetching: ${skillRef}`).start();
  try {
    const q = owner ? `${owner}/${name}` : name;
    const resp = await fetch(`${CLAWHUB_API}/search?q=${encodeURIComponent(q)}&limit=1`);
    if (!resp.ok) { spinner.fail(`API error: ${resp.status}`); return; }
    const data = await resp.json();
    const results = data.results || [];
    if (!results.length) { spinner.fail(`Not found: ${skillRef}`); printInfo("Try: synthclaw search skill <keyword>"); return; }

    const skill = results[0];
    const slug = skill.slug;
    const skillOwner = skill.ownerHandle || skill.owner?.handle || "unknown";
    spinner.text = `Installing: @${skillOwner}/${slug}`;

    const skillDir = join(skillsDir, slug);
    mkdirSync(skillDir, { recursive: true });
    const md = `# ${skill.displayName || slug}\n\n${skill.summary || ""}\n\nSource: ClawHub @${skillOwner}/${slug}\nVersion: ${skill.version || "latest"}\n`;
    writeFileSync(join(skillDir, "SKILL.md"), md);
    spinner.succeed(`Installed: @${skillOwner}/${slug}`);
    printSuccess(`Path: ${skillDir}`);
  } catch (err) { spinner.fail(err.message); }
}

export async function runSearch(args) {
  if (!args[0] || args[0] !== "skill" || !args[1]) {
    console.log(chalk.bold("\n  Usage:"));
    console.log(`    ${chalk.cyan("synthclaw search skill")} ${chalk.yellow("<query>")}`);
    console.log(chalk.dim("\n  Searches ClawHub registry (clawhub.ai)\n"));
    return;
  }

  const query = args.slice(1).join(" ");
  const spinner = ora(`Searching: "${query}"`).start();
  try {
    const resp = await fetch(`${CLAWHUB_API}/search?q=${encodeURIComponent(query)}&limit=15`);
    if (!resp.ok) { spinner.fail(`API error: ${resp.status}`); return; }
    const data = await resp.json();
    const results = data.results || [];
    if (!results.length) { spinner.info(`No results for: ${query}`); return; }

    spinner.succeed(`${results.length} result(s)\n`);
    for (const s of results) {
      const o = s.ownerHandle || s.owner?.handle || "?";
      console.log(`  ${chalk.hex("#e85d04")("@" + o + "/")}${chalk.bold(s.slug)}${s.displayName && s.displayName !== s.slug ? chalk.dim(` (${s.displayName})`) : ""}`);
      if (s.summary) console.log(chalk.dim(`    ${s.summary.slice(0, 80)}`));
      console.log("");
    }
    console.log(chalk.dim("  Install: synthclaw import skill @owner/name\n"));
  } catch (err) { spinner.fail(err.message); }
}
