"""
SynthClaw-CoAgent — Telegram Bot Interface
Main entry point for the Telegram agent.
"""
import asyncio
import json
import logging
import re
import sys
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes,
)
import memory as mem
from tools import execute_tool, get_tools_description
import config as cfg

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(cfg.LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Init ─────────────────────────────────────────────────────────────────────

mem.init_db()
cfg.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_API_BASE)

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a personal AI assistant running on a server.
You belong to one person — your owner — and you chat with them directly.

== YOUR PERSONALITY ==
You are smart, direct, and slightly informal. You hold real conversations:
- Answer questions, explain concepts, share opinions when asked.
- Keep replies concise unless the topic needs depth.
- Don't force every message into a task. If someone asks "what do you think about X?"
  just answer like a knowledgeable friend would — no tool calls needed.
- You're not a yes-machine, but you're also not cold. Friendly without being cringe.
- Light humor is fine where natural. Don't overdo it.
- NEVER reply with empty output. Always say something, even if just a short answer.

== WHEN TO USE TOOLS ==
Read the intent carefully before reaching for a tool:
- Questions, opinions, explanations, advice → just reply in plain text.
- "run X", "create a script", "set up Y", "check my CPU", "deploy Z" → use tools.
- If genuinely unsure, answer conversationally first and offer to execute if wanted.

== YOUR SERVER CAPABILITIES ==
When the owner needs something done, you have full control:
run shell commands, manage files, start background services, call APIs,
store encrypted credentials, remember facts across conversations.

== AVAILABLE TOOLS ==
{tools}

== HOW TO USE A TOOL ==
When you need a tool, output EXACTLY this (nothing else on that turn):

<tool_call>
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
</tool_call>

After you see the <tool_result>, decide whether to call another tool or give a final reply.
When done, respond in plain text — no <tool_call> block.

== TASK RULES ==
- ANY credential shared (API key, token, password, secret — no matter how sensitive or critical) → store_cred it IMMEDIATELY. No asking, no hesitation, no exceptions. Just store it.
- Personal fact shared → remember it.
- Scripts/programs: write_file first, then run_command to execute.
- Persistent services: write the script then spawn_service.
- Show command output only when it adds value.
"""

PLAN_PROMPT = """\
You are a thoughtful planner. The user wants a plan — not execution.
Break down the request into clear numbered steps. Be specific about what each step does.
Do NOT use any tools. Do NOT run anything. Just produce a structured plan.
State any assumptions upfront.
"""

AGENT_PROMPT = """\
You are in AGENT MODE. Execute the user's request autonomously using all available tools.
Do not ask for confirmation — make decisions and take action.
Report what you did when complete.

== AVAILABLE TOOLS ==
{tools}

== HOW TO USE A TOOL ==
Output EXACTLY this (nothing else on that turn):

<tool_call>
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
</tool_call>

Chain as many tool calls as needed. Give a concise summary when done.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_tool_call(reply: str) -> tuple[str, dict] | None:
    """Extract (name, args) from a tool call in the reply, or return None.
    Handles proper <tool_call> tags. Also catches bare JSON only when the
    entire reply is nothing but a tool call for a real registered tool.
    """
    from tools import TOOL_REGISTRY

    # Method 1: proper <tool_call> tags (always trust these)
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", reply, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            return parsed.get("name", ""), parsed.get("arguments", {})
        except json.JSONDecodeError:
            pass

    # Method 2: bare JSON — ONLY when the entire reply is a single JSON object
    # AND the name matches a real tool. This prevents catching JSON in normal text.
    stripped = reply.strip()
    bare = re.sub(r'^```(?:json)?\s*', '', stripped)
    bare = re.sub(r'\s*```$', '', bare).strip()
    if bare == stripped or bare == stripped.strip('`').strip():
        try:
            parsed = json.loads(bare)
            if (isinstance(parsed, dict)
                    and isinstance(parsed.get("name"), str)
                    and isinstance(parsed.get("arguments"), dict)
                    and parsed["name"] in TOOL_REGISTRY):
                logger.warning(f"Caught bare JSON tool call (no tags): {parsed['name']}")
                return parsed["name"], parsed["arguments"]
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def get_owner_id() -> int | None:
    val = mem.get_config("owner_telegram_id")
    return int(val) if val else None


def is_owner(user_id: int) -> bool:
    owner = get_owner_id()
    return True if owner is None else user_id == owner


def get_current_model() -> str:
    return mem.get_config("current_model", cfg.DEFAULT_MODEL)


# ── Core agent loop ───────────────────────────────────────────────────────────

async def run_agent(chat_id: int, user_message: str, model: str) -> str:
    """Run the full agent loop: LLM → tool calls → final reply."""
    mem.save_message(chat_id, "user", user_message)
    history = mem.get_history(chat_id, cfg.MAX_HISTORY_MESSAGES)

    system = SYSTEM_PROMPT.format(tools=get_tools_description())
    messages = [{"role": "system", "content": system}] + history

    for iteration in range(cfg.MAX_TOOL_ITERATIONS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"❌ LLM error: {e}"

        reply = response.choices[0].message.content.strip()

        # Detect tool call (handles tagged AND bare JSON forms)
        tool_call = _parse_tool_call(reply)
        if tool_call:
            name, args = tool_call
            if not name:
                logger.warning("Tool call parsed but name was empty; skipping")
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": "Tool name was missing. Retry with a valid <tool_call>."})
                continue

            logger.info(f"Tool call [{iteration+1}]: {name}({args})")
            result = execute_tool(name, args)

            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": f"<tool_result>\n{result}\n</tool_result>\n"
                           "Continue based on this result. If done, give the final reply.",
            })
        else:
            # Final reply
            mem.save_message(chat_id, "assistant", reply)
            return reply

    return "⚠️ Reached max tool iterations. Something might need manual attention."


# ── Telegram command handlers ─────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if get_owner_id() is None:
        mem.set_config("owner_telegram_id", str(user_id))
        await update.message.reply_text(
            f"👋 Welcome! Owner locked to your ID: `{user_id}`\n\n"
            "I'm your personal AI agent — I can run code, manage files, call APIs, "
            "remember things, and run background services on this server.\n\n"
            "Type anything or use /help.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text("🟢 Agent is running. Type /help for commands.")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        return
    await update.message.reply_text(
        "🤖 *SynthClaw\\-CoAgent — Commands*\n\n"
        "/start — Register as owner\n"
        "/help — This message\n"
        "/clear — Wipe conversation history\n"
        "/model \\[name\\] — Show or switch model\n"
        "/models — List all available models\n"
        "/status — Show running services\n"
        "/creds — List stored credentials\n"
        "/memory — Show remembered facts\n"
        "/run \\<cmd\\> — Run shell command directly\n"
        "/plan \\<task\\> — Break task into steps \\(no execution\\)\n"
        "/agent \\<task\\> — Autonomous execution mode\n"
        "/ping — Check if alive\n\n"
        "*Just chat normally:*\n"
        "• _What's the best way to set up a cron job?_\n"
        "• _Create a Python price tracker and run it hourly_\n"
        "• _Store my AWS key as aws\\_key_\n"
        "• _Remember my timezone is UTC\\+5_",
        parse_mode="MarkdownV2",
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        return
    mem.clear_history(update.effective_chat.id)
    await update.message.reply_text("🧹 History cleared.")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        return
    args = context.args
    if not args:
        current = get_current_model()
        await update.message.reply_text(
            f"Current model: `{current}`\n\nUse /models to see all options.",
            parse_mode="Markdown",
        )
        return
    new_model = args[0]
    if new_model not in cfg.AVAILABLE_MODELS:
        await update.message.reply_text("❌ Unknown model. Use /models.")
        return
    mem.set_config("current_model", new_model)
    await update.message.reply_text(f"✅ Switched to `{new_model}`", parse_mode="Markdown")


async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        return
    current = get_current_model()
    lines = [
        ("▶️ " if m == current else "   ") + f"`{m}`"
        for m in cfg.AVAILABLE_MODELS
    ]
    await update.message.reply_text(
        "*Available Models:*\n" + "\n".join(lines), parse_mode="Markdown"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        return
    result = json.loads(execute_tool("service_status", {}))
    output = result.get("output", "No running services.")
    await update.message.reply_text(f"```\n{output[:3500]}\n```", parse_mode="Markdown")


async def cmd_creds(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        return
    creds = mem.list_credentials()
    if not creds:
        await update.message.reply_text("No credentials stored yet.")
        return
    lines = ["*Stored Credentials* (values hidden):"] + [
        f"• `{c['name']}` — {c['description'] or '—'}" for c in creds
    ]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_memory_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        return
    all_mem = mem.get_all_memory()
    if not all_mem:
        await update.message.reply_text("No memories yet.")
        return
    lines = ["*Stored Memories:*"] + [f"• `{k}`: {v}" for k, v in all_mem.items()]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        return
    command = " ".join(context.args)
    if not command:
        await update.message.reply_text("Usage: /run <shell command>")
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    result = json.loads(execute_tool("run_command", {"command": command, "timeout": 30}))
    output = result.get("stdout", "") + result.get("stderr", "") or result.get("error", "No output")
    await update.message.reply_text(f"```\n{output[:3500]}\n```", parse_mode="Markdown")


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🟢 Alive!")


async def cmd_plan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Break down a task into steps without executing anything."""
    if not is_owner(update.effective_user.id):
        return
    task = " ".join(context.args)
    if not task:
        await update.message.reply_text("Usage: /plan <describe what you want to do>")
        return
    chat_id = update.effective_chat.id
    model = get_current_model()
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PLAN_PROMPT},
                {"role": "user", "content": task},
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        reply = response.choices[0].message.content.strip()
        if not reply:
            reply = "(model returned empty plan)"
        await update.message.reply_text(f"📋 *Plan:* {task}\n\n{reply}", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"cmd_plan error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ {e}")


async def cmd_agent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force autonomous execution mode — acts without asking for confirmation."""
    if not is_owner(update.effective_user.id):
        return
    task = " ".join(context.args)
    if not task:
        await update.message.reply_text("Usage: /agent <task to execute autonomously>")
        return
    chat_id = update.effective_chat.id
    model = get_current_model()
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    await update.message.reply_text(f"🤖 Agent mode — working on: _{task}_", parse_mode="Markdown")
    try:
        system = AGENT_PROMPT.format(tools=get_tools_description())
        messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
        for iteration in range(cfg.MAX_TOOL_ITERATIONS):
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0.2, max_tokens=2048
            )
            reply = response.choices[0].message.content.strip()
            tc = _parse_tool_call(reply)
            if tc:
                name, args = tc
                if not name:
                    break
                logger.info(f"[/agent] Tool [{iteration+1}]: {name}({args})")
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                result = execute_tool(name, args)
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"<tool_result>\n{result}\n</tool_result>\nContinue."})
            else:
                if not reply:
                    reply = "Done."
                for chunk in [reply[i:i+4000] for i in range(0, len(reply), 4000)]:
                    await update.message.reply_text(chunk)
                return
        await update.message.reply_text("⚠️ Max tool iterations reached.")
    except Exception as e:
        logger.error(f"cmd_agent error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ {e}")


# ── Message handler ───────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_owner(user_id):
        await update.message.reply_text("⛔ Unauthorized.")
        return

    chat_id = update.effective_chat.id
    model = get_current_model()

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        reply = await run_agent(chat_id, update.message.text, model)
        if not reply or not reply.strip():
            reply = "_(got an empty response — try rephrasing or /clear to reset history)_"
        for chunk in [reply[i:i+4000] for i in range(0, len(reply), 4000)]:
            await update.message.reply_text(chunk)
    except Exception as e:
        logger.error(f"handle_message error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not cfg.TELEGRAM_TOKEN:
        print("ERROR: TELEGRAM_TOKEN not set. Run `python setup_cli.py` first.")
        sys.exit(1)
    if not cfg.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Run `python setup_cli.py` first.")
        sys.exit(1)

    app = Application.builder().token(cfg.TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("help",    cmd_help))
    app.add_handler(CommandHandler("clear",   cmd_clear))
    app.add_handler(CommandHandler("model",   cmd_model))
    app.add_handler(CommandHandler("models",  cmd_models))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("creds",   cmd_creds))
    app.add_handler(CommandHandler("memory",  cmd_memory_cmd))
    app.add_handler(CommandHandler("run",     cmd_run))
    app.add_handler(CommandHandler("ping",    cmd_ping))
    app.add_handler(CommandHandler("plan",    cmd_plan))
    app.add_handler(CommandHandler("agent",   cmd_agent))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🤖 SynthClaw Telegram agent starting (polling)…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
