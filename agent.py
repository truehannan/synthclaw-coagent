"""
SynthClaw-CoAgent — Telegram Bot Interface
Main entry point for the Telegram agent.
"""
import asyncio
import json
import logging
import re
import sys
import time
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes,
)
import memory as mem
from tools import execute_tool, get_tools_description, TOOL_REGISTRY
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

# Client is created lazily via _get_client() so it can be reconfigured in-bot


def _get_client() -> OpenAI:
    """Return an OpenAI client, reading key/base from DB config if set."""
    api_key = mem.get_config("llm_api_key") or cfg.OPENAI_API_KEY
    api_base = mem.get_config("llm_api_base") or cfg.OPENAI_API_BASE
    return OpenAI(api_key=api_key, base_url=api_base)


def _llm_call(**kwargs):
    """Wrap every LLM call with exponential-backoff retry on rate-limit errors.

    Retries up to 5 times: 5 → 15 → 30 → 60 → 120 seconds.
    Non-rate-limit errors are re-raised immediately.
    """
    delays = [5, 15, 30, 60, 120]
    for attempt, delay in enumerate(delays, 1):
        try:
            return _get_client().chat.completions.create(**kwargs)
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ("rate limit", "ratelimit", "429", "too many requests", "throttle")):
                logger.warning(
                    f"⏳ Rate limit hit — attempt {attempt}/{len(delays)}, "
                    f"waiting {delay}s before retry…"
                )
                time.sleep(delay)
            else:
                raise
    return _get_client().chat.completions.create(**kwargs)  # final attempt after all waits


def _is_configured() -> bool:
    """True if an LLM API key is available (env or DB)."""
    return bool(mem.get_config("llm_api_key") or cfg.OPENAI_API_KEY)

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


def _extract_json_objects(text: str) -> list[str]:
    """Extract all top-level JSON objects using bracket counting.
    Handles arbitrary nesting — the old flat regex broke on nested {} args.
    """
    results = []
    depth = 0
    start = -1
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    results.append(text[start:i + 1])
                    start = -1
        i += 1
    return results


def _parse_tool_call(reply: str) -> tuple[str, dict] | None:
    """Extract (name, args) from a reply, guarded by TOOL_REGISTRY.

    Priority order:
      1. <tool_call>...</tool_call>  — primary tagged format
      2. <tool_call>...{no close tag} — truncated output fallback
      3. Bare JSON outside fenced blocks — model forgot the tags
         (fenced blocks stripped first; bracket-counter handles any nesting)
    All paths require name in TOOL_REGISTRY to prevent false positives.
    """
    tool_names = set(TOOL_REGISTRY.keys())

    def _x(raw: str):
        try:
            p = json.loads(raw.strip())
            n = p.get("name", "")
            if n and n in tool_names:
                return n, p.get("arguments", {})
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    # 1. Closed <tool_call> tag
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", reply, re.DOTALL)
    if m:
        r = _x(m.group(1))
        if r: return r

    # 2. Unclosed <tool_call> tag (output truncated)
    m = re.search(r"<tool_call>\s*(\{.*)", reply, re.DOTALL)
    if m:
        r = _x(m.group(1))
        if r: return r

    # 3. Bare JSON — strip ALL fenced code blocks first to avoid false positives
    #    then use bracket-counter (handles nested braces, ${VAR}, nested objects)
    stripped = re.sub(r"```[\s\S]*?```", "", reply)
    for candidate in _extract_json_objects(stripped):
        r = _x(candidate)
        if r: return r

    return None


# ── Smart context (MD file system) ────────────────────────────────────────────

SUMMARIZE_THRESHOLD = 30   # total msgs before we summarize + prune
RECENT_WINDOW = 15         # keep this many recent messages verbatim
SUMMARIZE_BATCH = 15       # how many old messages to summarize at once

SUMMARY_SYSTEM = (
    "You are a summarizer. Condense the following conversation into a concise markdown "
    "summary (bullet points preferred). Capture: key facts about the user, decisions made, "
    "tasks completed, credentials/services mentioned, preferences learned, and important "
    "context. If a previous summary exists, merge new info into it. "
    "Output ONLY the updated markdown summary, nothing else."
)

PROFILE_EXTRACT_SYSTEM = (
    "Extract personal facts about the user from this conversation. "
    "Output a markdown list of facts (name, timezone, preferences, projects, tech stack, etc). "
    "Only include things the user explicitly stated about themselves. "
    "If nothing personal was shared, output exactly: NONE"
)


def _maybe_summarize(chat_id: int):
    """If message count exceeds threshold, summarize oldest batch into MD files and prune."""
    total = mem.count_messages(chat_id)
    if total <= SUMMARIZE_THRESHOLD:
        return

    oldest = mem.get_oldest_messages(chat_id, SUMMARIZE_BATCH)
    if len(oldest) < 5:
        return

    # Build conversation text
    convo_lines = []
    for msg in oldest:
        role = msg["role"].upper()
        convo_lines.append(f"{role}: {msg['content'][:500]}")
    convo_text = "\n".join(convo_lines)

    # ── 1. Update summary.md ──
    prev_summary = mem.get_md_summary(chat_id)
    summary_input = convo_text
    if prev_summary:
        summary_input = f"[Previous summary]:\n{prev_summary}\n\n[New messages]:\n{convo_text}"

    try:
        resp = _llm_call(
            model=cfg.DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": summary_input},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        summary = resp.choices[0].message.content.strip()
        if summary:
            mem.save_md_summary(chat_id, summary)
            mem.save_summary(chat_id, summary, total)
            logger.info(f"Updated summary.md for chat {chat_id}")
    except Exception as e:
        logger.error(f"Summary update failed: {e}")

    # ── 2. Extract profile facts ──
    try:
        resp = _llm_call(
            model=cfg.DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": PROFILE_EXTRACT_SYSTEM},
                {"role": "user", "content": convo_text},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        profile_facts = resp.choices[0].message.content.strip()
        if profile_facts and profile_facts != "NONE":
            for line in profile_facts.split("\n"):
                line = line.strip().lstrip("-•* ")
                if line:
                    mem.append_to_profile(chat_id, line)
            logger.info(f"Updated profile.md for chat {chat_id}")
    except Exception as e:
        logger.error(f"Profile extraction failed: {e}")

    # ── 3. Log key events to session file ──
    for msg in oldest:
        if msg["role"] == "user" and len(msg["content"]) > 10:
            short = msg["content"][:120].replace("\n", " ")
            mem.append_to_session(chat_id, f"User: {short}")

    # ── 4. Prune the old messages from DB ──
    msg_ids = [m["id"] for m in oldest]
    mem.delete_messages_by_ids(msg_ids)
    logger.info(f"Summarized + pruned {len(oldest)} messages for chat {chat_id}")


def _build_context(chat_id: int) -> list[dict]:
    """Build the full message list with MD context, memories, cred names."""
    all_memories = mem.get_all_memory()
    cred_list = mem.list_credentials()

    extra_context = []

    # MD file long-term context
    md_context = mem.get_full_context_md(chat_id)
    if md_context:
        extra_context.append(f"\n== LONG-TERM CONTEXT ==\n{md_context}")

    if all_memories:
        facts = "; ".join(f"{k}: {v}" for k, v in all_memories.items())
        extra_context.append(f"\n== KNOWN FACTS ABOUT OWNER ==\n{facts}")
    if cred_list:
        names = ", ".join(c["name"] for c in cred_list)
        extra_context.append(f"\n== STORED CREDENTIALS (available via get_cred) ==\n{names}")

    system = SYSTEM_PROMPT.format(tools=get_tools_description())
    if extra_context:
        system += "\n" + "\n".join(extra_context)

    messages = [{"role": "system", "content": system}]

    history = mem.get_history(chat_id, RECENT_WINDOW)
    messages.extend(history)

    return messages


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

    # Smart context: summarize old messages if needed, then build enriched context
    _maybe_summarize(chat_id)
    messages = _build_context(chat_id)

    for iteration in range(cfg.MAX_TOOL_ITERATIONS):
        try:
            response = _llm_call(
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
            # Guardrail: never surface raw <tool_call> payloads to user if parse failed
            if "<tool_call>" in reply or "</tool_call>" in reply:
                logger.warning("Tagged tool_call detected but parse failed; asking model to retry valid tool JSON")
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your last reply looked like a tool call but was invalid/truncated and could not be parsed. "
                        "Retry with ONLY a valid <tool_call> JSON using a known tool name from the tool list. "
                        "Do not include extra prose."
                    ),
                })
                continue

            # Final reply
            mem.save_message(chat_id, "assistant", reply)
            return reply

    return (
        "⚠️ *Task stopped — too many steps reached (40 tool calls)*\n\n"
        "The agent ran 40 steps on this task without finishing. "
        "This usually means it got stuck in a loop or the task is too complex to complete in one go.\n\n"
        "What you can do:\n"
        "• /clear — Reset the conversation and try a simpler or shorter instruction\n"
        "• Break your task into smaller parts and send them one at a time\n"
        "• If something went partially wrong, check /status and ask me to inspect the workspace"
    )


# ── Telegram command handlers ─────────────────────────────────────────────────

ONBOARDING_STEP_KEY = "onboarding_step"  # values: "api_key", "api_base", "done"


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    # First time — lock owner
    if get_owner_id() is None:
        mem.set_config("owner_telegram_id", str(user_id))
        # Decide if we need onboarding (no API key configured)
        if not _is_configured():
            mem.set_config(ONBOARDING_STEP_KEY, "api_key")
            await update.message.reply_text(
                f"👋 Welcome! I'm *SynthClaw-CoAgent* — your personal AI agent.\n\n"
                f"🔒 You're now the owner \(ID: `{user_id}`\)\.\n\n"
                "Let's get you set up in 2 quick steps\.\n\n"
                "*Step 1 of 2 — LLM API Key*\n"
                "Send me your API key now\. Example:\n"
                "`sk-do-xxxx...` \(DigitalOcean Gradient AI\)\n"
                "`sk-xxxx...` \(OpenAI\)\n\n"
                "_Your key is stored encrypted on this server and never sent anywhere else\._",
                parse_mode="MarkdownV2",
            )
        else:
            mem.set_config(ONBOARDING_STEP_KEY, "done")
            await update.message.reply_text(
                f"👋 Welcome back\! Owner set to ID `{user_id}`\.\n\n"
                "✅ LLM API key already configured \(from environment\)\.\n"
                "I'm ready to go — type anything or use /help\.",
                parse_mode="MarkdownV2",
            )
        return

    # Already has an owner — send to owner only
    if not is_owner(user_id):
        await update.message.reply_text("⛔ This agent already has an owner.")
        return

    step = mem.get_config(ONBOARDING_STEP_KEY, "done")
    if step == "done":
        await update.message.reply_text(
            "🟢 Agent is running\. Use /setup to view config or /help for commands\.",
            parse_mode="MarkdownV2",
        )
    else:
        # Resume interrupted onboarding
        await update.message.reply_text(
            "⏸ Onboarding isn't finished yet\. Use /setup to check what's missing\.",
            parse_mode="MarkdownV2",
        )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        return
    await update.message.reply_text(
        "🤖 *SynthClaw\\-CoAgent — Commands*\n\n"
        "⚙️ *Setup*\n"
        "/setup — Show configuration status\n"
        "/setkey \\<key\\> — Set LLM API key\n"
        "/setbase \\<url\\> — Set LLM API base URL\n\n"
        "💬 *General*\n"
        "/start — Register as owner / resume onboarding\n"
        "/help — This message\n"
        "/clear — Wipe conversation history\n"
        "/model \\[name\\] — Show or switch model\n"
        "/models — List all available models\n"
        "/ping — Check if alive\n\n"
        "🖥 *Server*\n"
        "/status — Show running services\n"
        "/run \\<cmd\\> — Run shell command directly\n\n"
        "🧠 *Memory*\n"
        "/creds — List stored credentials\n"
        "/storekey \\<NAME\\> \\<VALUE\\> — Store a key directly \\(bypasses AI\\)\n"
        "/memory — Show remembered facts\n\n"
        "🤖 *Agent*\n"
        "/plan \\<task\\> — Break task into steps \\(no execution\\)\n"
        "/agent \\<task\\> — Autonomous execution mode\n\n"
        "*Just chat normally:*\n"
        "• _What's the best way to set up a cron job?_\n"
        "• _Create a Python price tracker and run it hourly_\n"
        "• _Store my AWS key as aws\\_key_\n"
        "• _Remember my timezone is UTC\\+5_",
        parse_mode="MarkdownV2",
    )


async def cmd_setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current configuration status and what still needs to be done."""
    if not is_owner(update.effective_user.id):
        return

    api_key = mem.get_config("llm_api_key") or cfg.OPENAI_API_KEY
    api_base = mem.get_config("llm_api_base") or cfg.OPENAI_API_BASE
    model = get_current_model()

    key_source = "🔐 DB" if mem.get_config("llm_api_key") else ("✅ env" if cfg.OPENAI_API_KEY else "❌ missing")
    base_source = "🔐 DB" if mem.get_config("llm_api_base") else "✅ env/default"
    key_preview = f"`{api_key[:8]}...{api_key[-4:]}`" if api_key and len(api_key) > 12 else ("set" if api_key else "**not set**")

    lines = [
        "⚙️ *SynthClaw Configuration*\n",
        f"🔑 *API Key:* {key_source} — {key_preview}",
        f"🌐 *API Base:* {base_source} — `{api_base}`",
        f"🧠 *Model:* `{model}`",
        "",
    ]

    if not api_key:
        lines.append("❌ *Action needed:* Send `/setkey <your-api-key>` to configure the LLM.")
    else:
        lines.append("✅ Ready to chat. Type anything or use /help.")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_setkey(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set the LLM API key (stored encrypted in DB)."""
    if not is_owner(update.effective_user.id):
        return

    key = " ".join(context.args).strip()
    if not key:
        await update.message.reply_text(
            "Usage: `/setkey <api-key>`\n\nExample: `/setkey sk-do-xxxxxxxxxxxx`",
            parse_mode="Markdown",
        )
        return

    mem.set_config("llm_api_key", key)
    preview = f"`{key[:8]}...{key[-4:]}`" if len(key) > 12 else "`set`"

    # Advance onboarding step if still in progress
    step = mem.get_config(ONBOARDING_STEP_KEY, "done")
    if step == "api_key":
        mem.set_config(ONBOARDING_STEP_KEY, "done")
        await update.message.reply_text(
            f"✅ API key saved: {preview}\n\n"
            "🎉 *Setup complete!* You're ready to go.\n\n"
            "Try chatting with me, or use /help to see all commands.\n\n"
            "_Optional: use `/setbase <url>` if you're using a non-default API endpoint._",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            f"✅ API key updated: {preview}", parse_mode="Markdown"
        )


async def cmd_setbase(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set the LLM API base URL."""
    if not is_owner(update.effective_user.id):
        return

    base = " ".join(context.args).strip()
    if not base:
        current = mem.get_config("llm_api_base") or cfg.OPENAI_API_BASE
        await update.message.reply_text(
            f"Current API base: `{current}`\n\n"
            "Usage: `/setbase <url>`\n\nExamples:\n"
            "• `https://inference.do-ai.run/v1` (DigitalOcean — default)\n"
            "• `https://api.openai.com/v1` (OpenAI)\n"
            "• `http://localhost:11434/v1` (Ollama)",
            parse_mode="Markdown",
        )
        return

    mem.set_config("llm_api_base", base)
    await update.message.reply_text(f"✅ API base updated: `{base}`", parse_mode="Markdown")


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


async def cmd_storekey(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Directly store a credential — LLM never sees the value."""
    if not is_owner(update.effective_user.id):
        return
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Usage: `/storekey NAME VALUE`\n"
            "Example: `/storekey OPENAI_KEY sk-abc123`\n\n"
            "Optional description after `|`:\n"
            "`/storekey OPENAI_KEY sk-abc123 | my openai key`",
            parse_mode="Markdown",
        )
        return
    name = context.args[0].upper()
    rest = " ".join(context.args[1:])
    if "|" in rest:
        value, _, description = rest.partition("|")
        value = value.strip()
        description = description.strip()
    else:
        value = rest.strip()
        description = ""
    mem.store_credential(name, value, description)
    await update.message.reply_text(
        f"✅ Stored `{name}`" + (f" — _{description}_" if description else "") + "\n"
        "_(Value never sent to AI — stored directly on server)_",
        parse_mode="Markdown",
    )


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
        response = _llm_call(
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
            response = _llm_call(
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

    app = Application.builder().token(cfg.TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("help",    cmd_help))
    app.add_handler(CommandHandler("setup",   cmd_setup))
    app.add_handler(CommandHandler("setkey",  cmd_setkey))
    app.add_handler(CommandHandler("setbase", cmd_setbase))
    app.add_handler(CommandHandler("clear",   cmd_clear))
    app.add_handler(CommandHandler("model",   cmd_model))
    app.add_handler(CommandHandler("models",  cmd_models))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("creds",    cmd_creds))
    app.add_handler(CommandHandler("storekey", cmd_storekey))
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
