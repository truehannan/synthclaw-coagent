import asyncio
import datetime
import json
import logging
import re
import sys
import time
from pathlib import Path
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

client = OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_API_BASE)

# Active task tracking — allows /stop to interrupt a running agent loop
ACTIVE_TASKS: dict[int, bool] = {}  # chat_id -> is_running


def _llm_call(**kwargs):
    """Wrap every LLM call with exponential-backoff retry on rate-limit errors.

    Retries up to 5 times: 5 → 15 → 30 → 60 → 120 seconds.
    Non-rate-limit errors are re-raised immediately.
    """
    delays = [5, 15, 30, 60, 120]
    for attempt, delay in enumerate(delays, 1):
        try:
            return client.chat.completions.create(**kwargs)
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
    try:                                     # Fix #10: wrap final attempt — don't crash on last retry
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        logger.error(f"LLM final retry failed: {e}")
        raise


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
== THINK BEFORE YOU ACT — MANDATORY ==
Every response that uses a tool MUST begin with a <think> block:

<think>
User wants: [restate their FULL message — not just a keyword]
What I know: [current state, what's already done]
What to verify: [packages exist? versions valid? files present?]
Plan: [numbered steps, using batch tools where possible]
</think>

The <think> block is internal — the user will never see it.
After </think>, you may include a short status message (the user sees this),
then your <tool_call> block(s).

For plain-text replies (no tools), you do NOT need a <think> block.

== MANDATORY WORKFLOW ==
Before EVERY action, follow this sequence:

1. UNDERSTAND — Read the user's ENTIRE message. Do NOT react to a single keyword.
   Restate what they actually want inside <think>. If the message is a question
   or opinion request, just answer — no tools needed.

2. VERIFY — Before installing, downloading, or using anything:
   • Check the package/version EXISTS: pip index versions X, npm view X versions,
     apt-cache show X, or similar.
   • Check files/paths EXIST before editing (read_file or list_files first).
   • NEVER guess version numbers — look them up.
   • NEVER install something without first confirming it exists.

3. PLAN — In <think>, list the exact steps in order. Use batch tools:
   • read_files(paths) to read many files in ONE call
   • write_files(files) to write many files in ONE call
   • run_commands(commands) to run many commands in ONE call
   • Multiple <tool_call> blocks in one response — all execute at once
   Group reads together, then writes together. Minimize round-trips.

4. EXECUTE — Only after understanding, verifying, and planning.

== TOOL CALL FORMAT ==
<tool_call>
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
</tool_call>

Multiple tool calls in one response (all execute at once):
<tool_call>
{{"name": "read_file", "arguments": {{"path": "config.py"}}}}
</tool_call>
<tool_call>
{{"name": "read_file", "arguments": {{"path": "main.py"}}}}
</tool_call>

❌ No raw JSON without tags. ❌ No markdown fences.
✅ Always use <tool_call> tags — the only format that works.

== EXIT CODE RULES (CRITICAL) ==
- returncode 0 = SUCCESS. Proceed.
- returncode != 0 = FAILED. The command DID NOT WORK.
  → Do NOT proceed as if it succeeded.
  → Read the stderr. Diagnose the problem.
  → Fix the issue (install missing dep, fix typo, correct path) and retry.
  → If you cannot fix it after 2 attempts, STOP and tell the user what failed and why.
- NEVER ignore a failed command. NEVER assume it worked when returncode != 0.
- After ANY install command (pip/npm/apt), CHECK the result before using the package.

== VERIFICATION RULES (CRITICAL) ==
- pip install <pkg>: FIRST run `pip index versions <pkg>` or `pip install <pkg>==`
  to check it exists and see available versions.
- pip install <pkg>==X.Y.Z: FIRST verify that version exists.
- npm install <pkg>: FIRST run `npm view <pkg> versions --json` to check.
- apt install <pkg>: FIRST run `apt-cache show <pkg>` to verify.
- curl/wget a URL: FIRST check the URL responds (http_request with HEAD/GET).
- Editing a file: FIRST read it (or use read_files to batch-read).
- NEVER assume a package name, version, or URL is correct — verify it.

== FAILURE PROTOCOL ==
- If a step FAILS, diagnose → fix → retry. Do NOT skip to the next step.
- Example: if `pip install X` fails, read the error, fix it, retry before moving on.
- If 2 attempts both fail, STOP and report WHAT failed and WHY in plain terms.
- NEVER ask the user to fix it. NEVER say "you should", "you can", "run this yourself",
  "you'll need to", "manually do". You have full server access — handle it yourself.

== YOU NEVER ASK THE USER TO DO ANYTHING (ABSOLUTE RULE) ==
❌ NEVER say "you need to", "you should", "you'll have to", "please do", "run this command"
❌ NEVER give the user a command to run in their own terminal.
❌ NEVER suggest the user install, configure, or fix anything themselves.
❌ NEVER say "let me know when you've done X".
✅ If something is hard or failed: you try harder. If it truly can't be done: report why — without delegating.
✅ You are acting ON the server, not coaching the user to act on it.

== WHO YOU ARE ==
Personal AI assistant on a DigitalOcean droplet (Singapore).
You belong to one person — your owner — chatting via Telegram.

== DO NOT NARRATE — JUST ACT (CRITICAL) ==
❌ NEVER list steps you are about to take.
❌ NEVER say "Here's what I'll do", "Here's my plan", "Steps:", "I will:", "Remaining:", "Here's how", "This is how it works".
❌ NEVER recap what the user asked for.
❌ If they say "yes" or "do it" — execute IMMEDIATELY. No recap, no list, no explanation.
❌ NEVER output a numbered/bulleted list of upcoming actions.
✅ Just call tools. Results speak for themselves.
✅ ONE short status line max before tool calls: "Installing..." or "Creating file..."
✅ Final reply: just the outcome. "Done — service is running on port 8080." Not a summary of everything you did.

== PERSONALITY ==
Smart, direct, slightly informal. Hold real conversations:
- Questions, opinions, advice → plain text, no tools.
- "run X", "create a script", "deploy Z" → use tools immediately, no preamble.
- Concise. Friendly but not cringe.
- NEVER reply with empty output.

== CAPABILITIES ==
Full server control: shell commands, file management, background services,
HTTP APIs, encrypted credential storage, persistent memory.
Timeouts auto-scale: 30s normal commands, 180s installs, 300s builds.

Media: receive photos/voice/audio/video/docs/stickers (auto-saved to server).
Send back via send_media. Download via download_url. Browse via list_media.
Generate images via generate_image. Storage: /opt/agent/workspace/media/.

Search files with search_files (grep). Edit parts of files with patch_file.
Get system info with system_info. Check ports with check_port.

== AVAILABLE TOOLS ==
{tools}

== TASK RULES ==
- Credentials shared → store_cred IMMEDIATELY. No asking.
- Personal facts → remember.
- Scripts: write_file then run_command.
- Services: write script then spawn_service. Check port first with check_port.
- Show output only when it adds value.
- When approved to proceed: EXECUTE, do not re-list the steps.
- pip/pip3 are auto-routed to the venv. Just run `pip install X` normally.
- python/python3 are auto-routed to venv Python. Just use them normally.

== ABSOLUTE RULE: NEVER DELEGATE TO THE USER ==
❌ You must NEVER tell the user to run a command, install something, edit a file, or take any action.
❌ You must NEVER say "you need to", "you should", "you'll need to", "try running", "please run".
❌ If you hit a problem: YOU solve it. If you truly cannot: report the error — not a to-do list for the user.
✅ The user only asks. You only do.
"""

PLAN_PROMPT = """\
You are a thoughtful planner. The user wants a plan — not execution.
Break down the request into clear numbered steps. Be specific about what each step does.
Do NOT use any tools. Do NOT run anything. Just produce a structured plan.
State any assumptions upfront.
"""

AGENT_PROMPT = """\
You are in AGENT MODE. Execute the user's request autonomously.
Do not ask for confirmation — make decisions and take action.

== THINK FIRST (internal only) ==
Every response with tool calls MUST start with <think>:
<think>
User wants: [full restatement]
Verify: [what to check first]
Plan: [numbered steps]
</think>
The <think> block is NEVER shown to the user. Keep all planning inside it.

== NO NARRATION — JUST ACT ==
❌ NEVER output a list of steps to the user.
❌ NEVER say "Here's what I'll do", "I'll now", "Step 1:", "Remaining:", "Here's how".
❌ NEVER explain your approach or recap the plan outside <think>.
✅ One short status line before tool calls is fine: "Installing..." "Creating service..."
✅ When done: single-line confirmation only. Not a summary of all actions taken.

== EXECUTION RULES ==
- VERIFY before installing (pip index versions, npm view, apt-cache show).
- NEVER guess versions — look them up.
- Batch reads together, batch writes together. Minimize round-trips.
- Use read_files/write_files/run_commands for bulk operations.
- Multiple <tool_call> blocks in one response execute at once.

== EXIT CODES ==
- returncode 0 = success. Proceed.
- returncode != 0 = FAILED. Do NOT proceed. Read stderr, diagnose, fix, retry.
- After any install, VERIFY it worked (check returncode, pip show, which, etc).
- If 2 attempts fail, STOP and report the error. Do not loop endlessly.

== FAILURE PROTOCOL ==
- If a step fails, diagnose and fix BEFORE moving to the next step.
- NEVER skip a failed step. NEVER assume it worked.
- If 2 attempts fail, STOP and report the error in plain terms.
- NEVER ask the user to fix it or take any action themselves.

== ABSOLUTE RULE: NEVER DELEGATE TO THE USER ==
❌ NEVER tell the user to run a command, install a package, edit a file, or do anything.
❌ NEVER say "you need to", "you should", "try running", "please do", "you'll have to".
❌ You have full server access. You handle everything. The user only requests.
✅ On failure: report what failed and why. That is all.

== TOOL FORMAT ==
<tool_call>
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
</tool_call>

== AVAILABLE TOOLS ==
{tools}

Chain tool calls as needed. Final reply: result only, no narration.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json_objects(text: str) -> list[str]:
    """Extract all top-level JSON objects from text using bracket counting.
    Handles arbitrary nesting depth — the old regex broke on nested {}.
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
                i += 2   # skip escaped char
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


def _parse_tool_calls(reply: str) -> list[tuple[str, dict]]:
    """Extract ALL (name, args) tool calls from a reply.

    Returns a list of (name, args) tuples — supports multiple tool calls
    in a single LLM response for batched execution.

    Priority order:
      1. <tool_call>...</tool_call> — ALL tagged tool calls
      2. <tool_call>...{no close tag} — truncated output fallback (single)
      3. Bare JSON outside fenced blocks — model forgot the tags
    All paths require name in TOOL_REGISTRY to prevent false positives.
    """
    tool_names = set(TOOL_REGISTRY.keys())
    results = []

    def _x(raw: str):
        try:
            p = json.loads(raw.strip())
            n = p.get("name", "")
            if n and n in tool_names:
                return n, p.get("arguments", {})
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    # 1. All closed <tool_call> tags
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", reply, re.DOTALL):
        r = _x(m.group(1))
        if r:
            results.append(r)
    if results:
        return results

    # 2. Unclosed <tool_call> tag (output truncated)
    m = re.search(r"<tool_call>\s*(\{.*)", reply, re.DOTALL)
    if m:
        r = _x(m.group(1))
        if r:
            return [r]

    # 3. Bare JSON — strip fenced code blocks, use bracket-counter
    stripped = re.sub(r"```[\s\S]*?```", "", reply)
    for candidate in _extract_json_objects(stripped):
        r = _x(candidate)
        if r:
            results.append(r)

    return results


def _strip_think_block(text: str) -> str:
    """Remove <think>...</think> blocks from text. These are internal reasoning."""
    if not text:                             # Fix #18: guard None/empty input
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_think_block(reply: str) -> str:
    """Extract the content of the <think> block for logging."""
    if not reply:                            # Fix #18: guard None/empty
        return ""
    m = re.search(r"<think>(.*?)</think>", reply, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_pre_tool_text(reply: str) -> str:
    """Extract conversational text before the first <tool_call> tag.
    Strips out <think> blocks — those are internal, not for the user.
    """
    idx = reply.find("<tool_call>")
    if idx <= 0:
        return ""
    text = reply[:idx].strip()
    text = _strip_think_block(text)
    text = text.rstrip("`\n ")
    return text if len(text) > 3 else ""


# ── Smart context (MD file system) ────────────────────────────────────────────

SUMMARIZE_THRESHOLD = 20   # total msgs before we summarize + prune
RECENT_WINDOW = 30         # keep this many recent messages verbatim
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
            # Also save in DB for backward compat
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
            # Log meaningful user messages
            short = msg["content"][:120].replace("\n", " ")
            mem.append_to_session(chat_id, f"User: {short}")

    # ── 4. Prune the old messages from DB ──
    msg_ids = [m["id"] for m in oldest]
    mem.delete_messages_by_ids(msg_ids)
    logger.info(f"Summarized + pruned {len(oldest)} messages for chat {chat_id}")


def _build_context(chat_id: int) -> list[dict]:
    """Build the full message list with MD context, memories, cred names."""
    # 1. Get stored memories and credential names
    all_memories = mem.get_all_memory()
    cred_list = mem.list_credentials()

    # 2. Build enriched system prompt
    extra_context = []

    # MD file long-term context
    md_context = mem.get_full_context_md(chat_id)
    if md_context:
        extra_context.append(f"\n== LONG-TERM CONTEXT ==\n{md_context}")
    else:
        # No MD context yet — inject basic session info so agent knows convo is ongoing
        msg_count = mem.count_messages(chat_id)
        if msg_count > 0:
            extra_context.append(
                f"\n== SESSION INFO ==\nActive conversation. "
                f"Messages logged so far: {msg_count}. "
                "Long-term memory context not yet built (accumulates after first summarization)."
            )

    # Key-value memories
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

    # Recent messages (the actual conversation window)
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

async def run_agent(chat_id: int, user_message: str, model: str, send_fn=None) -> tuple[str, list[dict]]:
    """Core agent loop. Returns (text_reply, list_of_media_to_send).

    send_fn: optional async callable(str) to send intermediate progress messages.
    """
    media_to_send: list[dict] = []
    try:                                     # Fix #16: guard memory write failure
        mem.save_message(chat_id, "user", user_message)
    except Exception as me:
        logger.error(f"mem.save_message (user) failed: {me}")

    # Mark task as active (can be stopped via /stop)
    ACTIVE_TASKS[chat_id] = True

    # Smart context: summarize old messages if needed, then build enriched context
    try:                                     # Fix #16: guard summarize/context build
        _maybe_summarize(chat_id)
        messages = _build_context(chat_id)
    except Exception as me:
        logger.error(f"Context build failed: {me}")
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(tools=get_tools_description())}]

    for iteration in range(cfg.MAX_TOOL_ITERATIONS):
        # Check if user sent /stop
        if not ACTIVE_TASKS.get(chat_id, True):
            logger.info(f"Task stopped by user at iteration {iteration+1}")
            mem.save_message(chat_id, "assistant", "🛑 Task stopped by user.")
            return "🛑 Task stopped.", media_to_send

        try:
            response = _llm_call(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
            )
        except Exception as e:
            logger.error(f"LLM error: {e}")
            ACTIVE_TASKS.pop(chat_id, None)
            return f"❌ LLM error: {e}", media_to_send

        # Track token usage per model
        try:
            usage = response.usage
            if usage:
                mem.record_model_usage(model, usage.prompt_tokens, usage.completion_tokens)
        except Exception as ue:
            logger.debug(f"Usage tracking error: {ue}")

        reply = (response.choices[0].message.content or "").strip()  # Fix #14: guard None content

        # Log think block if present (internal reasoning — never sent to user)
        think_content = _extract_think_block(reply)
        if think_content:
            logger.info(f"🧠 Think [{iteration+1}]: {think_content[:300]}")

        logger.info(f"LLM reply [{iteration+1}]: {_strip_think_block(reply)[:300]!r}")

        # Detect tool calls (handles tagged AND bare JSON; supports multiple)
        tool_calls = _parse_tool_calls(reply)

        # Anti-loop detection: if the model sends the exact same tool calls twice in a row, break
        if tool_calls and iteration > 0:
            current_sig = str([(tc[0], json.dumps(tc[1], sort_keys=True)) for tc in tool_calls])
            if hasattr(run_agent, '_last_tool_sig') and run_agent._last_tool_sig.get(chat_id) == current_sig:
                logger.warning(f"Anti-loop: identical tool calls detected at iteration {iteration+1}, breaking loop")
                clean_reply = _strip_think_block(reply)
                error_msg = (
                    "⚠️ I detected I was about to repeat the same action. "
                    "Something isn't working as expected. Here's where I got stuck:\n\n"
                    + (clean_reply[:500] if clean_reply else "(no text in response)")
                )
                mem.save_message(chat_id, "assistant", error_msg)
                ACTIVE_TASKS.pop(chat_id, None)
                return error_msg, media_to_send
            if not hasattr(run_agent, '_last_tool_sig'):
                run_agent._last_tool_sig = {}
            run_agent._last_tool_sig[chat_id] = current_sig
        elif not tool_calls and hasattr(run_agent, '_last_tool_sig'):
            run_agent._last_tool_sig.pop(chat_id, None)

        logger.info(
            f"parse_tool_calls -> {len(tool_calls)} tool(s): "
            + (", ".join(tc[0] for tc in tool_calls) if tool_calls else "None (final reply)")
        )
        if tool_calls:
            # Send intermediate progress text before executing tools
            pre_text = _extract_pre_tool_text(reply)
            if pre_text and send_fn:
                try:
                    await send_fn(pre_text)
                except Exception as e:
                    logger.warning(f"Failed to send intermediate message: {e}")

            # Execute ALL tool calls — parallel for independent tools, sequential for ask_user
            combined_results = []

            # Stop check before firing any tools
            if not ACTIVE_TASKS.get(chat_id, True):
                ACTIVE_TASKS.pop(chat_id, None)
                mem.save_message(chat_id, "assistant", "🛑 Task stopped by user.")
                return "🛑 Task stopped.", media_to_send

            regular_calls = [(n, a) for n, a in tool_calls if n and n != "ask_user"]
            ask_calls     = [(n, a) for n, a in tool_calls if n == "ask_user"]

            if regular_calls:
                _loop = asyncio.get_running_loop()

                async def _run_tool_parallel(idx, tname, targs):
                    logger.info(f"Tool call [{iteration+1}][{idx}]: {tname}({targs})")
                    return tname, await _loop.run_in_executor(None, execute_tool, tname, targs)

                tool_results = await asyncio.gather(
                    *[_run_tool_parallel(i + 1, n, a) for i, (n, a) in enumerate(regular_calls)]
                )
                for tname, result in tool_results:
                    if tname in ("send_media", "generate_image"):
                        try:
                            result_data = json.loads(result)
                            if result_data.get("queued"):
                                media_to_send.append(result_data)
                        except json.JSONDecodeError as jde:
                            logger.warning(f"JSON decode error capturing media from {tname}: {jde} | raw: {result[:200]}")
                    combined_results.append(f"[{tname}]: {result}")

            # ask_user calls handled sequentially (they pause the loop)
            for name, args in ask_calls:
                logger.info(f"Tool call [{iteration+1}]: {name}({args})")
                result = execute_tool(name, args)
                try:
                    result_data = json.loads(result)
                    if result_data.get("ask_user") and result_data.get("question"):
                        question = result_data["question"]
                        mem.save_message(chat_id, "assistant", question)
                        ACTIVE_TASKS.pop(chat_id, None)
                        return question, media_to_send
                except json.JSONDecodeError as jde:   # Fix #5
                    logger.warning(f"JSON decode error in ask_user: {jde} | raw: {result[:200]}")
                combined_results.append(f"[{name}]: {result}")

            messages.append({"role": "assistant", "content": reply})
            if len(combined_results) == 1:
                result_text = combined_results[0]
            else:
                result_text = "\n\n".join(
                    f"[{i+1}/{len(combined_results)}] {r}"
                    for i, r in enumerate(combined_results)
                )
            if not result_text:              # Fix #11: guard empty result_text
                result_text = "(tool returned no output)"
            # Compress tool results if they're too large (save context tokens)
            if len(result_text) > 6000:
                result_text = result_text[:5500] + "\n\n... [output truncated to save context] ..."
            messages.append({
                "role": "user",
                "content": f"<tool_result>\n{result_text}\n</tool_result>\n"
                           "CRITICAL: returncode 0=success. returncode != 0 means FAILED — stop and diagnose, "
                           "do NOT continue to next step. If \"error\" key is present, fix it before continuing.\n"
                           "Continue based on these results. If everything succeeded, give the final reply.",
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

            # Final reply — strip any <think> block before sending
            clean_reply = _strip_think_block(reply)
            if not clean_reply:              # Fix #11/#14: guard empty final reply
                clean_reply = "Done."
            try:                             # Fix #16: guard memory write failure
                mem.save_message(chat_id, "assistant", clean_reply)
            except Exception as me:
                logger.error(f"mem.save_message failed: {me}")
            ACTIVE_TASKS.pop(chat_id, None)
            return clean_reply, media_to_send

    ACTIVE_TASKS.pop(chat_id, None)
    return (
        "⚠️ *Task stopped — too many steps reached (40 tool calls)*\n\n"
        "The agent ran 40 steps on this task without finishing. "
        "This usually means it got stuck in a loop or the task is too complex to complete in one go.\n\n"
        "What you can do:\n"
        "• /stop — Stop the current task\n"
        "• /clear — Reset the conversation and try a simpler or shorter instruction\n"
        "• Break your task into smaller parts and send them one at a time"
    ), media_to_send


# ── Telegram command handlers ─────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if get_owner_id() is None:
        mem.set_config("owner_telegram_id", str(user_id))
        await update.message.reply_text(
            f"👋 Welcome! Owner locked to your ID: `{user_id}`\n\n"
            "I'm your personal AI agent — I can run code, manage files, call APIs, "
            "remember things, and run background services on this droplet.\n\n"
            "Type anything or use /help.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text("🟢 Agent is running. Type /help for commands.")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        return
    await update.message.reply_text(
        "🤖 *Personal AI Agent — Commands*\n\n"
        "/start — Register as owner\n"
        "/help — This message\n"
        "/clear — Wipe conversation history\n"
        "/model \\[name\\] — Show or switch model\n"
        "/models — List all available models\n"
        "/status — Show running services\n"
        "/creds — List stored credentials\n"
        "/storekey \\<NAME\\> \\<VALUE\\> — Store a key directly \\(bypasses AI\\)\n"
        "/memory — Show remembered facts\n"
        "/run \\<cmd\\> — Run shell command directly\n"
        "/plan \\<task\\> — Break task into steps \\(no execution\\)\n"
        "/agent \\<task\\> — Autonomous execution mode\n"
        "/stop — Stop a running task\n"
        "/ping — Check if alive\n\n"
        "*Media:* Send me photos, voice, audio, video, or files\\. "
        "I'll save and process them\\. I can also send files back to you\\.\n\n"
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
    usage_summary = mem.get_model_usage_summary()

    groups = {
        "🌊 DigitalOcean": [],
        "🟣 Anthropic": [],
        "🟢 OpenAI": [],
    }

    for m in cfg.AVAILABLE_MODELS:
        marker = "▶️" if m == current else "  "
        pricing = cfg.MODEL_PRICING.get(m)
        if pricing:
            price_str = f"${pricing[0]:.2f}/${pricing[1]:.2f}"
        else:
            price_str = "?/?"

        u = usage_summary.get(m)
        if u and u["calls"] > 0:
            in_k  = u["input_tokens"] / 1000
            out_k = u["output_tokens"] / 1000
            if pricing:
                cost = (u["input_tokens"] * pricing[0] + u["output_tokens"] * pricing[1]) / 1_000_000
                usage_str = f"{in_k:.0f}K/{out_k:.0f}K tok · ${cost:.4f}"
            else:
                usage_str = f"{in_k:.0f}K/{out_k:.0f}K tok"
        else:
            usage_str = "no usage"

        entry = f"{marker} `{m}`\n      💰 {price_str}/M  📊 {usage_str}"

        if m.startswith("anthropic-"):
            groups["🟣 Anthropic"].append(entry)
        elif m.startswith("openai-"):
            groups["🟢 OpenAI"].append(entry)
        else:
            groups["🌊 DigitalOcean"].append(entry)

    lines = ["💰 = input/output per 1M tokens  📊 = your total usage"]
    for provider, models in groups.items():
        if models:
            lines.append(f"\n*{provider}*")
            lines.extend(models)

    # Totals
    total_cost = 0.0
    has_cost = False
    for m, u in usage_summary.items():
        p = cfg.MODEL_PRICING.get(m)
        if p and u["calls"] > 0:
            total_cost += (u["input_tokens"] * p[0] + u["output_tokens"] * p[1]) / 1_000_000
            has_cost = True
    if has_cost:
        lines.append(f"\n*Total estimated spend: ${total_cost:.4f}*")

    await update.message.reply_text(
        f"*Available Models* ({len(cfg.AVAILABLE_MODELS)} total):\n" + "\n".join(lines),
        parse_mode="Markdown",
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


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop a currently running agent task."""
    if not is_owner(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    if ACTIVE_TASKS.get(chat_id):
        ACTIVE_TASKS[chat_id] = False
        await update.message.reply_text("🛑 Stopping current task...")
    else:
        await update.message.reply_text("No task is currently running.")


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
        reply = (response.choices[0].message.content or "").strip()  # Fix #14
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
        agent_media: list[dict] = []
        ACTIVE_TASKS[chat_id] = True
        for iteration in range(cfg.MAX_TOOL_ITERATIONS):
            # Check stop flag
            if not ACTIVE_TASKS.get(chat_id, True):
                await update.message.reply_text("🛑 Task stopped.")
                ACTIVE_TASKS.pop(chat_id, None)
                await _send_queued_media(update, agent_media)
                return

            response = _llm_call(
                model=model, messages=messages, temperature=0.2, max_tokens=4096
            )
            # Track token usage per model
            try:
                usage = response.usage
                if usage:
                    mem.record_model_usage(model, usage.prompt_tokens, usage.completion_tokens)
            except Exception as ue:
                logger.debug(f"[/agent] Usage tracking error: {ue}")

            reply = (response.choices[0].message.content or "").strip()  # Fix #14
            if not reply:
                reply = "Done."

            # Log think block
            think_content = _extract_think_block(reply)
            if think_content:
                logger.info(f"🧠 [/agent] Think [{iteration+1}]: {think_content[:300]}")

            tool_calls = _parse_tool_calls(reply)
            if tool_calls:
                # Send intermediate progress text (think block stripped)
                pre_text = _extract_pre_tool_text(reply)
                if pre_text:
                    for chunk in [pre_text[i:i+4000] for i in range(0, len(pre_text), 4000)]:
                        await update.message.reply_text(chunk)

                # Execute ALL tool calls — parallel for independent tools
                combined_results = []

                # Stop check before firing tools
                if not ACTIVE_TASKS.get(chat_id, True):
                    await update.message.reply_text("🛑 Task stopped.")
                    ACTIVE_TASKS.pop(chat_id, None)
                    await _send_queued_media(update, agent_media)
                    return

                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                valid_calls = [(n, a) for n, a in tool_calls if n]
                _aloop = asyncio.get_running_loop()

                async def _agent_run_tool(idx, tname, targs):
                    logger.info(f"[/agent] Tool [{iteration+1}][{idx}]: {tname}({targs})")
                    return tname, await _aloop.run_in_executor(None, execute_tool, tname, targs)

                a_results = await asyncio.gather(
                    *[_agent_run_tool(i + 1, n, a) for i, (n, a) in enumerate(valid_calls)]
                )
                for tname, result in a_results:
                    if tname in ("send_media", "generate_image"):
                        try:
                            rd = json.loads(result)
                            if rd.get("queued"):
                                agent_media.append(rd)
                        except json.JSONDecodeError:
                            pass
                    combined_results.append(f"[{tname}]: {result}")

                messages.append({"role": "assistant", "content": reply})
                if len(combined_results) == 1:
                    result_text = combined_results[0]
                else:
                    result_text = "\n\n".join(
                        f"[{i+1}/{len(combined_results)}] {r}"
                        for i, r in enumerate(combined_results)
                    )
                if len(result_text) > 6000:
                    result_text = result_text[:5500] + "\n\n... [output truncated] ..."
                messages.append({
                    "role": "user",
                    "content": f"<tool_result>\n{result_text}\n</tool_result>\n"
                               "Check returncode: 0=success, non-zero=FAILED. Continue.",
                })
            else:
                # Guardrail: never send raw tagged tool payloads to chat
                if "<tool_call>" in reply or "</tool_call>" in reply:
                    logger.warning("[/agent] Tagged tool_call detected but parse failed; forcing retry")
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your last reply looked like a tool call but was invalid/truncated and could not be parsed. "
                            "Retry with ONLY a valid <tool_call> JSON using a known tool name from the tool list."
                        ),
                    })
                    continue

                clean_reply = _strip_think_block(reply) if reply else "Done."
                if not clean_reply:
                    clean_reply = "Done."
                for chunk in [clean_reply[i:i+4000] for i in range(0, len(clean_reply), 4000)]:
                    await update.message.reply_text(chunk)
                ACTIVE_TASKS.pop(chat_id, None)
                await _send_queued_media(update, agent_media)
                return
        ACTIVE_TASKS.pop(chat_id, None)
        await update.message.reply_text(
            "⚠️ *Task stopped — too many steps (40 tool calls)*\n\n"
            "The agent executed 40 steps without finishing.\n"
            "Use /stop to interrupt, /clear to reset, or break into smaller tasks.",
            parse_mode="Markdown",
        )
        await _send_queued_media(update, agent_media)
    except Exception as e:
        logger.error(f"cmd_agent error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ {e}")


# ── Media helpers ─────────────────────────────────────────────────────────────

async def _send_queued_media(update: Update, media_items: list[dict]):
    """Send all queued media files to the user after the text reply."""
    for item in media_items:
        path = item.get("path", "")
        media_type = item.get("type", "document")
        caption = item.get("caption", "")
        try:
            with open(path, "rb") as f:
                if media_type == "photo":
                    await update.message.reply_photo(f, caption=caption[:1024] or None)
                elif media_type == "video":
                    await update.message.reply_video(f, caption=caption[:1024] or None)
                elif media_type == "audio":
                    await update.message.reply_audio(f, caption=caption[:1024] or None)
                elif media_type == "voice":
                    await update.message.reply_voice(f, caption=caption[:1024] or None)
                else:
                    await update.message.reply_document(f, caption=caption[:1024] or None)
        except Exception as e:
            logger.error(f"Failed to send media {path}: {e}")
            await update.message.reply_text(f"⚠️ Couldn't send file: {e}")


# ── Message handler ───────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_owner(user_id):
        await update.message.reply_text("⛔ Unauthorized.")
        return

    chat_id = update.effective_chat.id
    model = get_current_model()

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    async def _send(text):
        for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
            await update.message.reply_text(chunk)
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        reply, media_items = await run_agent(chat_id, update.message.text, model, send_fn=_send)
        if not reply or not reply.strip():
            reply = "_(got an empty response — try rephrasing or /clear to reset history)_"
        # Telegram message limit is 4096 chars
        for chunk in [reply[i:i+4000] for i in range(0, len(reply), 4000)]:
            await update.message.reply_text(chunk)
        # Send any queued media
        await _send_queued_media(update, media_items)
    except Exception as e:
        logger.error(f"handle_message error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ {e}")


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming media: photos, voice, audio, video, documents, stickers."""
    user_id = update.effective_user.id
    if not is_owner(user_id):
        return

    chat_id = update.effective_chat.id
    model = get_current_model()
    msg = update.message

    # Determine media type and get Telegram file object
    original_name = None
    if msg.photo:
        tg_file = await msg.photo[-1].get_file()  # highest resolution
        media_type, subfolder, ext = "photo", "photos", ".jpg"
    elif msg.voice:
        tg_file = await msg.voice.get_file()
        media_type, subfolder, ext = "voice message", "voice", ".ogg"
    elif msg.audio:
        tg_file = await msg.audio.get_file()
        media_type, subfolder, ext = "audio", "audio", ".mp3"
        original_name = msg.audio.file_name
    elif msg.video:
        tg_file = await msg.video.get_file()
        media_type, subfolder, ext = "video", "video", ".mp4"
    elif msg.video_note:
        tg_file = await msg.video_note.get_file()
        media_type, subfolder, ext = "video note", "video", ".mp4"
    elif msg.document:
        tg_file = await msg.document.get_file()
        media_type, subfolder = "document", "documents"
        original_name = msg.document.file_name
        ext = Path(original_name).suffix if original_name else ""
    elif msg.sticker:
        tg_file = await msg.sticker.get_file()
        media_type, subfolder, ext = "sticker", "stickers", ".webp"
    else:
        return

    # Build filename and save
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = cfg.MEDIA_DIR / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)

    if original_name:
        filename = f"{ts}_{original_name}"
    else:
        filename = f"{media_type.replace(' ', '_')}_{ts}{ext}"

    save_path = save_dir / filename
    await tg_file.download_to_drive(str(save_path))

    # Build text description for the LLM
    file_size = save_path.stat().st_size
    size_str = f"{file_size / 1024:.1f}KB" if file_size < 1048576 else f"{file_size / 1048576:.1f}MB"
    caption = msg.caption or ""

    desc = f"[User sent a {media_type}: saved as {save_path}, {size_str}]"
    if caption:
        desc = f"{caption}\n\n{desc}"

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    async def _send(text):
        for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
            await update.message.reply_text(chunk)
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        reply, media_items = await run_agent(chat_id, desc, model, send_fn=_send)
        if not reply or not reply.strip():
            reply = f"📎 Got your {media_type}! Saved as `{filename}`"
        for chunk in [reply[i:i+4000] for i in range(0, len(reply), 4000)]:
            await update.message.reply_text(chunk)
        await _send_queued_media(update, media_items)
    except Exception as e:
        logger.error(f"handle_media error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    app = Application.builder().token(cfg.TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("help",    cmd_help))
    app.add_handler(CommandHandler("clear",   cmd_clear))
    app.add_handler(CommandHandler("model",   cmd_model))
    app.add_handler(CommandHandler("models",  cmd_models))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("creds",    cmd_creds))
    app.add_handler(CommandHandler("storekey", cmd_storekey))
    app.add_handler(CommandHandler("memory",  cmd_memory_cmd))
    app.add_handler(CommandHandler("run",     cmd_run))
    app.add_handler(CommandHandler("ping",    cmd_ping))
    app.add_handler(CommandHandler("stop",    cmd_stop))
    app.add_handler(CommandHandler("plan",    cmd_plan))
    app.add_handler(CommandHandler("agent",   cmd_agent))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Media handlers — receive photos, voice, audio, video, documents, stickers
    media_filter = (
        filters.PHOTO | filters.VOICE | filters.AUDIO |
        filters.VIDEO | filters.VIDEO_NOTE |
        filters.Document.ALL | filters.Sticker.ALL
    )
    app.add_handler(MessageHandler(media_filter, handle_media))

    logger.info("🤖 Agent starting (polling)…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
