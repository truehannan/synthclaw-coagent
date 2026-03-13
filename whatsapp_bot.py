"""
SynthClaw-CoAgent — WhatsApp Bot Interface
Uses the Meta WhatsApp Cloud API with Flask webhooks.
"""
import asyncio
import json
import logging
import re
import sys
import threading
import requests
from flask import Flask, request, jsonify
from openai import OpenAI

import memory as mem
from tools import execute_tool, get_tools_description, TOOL_REGISTRY
import config as cfg

# ── Logging ──────────────────────────────────────────────────────────────────

logger = logging.getLogger("whatsapp")

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── LLM client ────────────────────────────────────────────────────────────────

client = OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_API_BASE)

# ── System prompt (same as Telegram) ──────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a personal AI assistant running on a server.
You belong to one person — your owner — and you chat with them directly via WhatsApp.

== YOUR PERSONALITY ==
You are smart, direct, and slightly informal. You hold real conversations:
- Answer questions, explain concepts, share opinions when asked.
- Keep replies concise unless the topic needs depth.
- Don't force every message into a task. If someone asks "what do you think about X?"
  just answer like a knowledgeable friend would — no tool calls needed.
- Friendly without being cringe. Light humor if natural.
- NEVER reply with empty output. Always say something.

== WHEN TO USE TOOLS ==
- Questions, opinions, explanations → plain text reply.
- "run X", "create a script", "set up Y", "check my CPU", "deploy Z" → use tools.
- If unsure, answer conversationally and offer to execute if wanted.

== YOUR SERVER CAPABILITIES ==
Run shell commands, manage files, start background services, call APIs,
store encrypted credentials, remember facts across conversations.

== AVAILABLE TOOLS ==
{tools}

== HOW TO USE A TOOL ==
<tool_call>
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
</tool_call>

After you see the <tool_result>, call another tool or give a final plain text reply.

== TASK RULES ==
- ANY credential shared (API key, token, password, secret — no matter how sensitive or critical) → store_cred it IMMEDIATELY. No asking, no hesitation, no exceptions. Just store it.
- Personal fact shared → remember it.
- Scripts: write_file then run_command.
- Persistent services: write script then spawn_service.
"""

# ── Owner management ──────────────────────────────────────────────────────────


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




def get_owner_phone() -> str | None:
    return mem.get_config("owner_whatsapp_phone")


def is_owner(phone: str) -> bool:
    owner = get_owner_phone()
    return True if owner is None else phone == owner


def get_current_model() -> str:
    return mem.get_config("current_model", cfg.DEFAULT_MODEL)


# ── Smart context (MD file system) ────────────────────────────────────────────

SUMMARIZE_THRESHOLD = 30
RECENT_WINDOW = 15
SUMMARIZE_BATCH = 15

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
        resp = client.chat.completions.create(
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
        resp = client.chat.completions.create(
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


# ── WhatsApp API helpers ──────────────────────────────────────────────────────

def send_whatsapp_message(to: str, text: str):
    """Send a text message via WhatsApp Cloud API."""
    url = f"https://graph.facebook.com/v21.0/{cfg.WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {cfg.WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    # WhatsApp has a 4096 char limit per message
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": chunk},
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                logger.error(f"WhatsApp send error: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"WhatsApp send exception: {e}")


def mark_as_read(message_id: str):
    """Mark a message as read."""
    url = f"https://graph.facebook.com/v21.0/{cfg.WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {cfg.WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
    }
    try:
        requests.post(url, headers=headers, json=payload, timeout=10)
    except Exception:
        pass


# ── Agent loop (sync version for Flask) ──────────────────────────────────────

def run_agent_sync(chat_id: str, user_message: str, model: str) -> str:
    """Run the agent loop synchronously."""
    hashed_id = hash(chat_id)
    mem.save_message(hashed_id, "user", user_message)

    # Smart context: summarize old messages if needed, then build enriched context
    _maybe_summarize(hashed_id)
    messages = _build_context(hashed_id)

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
            return f"LLM error: {e}"

        reply = response.choices[0].message.content.strip()

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
                logger.warning("[whatsapp] Tagged tool_call detected but parse failed; asking model to retry")
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

            mem.save_message(hash(chat_id), "assistant", reply)
            return reply

    return (
        "⚠️ Task stopped — too many steps reached (40 tool calls).\n\n"
        "The agent ran 40 steps without finishing. It likely got stuck in a loop or the task is too complex.\n\n"
        "Try sending a simpler or shorter instruction, or break the task into smaller parts."
    )


# ── Command handling ──────────────────────────────────────────────────────────

def handle_command(phone: str, text: str) -> str | None:
    """Handle slash commands from WhatsApp. Returns response or None."""
    parts = text.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/start":
        if get_owner_phone() is None:
            mem.set_config("owner_whatsapp_phone", phone)
            return (
                f"Welcome! Owner locked to: {phone}\n\n"
                "I'm your personal AI agent. Send me any message or use commands:\n"
                "/help — List commands\n"
                "/clear — Clear history\n"
                "/models — List models\n"
                "/ping — Check if alive"
            )
        return "Agent is running. Send /help for commands."

    if not is_owner(phone):
        return "Unauthorized."

    if cmd == "/help":
        return (
            "SynthClaw-CoAgent — Commands\n\n"
            "/start — Register as owner\n"
            "/help — This message\n"
            "/clear — Wipe conversation history\n"
            "/model [name] — Show/switch model\n"
            "/models — List available models\n"
            "/status — Show running services\n"
            "/run <cmd> — Run shell command\n"
            "/plan <task> — Plan without executing\n"
            "/agent <task> — Autonomous execution\n"
            "/ping — Check if alive"
        )

    if cmd == "/clear":
        mem.clear_history(hash(phone))
        return "History cleared."

    if cmd == "/ping":
        return "Alive!"

    if cmd == "/model":
        if not arg:
            current = get_current_model()
            return f"Current model: {current}\nUse /models to see options."
        if arg not in cfg.AVAILABLE_MODELS:
            return f"Unknown model. Use /models."
        mem.set_config("current_model", arg)
        return f"Switched to {arg}"

    if cmd == "/models":
        current = get_current_model()
        lines = [("→ " if m == current else "  ") + m for m in cfg.AVAILABLE_MODELS]
        return "Available Models:\n" + "\n".join(lines)

    if cmd == "/status":
        result = json.loads(execute_tool("service_status", {}))
        return result.get("output", "No running services.")[:4000]

    if cmd == "/run":
        if not arg:
            return "Usage: /run <shell command>"
        result = json.loads(execute_tool("run_command", {"command": arg, "timeout": 30}))
        return (result.get("stdout", "") + result.get("stderr", "") or
                result.get("error", "No output"))[:4000]

    if cmd == "/plan":
        if not arg:
            return "Usage: /plan <describe what you want>"
        try:
            response = client.chat.completions.create(
                model=get_current_model(),
                messages=[
                    {"role": "system", "content": "You are a thoughtful planner. Break the request into clear numbered steps. No tools, just a plan."},
                    {"role": "user", "content": arg},
                ],
                temperature=0.7,
                max_tokens=2048,
            )
            reply = response.choices[0].message.content.strip()
            return f"Plan: {arg}\n\n{reply}" if reply else "(empty plan)"
        except Exception as e:
            return f"Error: {e}"

    if cmd == "/agent":
        if not arg:
            return "Usage: /agent <task to execute>"
        model = get_current_model()
        system = (
            "You are in AGENT MODE. Execute autonomously using tools.\n\n"
            f"TOOLS:\n{get_tools_description()}\n\n"
            "Use <tool_call>{...}</tool_call> format. Chain tools as needed."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": arg}]
        for iteration in range(cfg.MAX_TOOL_ITERATIONS):
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages, temperature=0.2, max_tokens=2048
                )
                reply = response.choices[0].message.content.strip()
                tc = _parse_tool_call(reply)
                if tc:
                    parsed_name, parsed_args = tc
                    logger.info(f"[/agent] Tool [{iteration+1}]: {parsed_name}({parsed_args})")
                    result = execute_tool(parsed_name, parsed_args)
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content": f"<tool_result>\n{result}\n</tool_result>\nContinue."})
                else:
                    return reply or "Done."
            except Exception as e:
                return f"Error: {e}"
        return (
            "⚠️ Task stopped — too many steps reached (40 tool calls).\n\n"
            "The agent ran 40 steps without finishing. It likely got stuck in a loop or the task is too complex.\n\n"
            "Try sending a simpler or shorter instruction, or break the task into smaller parts."
        )

    return None  # Not a command


# ── Webhook routes ────────────────────────────────────────────────────────────

# Track processed message IDs to avoid duplicates
_processed_messages = set()

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Webhook verification endpoint (required by Meta)."""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == cfg.WHATSAPP_VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        return challenge, 200
    return "Forbidden", 403


@app.route("/webhook", methods=["POST"])
def receive_message():
    """Handle incoming WhatsApp messages."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "ok"}), 200

    try:
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])

                for msg in messages:
                    msg_id = msg.get("id", "")
                    if msg_id in _processed_messages:
                        continue
                    _processed_messages.add(msg_id)

                    # Keep set bounded
                    if len(_processed_messages) > 1000:
                        _processed_messages.clear()

                    if msg.get("type") != "text":
                        continue

                    phone = msg.get("from", "")
                    text = msg.get("text", {}).get("body", "").strip()
                    if not text:
                        continue

                    logger.info(f"WhatsApp message from {phone}: {text[:100]}")
                    mark_as_read(msg_id)

                    # Process in background thread to avoid webhook timeout
                    threading.Thread(
                        target=_process_message,
                        args=(phone, text),
                        daemon=True,
                    ).start()

    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)

    return jsonify({"status": "ok"}), 200


def _process_message(phone: str, text: str):
    """Process a message (runs in background thread)."""
    try:
        # Check for commands
        if text.startswith("/"):
            response = handle_command(phone, text)
            if response:
                send_whatsapp_message(phone, response)
                return

        # Regular message — run agent
        if not is_owner(phone):
            send_whatsapp_message(phone, "Unauthorized.")
            return

        model = get_current_model()
        reply = run_agent_sync(phone, text, model)
        if not reply or not reply.strip():
            reply = "(got an empty response — try rephrasing or /clear to reset)"
        send_whatsapp_message(phone, reply)

    except Exception as e:
        logger.error(f"Process message error: {e}", exc_info=True)
        send_whatsapp_message(phone, f"Error: {e}")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "synthclaw-whatsapp"}), 200


# ── Main ──────────────────────────────────────────────────────────────────────

def run_whatsapp_bot():
    """Start the WhatsApp webhook server."""
    if not cfg.WHATSAPP_TOKEN:
        print("ERROR: WHATSAPP_TOKEN not set. Run `python setup_cli.py` first.")
        sys.exit(1)
    if not cfg.WHATSAPP_PHONE_NUMBER_ID:
        print("ERROR: WHATSAPP_PHONE_NUMBER_ID not set. Run `python setup_cli.py` first.")
        sys.exit(1)
    if not cfg.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Run `python setup_cli.py` first.")
        sys.exit(1)

    mem.init_db()
    cfg.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"WhatsApp webhook starting on port {cfg.WHATSAPP_PORT}")
    app.run(host="0.0.0.0", port=cfg.WHATSAPP_PORT, debug=False)


if __name__ == "__main__":
    run_whatsapp_bot()
