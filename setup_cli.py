#!/usr/bin/env python3
"""
SynthClaw-CoAgent — Interactive Setup CLI
Generates the .env file with all required configuration.
"""
import os
import sys
import secrets
import string
from pathlib import Path

ENV_FILE = Path(__file__).parent / ".env"

BANNER = """
╔═══════════════════════════════════════════════════╗
║         SynthClaw-CoAgent — Setup Wizard          ║
║         Personal AI Agent Configuration           ║
╚═══════════════════════════════════════════════════╝
"""


def ask(prompt: str, default: str = "", secret: bool = False, required: bool = True) -> str:
    """Prompt user for input with optional default."""
    suffix = f" [{default}]" if default else ""
    while True:
        if secret:
            try:
                import getpass
                value = getpass.getpass(f"  {prompt}{suffix}: ") or default
            except Exception:
                value = input(f"  {prompt}{suffix}: ") or default
        else:
            value = input(f"  {prompt}{suffix}: ") or default
        if value or not required:
            return value
        print("  ⚠ This field is required.")


def ask_choice(prompt: str, options: list[str], default: str = "") -> str:
    """Prompt user to choose from options."""
    print(f"\n  {prompt}")
    for i, opt in enumerate(options, 1):
        marker = " (default)" if opt == default else ""
        print(f"    {i}. {opt}{marker}")
    while True:
        choice = input(f"  Choice [1-{len(options)}]: ").strip()
        if not choice and default:
            return default
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            if choice in options:
                return choice
        print(f"  ⚠ Enter a number 1-{len(options)}")


def generate_verify_token(length: int = 24) -> str:
    """Generate a random verification token."""
    chars = string.ascii_letters + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))


def main():
    print(BANNER)

    # Load existing .env if present
    existing = {}
    if ENV_FILE.exists():
        print("  Found existing .env — current values will be shown as defaults.\n")
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                existing[key.strip()] = value.strip().strip('"').strip("'")
    else:
        print("  No .env found — creating new configuration.\n")

    config = {}

    # ── Interface mode ────────────────────────────────────────────────────
    print("━" * 50)
    print("  1. INTERFACE MODE")
    print("━" * 50)
    mode = ask_choice(
        "Which messaging interface(s)?",
        ["telegram", "whatsapp", "both"],
        default=existing.get("INTERFACE_MODE", "telegram"),
    )
    config["INTERFACE_MODE"] = mode

    # ── Telegram ──────────────────────────────────────────────────────────
    if mode in ("telegram", "both"):
        print("\n" + "━" * 50)
        print("  2. TELEGRAM CONFIGURATION")
        print("━" * 50)
        print("  Get your bot token from @BotFather on Telegram.\n")
        config["TELEGRAM_TOKEN"] = ask(
            "Telegram Bot Token",
            default=existing.get("TELEGRAM_TOKEN", ""),
            secret=True,
        )

    # ── WhatsApp ──────────────────────────────────────────────────────────
    if mode in ("whatsapp", "both"):
        print("\n" + "━" * 50)
        print("  3. WHATSAPP CONFIGURATION")
        print("━" * 50)
        print("  You need a Meta WhatsApp Business API account.")
        print("  → https://developers.facebook.com/docs/whatsapp/cloud-api/get-started\n")
        config["WHATSAPP_TOKEN"] = ask(
            "WhatsApp API Access Token",
            default=existing.get("WHATSAPP_TOKEN", ""),
            secret=True,
        )
        config["WHATSAPP_PHONE_NUMBER_ID"] = ask(
            "WhatsApp Phone Number ID",
            default=existing.get("WHATSAPP_PHONE_NUMBER_ID", ""),
        )
        verify = existing.get("WHATSAPP_VERIFY_TOKEN", generate_verify_token())
        config["WHATSAPP_VERIFY_TOKEN"] = ask(
            "Webhook Verify Token (auto-generated)",
            default=verify,
        )
        config["WHATSAPP_PORT"] = ask(
            "Webhook Port",
            default=existing.get("WHATSAPP_PORT", "8443"),
            required=False,
        ) or "8443"

    # ── LLM Provider ─────────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("  4. LLM PROVIDER")
    print("━" * 50)
    print("  Any OpenAI-compatible API works (OpenAI, DigitalOcean AI, Ollama, etc.)\n")
    config["OPENAI_API_KEY"] = ask(
        "API Key",
        default=existing.get("OPENAI_API_KEY", ""),
        secret=True,
    )
    config["OPENAI_API_BASE"] = ask(
        "API Base URL",
        default=existing.get("OPENAI_API_BASE", "https://inference.do-ai.run/v1"),
    )
    config["DEFAULT_MODEL"] = ask(
        "Default Model",
        default=existing.get("DEFAULT_MODEL", "llama3.3-70b-instruct"),
    )

    # ── Server settings ──────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("  5. SERVER SETTINGS")
    print("━" * 50)
    config["SYNTHCLAW_BASE_DIR"] = ask(
        "Base directory",
        default=existing.get("SYNTHCLAW_BASE_DIR", "/opt/agent"),
    )
    config["MAX_TOOL_ITERATIONS"] = ask(
        "Max tool iterations per message",
        default=existing.get("MAX_TOOL_ITERATIONS", "10"),
        required=False,
    ) or "10"
    config["MAX_HISTORY_MESSAGES"] = ask(
        "Max conversation history messages",
        default=existing.get("MAX_HISTORY_MESSAGES", "20"),
        required=False,
    ) or "20"

    # ── Write .env ────────────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("  WRITING CONFIGURATION")
    print("━" * 50)

    lines = [
        "# SynthClaw-CoAgent Configuration",
        "# Generated by setup_cli.py — DO NOT commit this file",
        "",
        "# Interface: telegram | whatsapp | both",
        f'INTERFACE_MODE="{config["INTERFACE_MODE"]}"',
        "",
    ]

    if "TELEGRAM_TOKEN" in config:
        lines += [
            "# Telegram",
            f'TELEGRAM_TOKEN="{config["TELEGRAM_TOKEN"]}"',
            "",
        ]

    if "WHATSAPP_TOKEN" in config:
        lines += [
            "# WhatsApp (Meta Cloud API)",
            f'WHATSAPP_TOKEN="{config["WHATSAPP_TOKEN"]}"',
            f'WHATSAPP_PHONE_NUMBER_ID="{config["WHATSAPP_PHONE_NUMBER_ID"]}"',
            f'WHATSAPP_VERIFY_TOKEN="{config["WHATSAPP_VERIFY_TOKEN"]}"',
            f'WHATSAPP_PORT="{config["WHATSAPP_PORT"]}"',
            "",
        ]

    lines += [
        "# LLM Provider (any OpenAI-compatible API)",
        f'OPENAI_API_KEY="{config["OPENAI_API_KEY"]}"',
        f'OPENAI_API_BASE="{config["OPENAI_API_BASE"]}"',
        f'DEFAULT_MODEL="{config["DEFAULT_MODEL"]}"',
        "",
        "# Server",
        f'SYNTHCLAW_BASE_DIR="{config["SYNTHCLAW_BASE_DIR"]}"',
        f'MAX_TOOL_ITERATIONS="{config["MAX_TOOL_ITERATIONS"]}"',
        f'MAX_HISTORY_MESSAGES="{config["MAX_HISTORY_MESSAGES"]}"',
    ]

    ENV_FILE.write_text("\n".join(lines) + "\n")
    print(f"\n  ✅ Configuration saved to {ENV_FILE}")
    print(f"  ⚠  Do NOT commit .env to git — it contains secrets.\n")

    # ── Next steps ────────────────────────────────────────────────────────
    print("━" * 50)
    print("  NEXT STEPS")
    print("━" * 50)
    print("""
  1. Deploy to your server:
     scp -r ./* user@your-server:/opt/agent/

  2. Run the setup script:
     ssh user@your-server 'bash /opt/agent/setup_server.sh'

  3. Start the agent:
     ssh user@your-server 'systemctl start agent'

  4. Open Telegram and send /start to your bot!
""")


if __name__ == "__main__":
    main()
