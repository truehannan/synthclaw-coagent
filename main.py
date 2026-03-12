#!/usr/bin/env python3
"""
SynthClaw-CoAgent — Main Entry Point
Launches Telegram, WhatsApp, or both interfaces based on INTERFACE_MODE.
"""
import sys
import threading
import logging
import config as cfg

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(cfg.LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("synthclaw")


def main():
    mode = cfg.INTERFACE_MODE

    if mode == "telegram":
        from agent import main as telegram_main
        logger.info("Starting in Telegram mode")
        telegram_main()

    elif mode == "whatsapp":
        from whatsapp_bot import run_whatsapp_bot
        logger.info("Starting in WhatsApp mode")
        run_whatsapp_bot()

    elif mode == "both":
        from whatsapp_bot import run_whatsapp_bot

        # Start WhatsApp in a separate thread
        logger.info("Starting in dual mode (Telegram + WhatsApp)")
        wa_thread = threading.Thread(target=run_whatsapp_bot, daemon=True)
        wa_thread.start()

        # Telegram runs in main thread (uses asyncio)
        from agent import main as telegram_main
        telegram_main()

    else:
        print(f"ERROR: Unknown INTERFACE_MODE '{mode}'. Use: telegram, whatsapp, or both")
        sys.exit(1)


if __name__ == "__main__":
    main()
