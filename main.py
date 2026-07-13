#!/usr/bin/env python3
"""
SynthClaw-CoAgent — Main Entry Point
Launches Telegram, WhatsApp, or both interfaces based on INTERFACE_MODE.
Also starts the REST API server for the web frontend.
"""
import sys
import threading
import logging
import os
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

API_PORT = int(os.getenv("SYNTHCLAW_API_PORT", "8000"))
API_HOST = os.getenv("SYNTHCLAW_API_HOST", "0.0.0.0")


def start_api():
    """Start the REST API server in a background thread."""
    try:
        from api_server import start_api_server_background
        start_api_server_background(host=API_HOST, port=API_PORT)
        logger.info(f"API server running at http://{API_HOST}:{API_PORT}")
    except ImportError as e:
        logger.warning(f"Could not start API server (missing deps?): {e}")
    except Exception as e:
        logger.warning(f"API server failed to start: {e}")


def main():
    mode = cfg.INTERFACE_MODE

    # Initialize database before anything else
    import memory as mem
    mem.init_db()

    # Always start the API server (for frontend access)
    start_api()

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

    elif mode == "cli":
        # CLI-only mode: just run the API server in foreground
        logger.info("Starting in CLI-only mode (API server only)")
        from api_server import start_api_server
        start_api_server(host=API_HOST, port=API_PORT)

    else:
        print(f"ERROR: Unknown INTERFACE_MODE '{mode}'. Use: telegram, whatsapp, both, or cli")
        sys.exit(1)


if __name__ == "__main__":
    main()
