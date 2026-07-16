#!/usr/bin/env python3
"""
Conclave-CoAgent — Main Entry Point
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
logger = logging.getLogger("conclave")

API_PORT = int(os.getenv("CONCLAVE_API_PORT", "8000"))
API_HOST = os.getenv("CONCLAVE_API_HOST", "0.0.0.0")


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


def _sync_env_to_db(mem):
    """Sync env credentials to DB for persistence/backup.
    Rule: DB is source of truth. Only writes to DB if DB is empty AND env has value."""
    env_creds = {
        "OPENAI_API_KEY": cfg.OPENAI_API_KEY,
        "COMPOSIO_API_KEY": cfg.COMPOSIO_API_KEY,
        "GOOGLE_SEARCH_API_KEY": cfg.GOOGLE_SEARCH_API_KEY,
    }
    env_memory = {
        "cf_account_id": cfg.CF_ACCOUNT_ID,
    }
    for name, value in env_creds.items():
        if value and not mem.get_credential(name):
            try:
                mem.store_credential(name, value, f"Synced from env on startup")
                logger.info(f"Synced {name} from env to credential store")
            except Exception:
                pass
    for key, value in env_memory.items():
        if value and not mem.get_memory(key):
            try:
                mem.set_memory(key, value)
                logger.info(f"Synced {key} from env to memory store")
            except Exception:
                pass


def main():
    mode = cfg.INTERFACE_MODE

    # Initialize database before anything else
    import memory as mem
    mem.init_db()

    # Sync env credentials to DB
    _sync_env_to_db(mem)

    # Always start the API server (for frontend access)
    # In CLI mode, we start it in foreground (blocking), so skip background start
    if mode != "cli":
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
