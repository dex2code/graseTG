"""Family Telegram bot entry point: startup, wiring, polling."""

import asyncio
import logging

from urllib.parse import urlsplit

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramUnauthorizedError

from .config import load_config
from .handlers import build_router
from .llm import make_client

logger = logging.getLogger(__name__)


def _masked_proxy(proxy_url: str) -> str:
    """Return a proxy URL safe for logs: credentials stripped."""
    parts = urlsplit(proxy_url)
    if parts.username is None:
        return proxy_url
    netloc = f"{parts.hostname or ''}:{parts.port or ''}"
    return f"{parts.scheme}://{netloc}"


async def run_bot() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config()

    # Optional HTTP proxy for Telegram traffic only (LLM requests stay direct).
    session = None
    if config.secrets.telegram_proxy_url:
        session = AiohttpSession(proxy=config.secrets.telegram_proxy_url)
        logger.info(
            "Connecting to Telegram via HTTP proxy: %s",
            _masked_proxy(config.secrets.telegram_proxy_url),
        )

    bot = Bot(token=config.secrets.bot_token.get_secret_value(), session=session)

    # Verify the token before starting the polling loop.
    try:
        me = await bot.get_me()
    except TelegramUnauthorizedError:
        logger.error("Telegram rejected the token (Unauthorized) — check BOT_TOKEN in .env")
        raise SystemExit(1) from None
    if me.username is None:
        logger.error("The bot has no username — @mentions will not work.")
        return

    dp = Dispatcher()
    dp.include_router(build_router(config, make_client(config), me.username))
    logger.info(
        "Bot @%s started. Allowed groups: %s",
        me.username,
        config.app.allowed_chat_ids or "none configured",
    )
    # "message" is the only update type this bot needs.
    await dp.start_polling(bot, allowed_updates=["message"])


def main() -> None:
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        pass
