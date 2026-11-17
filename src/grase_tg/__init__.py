"""Точка входа семейного Telegram-бота."""

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
    """Прокси-адрес для логов: пароль и логин не светим."""
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
    session = None
    if config.secrets.telegram_proxy_url:
        session = AiohttpSession(proxy=config.secrets.telegram_proxy_url)
        logger.info(
            "Подключение к Telegram через HTTP-прокси: %s",
            _masked_proxy(config.secrets.telegram_proxy_url),
        )
    bot = Bot(token=config.secrets.bot_token, session=session)
    try:
        me = await bot.get_me()
    except TelegramUnauthorizedError:
        logger.error("Telegram отклонил токен (Unauthorized) — проверь BOT_TOKEN в .env")
        raise SystemExit(1) from None
    if me.username is None:
        logger.error("У бота нет username — @упоминания работать не смогут.")
        return

    dp = Dispatcher()
    dp.include_router(build_router(config, make_client(config), me.username))
    logger.info(
        "Бот @%s запущен. Разрешённые группы: %s",
        me.username,
        config.app.allowed_chat_ids or "не настроены",
    )
    await dp.start_polling(bot, allowed_updates=["message"])


def main() -> None:
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        pass
