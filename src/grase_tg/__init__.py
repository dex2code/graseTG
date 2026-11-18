"""Family Telegram bot entry point: startup, wiring, polling."""

import asyncio
import logging

from contextlib import AsyncExitStack
from urllib.parse import urlsplit

import yaml
from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramUnauthorizedError
from aiogram.utils.token import TokenValidationError
from pydantic import ValidationError

from .config import Config, default_config_dir, load_config
from .content import BotIdentity
from .handlers import build_router
from .llm import make_client
from .mcp_bridge import connect_mcp
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

# Exactly the proxy schemes aiogram's AiohttpSession can dial (via python-socks).
# https://, socks4a:// and socks5h:// are deliberately absent: the connector
# rejects them, so they fail fast at startup instead of at the first request.
SUPPORTED_PROXY_SCHEMES = frozenset({"http", "socks4", "socks5"})


def _masked_proxy(proxy_url: str) -> str:
    """Return a proxy URL safe for logs: credentials stripped."""
    parts = urlsplit(proxy_url)
    if parts.username is None:
        return proxy_url
    netloc = f"{parts.hostname or ''}:{parts.port or ''}"
    return f"{parts.scheme}://{netloc}"


def _build_session(proxy_url: str) -> AiohttpSession:
    """Create a proxied Telegram session.

    HTTP and SOCKS proxies share one path: aiogram routes proxied sessions
    through the aiohttp-socks connector (with remote DNS). Raises ValueError
    for an unsupported scheme or a malformed URL.
    """
    scheme = urlsplit(proxy_url).scheme.lower()
    if scheme not in SUPPORTED_PROXY_SCHEMES:
        raise ValueError(
            f"unsupported proxy scheme {scheme or '(missing)'} in {proxy_url!r}; "
            f"expected one of: {', '.join(sorted(SUPPORTED_PROXY_SCHEMES))}"
        )
    return AiohttpSession(proxy=proxy_url)


async def _build_mcp_registry(config: Config, stack: AsyncExitStack) -> ToolRegistry | None:
    """Wire the single configured MCP server (if any) into a ToolRegistry.

    The two failure classes from docs/mcp-tools-spec.md §6.1:
    - configuration errors (bad url, missing auth env var) raise ValueError
      -> clean startup exit: retries cannot help, the user must fix the config;
    - network unavailability -> warning + None: a family bot without tools
      still answers from model knowledge, and a restart race with the MCP
      server must not take the whole bot down.
    """
    mcp_settings = config.app.tools.mcp
    if mcp_settings is None:
        return None
    try:
        return await connect_mcp(mcp_settings, stack)
    except ValueError as error:
        logger.error("Invalid tools.mcp configuration: %s", error)
        raise SystemExit(1) from None
    except Exception as error:
        # Broad on purpose (httpx transports, anyio streams, McpError, ...):
        # anything that is NOT a ValueError is treated as infrastructure
        # unavailability. The class name in the log keeps an actual code bug
        # from hiding behind the "unreachable" wording.
        logger.warning(
            "MCP server is unreachable (%s: %s) — running without tools",
            type(error).__name__,
            error,
        )
        return None


async def run_bot() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_dir = default_config_dir()
    # Logged before loading: if .env/config.yaml are missing, the log shows
    # where the bot looked for them.
    logger.info("Config directory: %s", config_dir)
    try:
        config = load_config(config_dir)
    except (ValidationError, yaml.YAMLError) as error:
        # A config typo must be a clean one-line startup failure, not a raw
        # pydantic/yaml traceback in the journal.
        logger.error("Invalid configuration in %s:\n%s", config_dir, error)
        raise SystemExit(1) from None

    # Everything below shares one exit stack: the MCP session (when
    # configured) must live as long as the bot and be closed on every way out,
    # including the SystemExit paths and KeyboardInterrupt during polling.
    async with AsyncExitStack() as stack:
        registry = await _build_mcp_registry(config, stack)

        # Optional HTTP/SOCKS proxy for Telegram traffic only (LLM requests stay direct).
        session = None
        if config.secrets.telegram_proxy_url:
            proxy_url = config.secrets.telegram_proxy_url
            try:
                session = _build_session(proxy_url)
            except ValueError as error:
                # Unsupported scheme, missing port, empty host — fail fast.
                logger.error("Invalid TELEGRAM_PROXY_URL: %s", error)
                raise SystemExit(1) from None
            logger.info("Connecting to Telegram via proxy: %s", _masked_proxy(proxy_url))

        # aiogram validates the token format at construction; a malformed token
        # must be a clean startup failure (and must not leak the proxy session).
        try:
            bot = Bot(token=config.secrets.bot_token.get_secret_value(), session=session)
        except TokenValidationError:
            if session is not None:
                await session.close()
            logger.error("Malformed BOT_TOKEN — expected '123456789:AA…' from @BotFather. Check .env")
            raise SystemExit(1) from None

        # Verify the token before starting the polling loop.
        # On any fatal startup problem: close the session, log clearly, exit non-zero.
        try:
            me = await bot.get_me()
        except TelegramUnauthorizedError:
            await bot.session.close()
            logger.error("Telegram rejected the token (Unauthorized) — check BOT_TOKEN in .env")
            raise SystemExit(1) from None
        except Exception:
            # Network, DNS or proxy failures: a clear message instead of a raw traceback.
            await bot.session.close()
            logger.exception("Failed to reach the Telegram API — check network and TELEGRAM_PROXY_URL")
            raise SystemExit(1) from None
        if me.username is None:
            await bot.session.close()
            logger.error("The bot has no username — @mentions will not work.")
            raise SystemExit(1) from None

        dp = Dispatcher()
        dp.include_router(
            build_router(
                config,
                make_client(config),
                BotIdentity(username=me.username, user_id=me.id),
                registry,
            )
        )
        logger.info(
            "Bot @%s started. Allowed groups: %s",
            me.username,
            config.app.allowed_chat_ids or "none configured",
        )
        # "message" is the only update type this bot needs. Updates queued by
        # Telegram while the bot was down are dropped: no delayed replies to
        # stale mentions after a restart.
        await dp.start_polling(bot, allowed_updates=["message"], drop_pending_updates=True)


def main() -> None:
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        pass
