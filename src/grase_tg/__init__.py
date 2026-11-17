import asyncio
import logging
import os
import sys

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message

dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """Handle /start: greet the user."""
    if message.from_user is None:
        await message.answer("Hello!")
    else:
        await message.answer(f"Hello, {message.from_user.full_name}!")


async def run_bot() -> None:
    token = os.getenv("BOT_TOKEN")
    if not token:
        print("Error: BOT_TOKEN environment variable is not set.", file=sys.stderr)
        raise SystemExit(1)

    logging.basicConfig(level=logging.INFO)
    bot = Bot(token=token)
    await dp.start_polling(bot)


def main() -> None:
    asyncio.run(run_bot())
