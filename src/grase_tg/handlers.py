"""Handlers: only whitelisted groups, only messages addressing the bot."""

import logging

from aiogram import F, Router
from aiogram.enums import ChatType
from aiogram.filters import BaseFilter
from aiogram.types import Message
from openai import AsyncOpenAI

from .config import Config
from .content import extract_question
from .llm import ask_llm

logger = logging.getLogger(__name__)

# Telegram hard limit for a single text message.
MAX_MESSAGE_LENGTH = 4096

# User-facing texts.
FORMAT_HINT = (
    "To ask a question, write it in the same message after the mention:\n"
    "\"@{username} how do I cook borscht?\""
)
ACK_TEXT = "Got it, thinking…"
ERROR_TEXT = "Failed to get a response from the model. Please try again later."
NOT_SUPPORTED_TEXT = "This content type is not supported yet — ask with plain text for now."


class AllowedChatsFilter(BaseFilter):
    """Passes messages coming only from whitelisted group chats."""

    def __init__(self, chat_ids: frozenset[int]) -> None:
        self._chat_ids = chat_ids
        # Remember already-logged chats to avoid log spam.
        self._logged: set[int] = set()

    async def __call__(self, message: Message) -> bool:
        if message.chat.id in self._chat_ids:
            return True
        # Log unknown chats once: this is how the user discovers ids for the whitelist.
        if message.chat.id not in self._logged:
            self._logged.add(message.chat.id)
            logger.info(
                "Chat '%s' (id=%s) is not whitelisted — ignoring. "
                "Add this id to config.yaml to enable it.",
                message.chat.title or message.chat.full_name,
                message.chat.id,
            )
        return False


def split_text(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a long answer into chunks that fit into one Telegram message."""
    return [text[i : i + limit] for i in range(0, len(text), limit)] or [""]


def build_router(config: Config, llm_client: AsyncOpenAI, bot_username: str) -> Router:
    router = Router(name="family")

    # Private chats and other chat types have no handler at all — silently ignored.
    @router.message(
        F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}),
        AllowedChatsFilter(frozenset(config.app.allowed_chat_ids)),
    )
    async def on_group_message(message: Message) -> None:
        question = extract_question(message, bot_username)
        if question is None:
            return  # the bot was not addressed — silently ignore

        if not question.text:
            # Addressed, but no question text — explain the format.
            await message.reply(FORMAT_HINT.format(username=bot_username))
            return

        if question.kind != "text":
            # Recognized content type without an answering implementation yet.
            logger.info("Unsupported content type '%s' from chat %s", question.kind, message.chat.id)
            await message.reply(NOT_SUPPORTED_TEXT)
            return

        logger.info("Question from chat %s: %s", message.chat.id, question.text)

        # Acknowledge immediately, then edit this message with the answer.
        ack = await message.reply(ACK_TEXT)
        try:
            answer = await ask_llm(llm_client, config, question)
        except Exception:
            logger.exception("LLM request failed")
            await ack.edit_text(ERROR_TEXT)
            return

        # First chunk replaces the ack; the rest follow as replies to it.
        chunks = split_text(answer)
        try:
            await ack.edit_text(chunks[0])
            for chunk in chunks[1:]:
                await ack.reply(chunk)
        except Exception:
            logger.exception("Failed to deliver the answer to chat %s", message.chat.id)

    return router
