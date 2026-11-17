"""Handlers: only whitelisted groups. Only addressed messages get replies:
text goes to the LLM, other addressed kinds get per-type stubs,
unaddressed messages are silently ignored."""

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

# Per-type stub replies (text questions are answered by the real pipeline).
PHOTO_STUB_TEXT = "I can see the photo, but I can't analyze images yet — ask about it with text."
AUDIO_STUB_TEXT = "Voice and audio messages are not supported yet — please type your question."
STICKER_STUB_TEXT = "Nice sticker! I can't reply to stickers meaningfully yet."
# Fallback for any content type without its own handler.
OTHER_STUB_TEXT = "This message type is not supported yet — ask me with a text @mention."

STUB_TEXTS = {
    "photo": PHOTO_STUB_TEXT,
    "audio": AUDIO_STUB_TEXT,
    "sticker": STICKER_STUB_TEXT,
}


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
            return  # not addressed to the bot — silence, per the family-bot contract

        if question.kind == "text":
            if not question.text:
                # Addressed, but no question text — explain the format.
                await message.reply(FORMAT_HINT.format(username=bot_username))
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
            return

        # Addressed non-text kinds: per-type stub, falling back to "other"
        # when a kind has no dedicated handler (safety net for new types).
        stub = STUB_TEXTS.get(question.kind, OTHER_STUB_TEXT)
        logger.info("Stub reply for content type '%s' from chat %s", question.kind, message.chat.id)
        await message.reply(stub)

    return router
