"""Handlers: only whitelisted groups. Only addressed messages (an @mention or
a reply to one of the bot's messages) get replies: text goes to the LLM,
other addressed kinds get per-type stubs, unaddressed messages are silently
ignored."""

import logging
from dataclasses import replace

from aiogram import F, Router
from aiogram.enums import ChatType
from aiogram.filters import BaseFilter
from aiogram.types import Message
from openai import AsyncOpenAI

from .config import Config, MessagesSettings
from .content import BotIdentity, Question, QuestionKind, extract_question
from .llm import ask_llm
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

# Telegram hard limit for a single text message, measured in UTF-16 code
# units: characters outside the BMP (most emoji) count as two.
MAX_MESSAGE_LENGTH = 4096


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
    """Split an answer into chunks that fit into one Telegram message.

    Telegram measures message length in UTF-16 code units, so the limit is
    enforced in those units: a BMP character costs one unit, an emoji outside
    the BMP costs two. Chunk boundaries never split a code point; splitting
    inside a grapheme cluster (ZWJ emoji, combining marks) is still possible
    and accepted for LLM answers.

    Precondition: text is non-empty (ask_llm guarantees that); an empty input
    honestly gives an empty list.
    """
    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        units = 0
        end = start
        while end < text_len:
            char_units = 2 if ord(text[end]) > 0xFFFF else 1
            if units + char_units > limit:
                break
            units += char_units
            end += 1
        if end == start:
            # The limit is smaller than one character — impossible with the
            # Telegram limit, but never produce empty chunks / loop forever.
            end = start + 1
        chunks.append(text[start:end])
        start = end
    return chunks


async def send_reply(message: Message, text: str) -> Message | None:
    """Reply to the user; on failure log clearly and return None instead of raising.

    A failed send must not crash the handler: the exception would escape to
    aiogram as a raw traceback, and the user would get nothing either way.
    """
    try:
        return await message.reply(text)
    except Exception:
        logger.exception("Failed to send a message to chat %s", message.chat.id)
        return None


async def edit_ack(ack: Message, text: str, fallback: Message | None = None) -> None:
    """Edit the ack message; on failure log clearly instead of raising.

    With `fallback` set, the text is re-sent as a plain reply to that message:
    a failed edit must not leave the user staring at the stale ack.
    """
    try:
        await ack.edit_text(text)
    except Exception:
        logger.exception("Failed to edit the ack message")
        if fallback is not None:
            await send_reply(fallback, text)


def stub_texts(messages: MessagesSettings) -> dict[QuestionKind, str]:
    """Per-kind stub replies, taken from the configured texts.

    Text questions are answered by the real pipeline; stubs cover the rest.
    A typo in a key would be flagged by the type checker, since the keys are
    the QuestionKind literal; a kind without an entry lands in other_stub.
    """
    return {
        "media": messages.media_stub,
        "audio": messages.audio_stub,
        "sticker": messages.sticker_stub,
    }


def service_texts(messages: MessagesSettings, bot_username: str) -> frozenset[str]:
    """Everything the bot may send that is NOT a real answer.

    The ack, error and stub notices, the empty-answer and tool-failure
    fallbacks and the rendered format hint. Replying to one of them must not
    feed the text back to the model as fake "assistant said" context —
    strip_service_context filters on this set. Built from the configured
    values so translated texts are covered.
    """
    return frozenset(
        {
            messages.ack,
            messages.error,
            messages.empty_answer,
            messages.tool_failure,
            messages.media_stub,
            messages.audio_stub,
            messages.sticker_stub,
            messages.other_stub,
            messages.format_hint.format(username=bot_username),
        }
    )


def strip_service_context(question: Question, non_answers: frozenset[str]) -> Question:
    """Drop Question.context when it is one of the bot's service texts.

    The ack, error and stub notices are not answers: presenting them to the
    model as its own prior reply would be noise, not context.
    """
    if question.context is None:
        return question
    if question.context in non_answers:
        return replace(question, context=None)
    return question


def is_reply_to_pending_ack(question: Question, ack_text: str) -> bool:
    """True when the user replied to a pending ack ("Got it, thinking…").

    The question that ack acknowledges is still being generated: any follow-up
    to the ack itself is skipped by design — wait for the answer, then reply
    to it. Once the ack is edited into the answer, replies to that message
    stop matching and become normal follow-ups (with the answer as context).
    """
    return question.context == ack_text


def build_router(
    config: Config,
    llm_client: AsyncOpenAI,
    bot: BotIdentity,
    tools: ToolRegistry | None = None,
) -> Router:
    """Build the family router. `tools` is None (or empty) when no MCP server
    is configured — the text pipeline then runs exactly as before, in one
    stateless request (docs/mcp-tools-spec.md §4)."""
    router = Router(name="family")
    messages = config.app.messages
    stubs = stub_texts(messages)
    non_answers = service_texts(messages, bot.username)

    # Private chats and other chat types have no handler at all — silently ignored.
    @router.message(
        F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}),
        AllowedChatsFilter(frozenset(config.app.allowed_chat_ids)),
    )
    async def on_group_message(message: Message) -> None:
        question = extract_question(message, bot)
        if question is None:
            return  # not addressed to the bot — silence, per the family-bot contract

        if is_reply_to_pending_ack(question, messages.ack):
            # The question this ack acknowledges is still being generated:
            # additions sent as replies to the ack are skipped by design —
            # wait for the answer, then reply to it.
            logger.info("Skipping a reply to the pending ack in chat %s", message.chat.id)
            return

        if question.kind == "text":
            if not question.text:
                # Addressed, but no question text — explain the format.
                await send_reply(message, messages.format_hint.format(username=bot.username))
                return
            # Privacy: question texts never go to the logs — only where it came
            # from and how long it was.
            logger.info("Question from chat %s (%d chars)", message.chat.id, len(question.text))

            # Acknowledge immediately, then edit this message with the answer.
            ack = await send_reply(message, messages.ack)
            if ack is None:
                return  # could not even ack — the log holds the reason
            try:
                answer = await ask_llm(
                    llm_client,
                    config,
                    strip_service_context(question, non_answers),
                    tools,
                )
            except Exception:
                logger.exception("LLM request failed")
                await edit_ack(ack, messages.error, message)
                return

            # First chunk replaces the ack; the rest follow as replies to it.
            chunks = split_text(answer)
            delivered = 0
            try:
                await ack.edit_text(chunks[0])
                delivered = 1
                for chunk in chunks[1:]:
                    await ack.reply(chunk)
                    delivered += 1
            except Exception:
                # Re-send whatever was not delivered as plain replies: a failed
                # edit/reply must not leave the stale "thinking" ack behind.
                logger.exception(
                    "Failed to deliver the answer to chat %s (%d of %d chunks sent)",
                    message.chat.id,
                    delivered,
                    len(chunks),
                )
                for chunk in chunks[delivered:]:
                    await send_reply(message, chunk)
            return

        # Addressed non-text kinds: per-type stub, falling back to "other"
        # when a kind has no dedicated handler (safety net for new types).
        stub = stubs.get(question.kind, messages.other_stub)
        logger.info("Stub reply for content type '%s' from chat %s", question.kind, message.chat.id)
        await send_reply(message, stub)

    return router
