"""
Content extraction: routes messages by content type to per-type extractors.

The bot reacts ONLY to messages addressed to it personally. Addressing is:
- an @mention of the bot in the message text or photo caption, or
- a reply to one of the bot's own messages.

Per content kind:
- text: the question is the text after the mention; for a reply — the whole
  message text;
- media (photos, videos, GIFs, documents): the caption question (or the
  whole caption of a reply); answered with the "cannot analyze attachments
  yet" stub;
- audio / sticker: these types cannot carry an @mention, so a reply to the
  bot is their only addressing mechanism — answered with per-type stubs.

The "other" kind is a safety net: if a future extractor returns a kind
without a dedicated handler, handlers.py answers with the "other" stub.

Unaddressed messages always produce None -> silence, never a reply.
A reply-addressed Question carries the text of the bot message the user
replied to in Question.context (one level of dialogue context for the LLM).
When a mention and a reply are both present, the mention shapes the question
text and the reply still contributes its context.
Adding a new content type = adding one extractor to EXTRACTORS and a handler
branch (stub or real) for its kind in handlers.py.
"""

from dataclasses import dataclass
from typing import Callable

from aiogram.enums import MessageEntityType
from aiogram.types import Message


@dataclass(frozen=True)
class BotIdentity:
    """Who the bot is, for addressing: its @username and Telegram user id."""

    username: str
    user_id: int


@dataclass(frozen=True)
class Question:
    """Message content, normalized across content types."""

    kind: str                   # "text", "media", "audio", "sticker", "other"
    text: str                   # question text (message text or media caption)
    image: bytes | None = None  # photo payload, attached by extract_media in the future
    # The bot message the user replied to (reply addressing only): one level
    # of dialogue context for the LLM. None when addressed via @mention or
    # when the replied-to message has no text.
    context: str | None = None


def _utf16_to_py_index(text: str, utf16_pos: int) -> int:
    """Convert a UTF-16 code unit offset (used by Telegram entities) to a Python string index."""
    if utf16_pos <= 0:
        return 0
    units = 0
    for index, char in enumerate(text):
        if units >= utf16_pos:
            return index
        # Characters outside the BMP (emoji, etc.) occupy two UTF-16 units.
        units += 2 if ord(char) > 0xFFFF else 1
    return len(text)


def _find_mention_span(
    text: str | None,
    entities: list | None,
    bot_username: str,
) -> tuple[int, int] | None:
    """Find an @mention of this particular bot in the given text; return (start, end) bounds."""
    if text is None:
        return None
    mention = f"@{bot_username}".lower()
    for entity in entities or []:
        if entity.type != MessageEntityType.MENTION:
            continue
        start = _utf16_to_py_index(text, entity.offset)
        end = _utf16_to_py_index(text, entity.offset + entity.length)
        if text[start:end].lower() == mention:
            return start, end
    return None


def _is_reply_to_bot(message: Message, bot_user_id: int) -> bool:
    """True when the message is an explicit reply to one of the bot's messages.

    In forum topics Telegram attaches an implicit "reply" link to the topic
    root message on every message; that is threading, not addressing, so it is
    excluded (an explicit reply to the bot points at the bot's message, which
    can never be the topic root — that is a service message from a human).
    """
    replied = message.reply_to_message
    if replied is None or replied.from_user is None or replied.from_user.id != bot_user_id:
        return False
    if (
        message.is_topic_message
        and message.message_thread_id is not None
        and replied.message_id == message.message_thread_id
    ):
        return False
    return True


def _reply_context(message: Message) -> str | None:
    """Text of the bot message being replied to, for dialogue context.

    Never empty when not None: a missing or blank text yields None (the bot
    only ever sends text messages, so blank means "nothing useful to say").
    """
    replied = message.reply_to_message
    text = (replied.text or "").strip() if replied is not None else ""
    return text or None


def _question_around(text: str, span: tuple[int, int]) -> str:
    """Question text: what follows the mention; if empty — what precedes it."""
    start, end = span
    after = text[end:].strip()
    if after:
        return after
    return text[:start].strip()


def extract_text(message: Message, bot: BotIdentity) -> Question | None:
    """Text messages: addressed via an @mention in the text or a reply to the bot.

    The question is what follows the mention; for a reply — the whole text.
    With both present, the mention shapes the question and the reply still
    contributes its context (the user explicitly referenced that answer).
    Returns None when the message is not addressed — plain chat must stay silent.
    """
    if message.text is None:
        return None
    is_reply = _is_reply_to_bot(message, bot.user_id)
    context = _reply_context(message) if is_reply else None
    span = _find_mention_span(message.text, message.entities, bot.username)
    if span is not None:
        return Question(kind="text", text=_question_around(message.text, span), context=context)
    if is_reply:
        return Question(kind="text", text=message.text.strip(), context=context)
    return None


def extract_media(message: Message, bot: BotIdentity) -> Question | None:
    """Photos, videos, GIFs (animations) and documents — anything with a caption.

    Addressed via an @mention in the caption or a reply to the bot (both at
    once: the mention shapes the question, the reply contributes its context).
    The payloads are not attached yet — an addressed attachment is answered
    with the media stub; the caption question is kept in Question.text.
    """
    if not (message.photo or message.video or message.animation or message.document):
        return None
    is_reply = _is_reply_to_bot(message, bot.user_id)
    context = _reply_context(message) if is_reply else None
    span = _find_mention_span(message.caption, message.caption_entities, bot.username)
    if span is not None:
        return Question(kind="media", text=_question_around(message.caption or "", span), image=None, context=context)
    if is_reply:
        return Question(kind="media", text=(message.caption or "").strip(), image=None, context=context)
    return None


def extract_audio(message: Message, bot: BotIdentity) -> Question | None:
    """Voice notes, audio files and video notes, addressed via a reply to the bot.

    These types cannot carry an @mention, so a reply is their only addressing
    mechanism. The stub handler in handlers.py answers with a "please type
    your question" notice; the caption (audio files can have one) is kept
    in Question.text.
    """
    if not (message.voice or message.audio or message.video_note):
        return None
    if not _is_reply_to_bot(message, bot.user_id):
        return None
    return Question(kind="audio", text=(message.caption or "").strip(), context=_reply_context(message))


def extract_sticker(message: Message, bot: BotIdentity) -> Question | None:
    """Stickers: addressed via a reply to the bot (a sticker has no caption to mention in)."""
    if message.sticker is None:
        return None
    if not _is_reply_to_bot(message, bot.user_id):
        return None
    return Question(kind="sticker", text="", context=_reply_context(message))


# Registry of content extractors, tried in order. Add new content types here.
Extractor = Callable[[Message, BotIdentity], Question | None]
EXTRACTORS: tuple[Extractor, ...] = (
    extract_text,
    extract_media,
    extract_audio,
    extract_sticker,
)


def extract_question(message: Message, bot: BotIdentity) -> Question | None:
    """Try every extractor in order.

    Returns the first Question (the message is addressed to the bot) or None
    for unaddressed messages — the caller must stay silent in that case.
    """
    for extractor in EXTRACTORS:
        question = extractor(message, bot)
        if question is not None:
            return question
    return None
