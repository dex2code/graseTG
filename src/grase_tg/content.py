"""
Content extraction: routes messages by content type to per-type extractors.

The bot reacts ONLY to messages addressed to it personally:
- text: @mention in the message text -> answered (or a format hint);
- photo: @mention in the caption -> photo stub handler;
- audio / sticker: extractors are declared but dormant — these types cannot
  carry an @mention and no other addressing mechanism exists yet.

The "other" kind is a safety net: if a future extractor returns a kind
without a dedicated handler, handlers.py answers with the "other" stub.

Unaddressed messages always produce None -> silence, never a reply.
Adding a new content type = adding one extractor to EXTRACTORS and a handler
branch (stub or real) for its kind in handlers.py.
"""

from dataclasses import dataclass
from typing import Callable

from aiogram.enums import MessageEntityType
from aiogram.types import Message


@dataclass(frozen=True)
class Question:
    """Message content, normalized across content types."""

    kind: str                   # "text", "photo", "audio", "sticker", "other"
    text: str                   # question text (message text or photo caption)
    image: bytes | None = None  # reserved for future photo support
    audio: bytes | None = None  # reserved for future voice support


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


def _question_around(text: str, span: tuple[int, int]) -> str:
    """Question text: what follows the mention; if empty — what precedes it."""
    start, end = span
    after = text[end:].strip()
    if after:
        return after
    return text[:start].strip()


def extract_text(message: Message, bot_username: str) -> Question | None:
    """Text messages: the mention lives in the message text.

    Returns None when the bot is not mentioned — plain chat must stay silent.
    """
    span = _find_mention_span(message.text, message.entities, bot_username)
    if span is None:
        return None
    return Question(kind="text", text=_question_around(message.text or "", span))


def extract_photo(message: Message, bot_username: str) -> Question | None:
    """Photos: addressed only via an @mention in the caption.

    The image payload is not attached yet — an addressed photo is answered
    with the photo stub; the caption question is kept in Question.text.
    """
    if message.photo is None or message.caption is None:
        return None
    span = _find_mention_span(message.caption, message.caption_entities, bot_username)
    if span is None:
        return None
    return Question(kind="photo", text=_question_around(message.caption, span), image=None)


def extract_audio(message: Message, bot_username: str) -> Question | None:
    """Voice notes, audio files and video notes.

    Dormant: audio cannot carry an @mention and no other addressing mechanism
    (e.g. reply-to-bot) exists yet, so this extractor never fires. It is
    registered to declare the handler — implement addressing detection here
    to activate it (the stub handler is already wired in handlers.py).
    """
    return None


def extract_sticker(message: Message, bot_username: str) -> Question | None:
    """Stickers: dormant for the same reason as audio (see extract_audio)."""
    return None


# Registry of content extractors, tried in order. Add new content types here.
Extractor = Callable[[Message, str], Question | None]
EXTRACTORS: tuple[Extractor, ...] = (
    extract_text,
    extract_photo,
    extract_audio,
    extract_sticker,
)


def extract_question(message: Message, bot_username: str) -> Question | None:
    """Try every extractor in order.

    Returns the first Question (the message is addressed to the bot) or None
    for unaddressed messages — the caller must stay silent in that case.
    """
    for extractor in EXTRACTORS:
        question = extractor(message, bot_username)
        if question is not None:
            return question
    return None
