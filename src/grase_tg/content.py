"""
Content extraction: routes messages by content type to per-type extractors.

Each extractor checks whether the bot was addressed in a message of its type
and returns a normalized Question. Adding a new content type = adding one
extractor function to EXTRACTORS. Only text questions are answered for now:
photo is recognized (mention in the caption) but answered with a
"not supported" hint; other types are silently ignored.
"""

from dataclasses import dataclass
from typing import Callable

from aiogram.enums import MessageEntityType
from aiogram.types import Message


@dataclass(frozen=True)
class Question:
    """A question addressed to the bot, normalized across content types."""

    kind: str                   # "text", "photo", "voice", ...
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
    """Text messages: the mention lives in the message text."""
    span = _find_mention_span(message.text, message.entities, bot_username)
    if span is None:
        return None
    return Question(kind="text", text=_question_around(message.text or "", span))


def extract_photo(message: Message, bot_username: str) -> Question | None:
    """Photos: the mention can only live in the caption.

    The image payload is not attached yet — photo questions are recognized
    (kind="photo") but the answering pipeline rejects them for now.
    """
    if message.photo is None or message.caption is None:
        return None
    span = _find_mention_span(message.caption, message.caption_entities, bot_username)
    if span is None:
        return None
    return Question(kind="photo", text=_question_around(message.caption, span), image=None)


# Registry of content extractors, tried in order. Add new content types here.
Extractor = Callable[[Message, str], Question | None]
EXTRACTORS: tuple[Extractor, ...] = (
    extract_text,
    extract_photo,
)


def extract_question(message: Message, bot_username: str) -> Question | None:
    """Try every extractor in order; return the first Question
    or None if the bot was not addressed in this message."""
    for extractor in EXTRACTORS:
        question = extractor(message, bot_username)
        if question is not None:
            return question
    return None
