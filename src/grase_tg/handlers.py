"""Обработчики: только разрешённые группы, только @упоминания бота с текстом вопроса."""

import logging

from aiogram import F, Router
from aiogram.enums import ChatType, MessageEntityType
from aiogram.filters import BaseFilter
from aiogram.types import Message
from openai import AsyncOpenAI

from .config import Config
from .llm import ask_llm

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4096
FORMAT_HINT = (
    "Чтобы задать вопрос, напиши его в том же сообщении после упоминания:\n"
    "«@{username} как варить борщ?»"
)
ACK_TEXT = "Принял запрос, думаю…"
ERROR_TEXT = "Не удалось получить ответ от модели. Попробуй ещё раз позже."


class AllowedChatsFilter(BaseFilter):
    """Пропускает сообщения только из групп из белого списка."""

    def __init__(self, chat_ids: frozenset[int]) -> None:
        self._chat_ids = chat_ids
        self._logged: set[int] = set()

    async def __call__(self, message: Message) -> bool:
        if message.chat.id in self._chat_ids:
            return True
        if message.chat.id not in self._logged:
            self._logged.add(message.chat.id)
            logger.info(
                "Чат «%s» (id=%s) не в белом списке — игнорирую. "
                "Добавь этот id в config.yaml, чтобы включить.",
                message.chat.title or message.chat.full_name,
                message.chat.id,
            )
        return False


def _utf16_to_py_index(text: str, utf16_pos: int) -> int:
    """Переводит смещение в UTF-16 кодовых единицах (как в Telegram entities) в индекс Python-строки."""
    if utf16_pos <= 0:
        return 0
    units = 0
    for index, char in enumerate(text):
        if units >= utf16_pos:
            return index
        units += 2 if ord(char) > 0xFFFF else 1
    return len(text)


def find_bot_mention(message: Message, bot_username: str) -> tuple[int, int] | None:
    """Ищет в тексте @упоминание именно этого бота; возвращает границы (start, end)."""
    text = message.text
    if text is None:
        return None
    mention = f"@{bot_username}".lower()
    for entity in message.entities or []:
        if entity.type != MessageEntityType.MENTION:
            continue
        start = _utf16_to_py_index(text, entity.offset)
        end = _utf16_to_py_index(text, entity.offset + entity.length)
        if text[start:end].lower() == mention:
            return start, end
    return None


def extract_question(message: Message, span: tuple[int, int]) -> str:
    """Текст вопроса: то, что после @упоминания бота; если пусто — то, что до него."""
    text = message.text or ""
    start, end = span
    after = text[end:].strip()
    if after:
        return after
    return text[:start].strip()


def split_text(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Режет длинный ответ на части, влезающие в одно сообщение Telegram."""
    return [text[i : i + limit] for i in range(0, len(text), limit)] or [""]


def build_router(config: Config, llm_client: AsyncOpenAI, bot_username: str) -> Router:
    router = Router(name="family")

    # Личные чаты и прочие типы — вообще без хендлера, молча игнорируются.
    @router.message(
        F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}),
        AllowedChatsFilter(frozenset(config.app.allowed_chat_ids)),
    )
    async def on_group_message(message: Message) -> None:
        span = find_bot_mention(message, bot_username)
        if span is None:
            return  # тэгнули не бота — молча игнорируем

        question = extract_question(message, span)
        if not question:
            await message.reply(FORMAT_HINT.format(username=bot_username))
            return

        logger.info("Вопрос из чата %s: %s", message.chat.id, question)
        ack = await message.reply(ACK_TEXT)
        try:
            answer = await ask_llm(llm_client, config, question)
        except Exception:
            logger.exception("Ошибка при обращении к LLM")
            await ack.edit_text(ERROR_TEXT)
            return

        chunks = split_text(answer)
        try:
            await ack.edit_text(chunks[0])
            for chunk in chunks[1:]:
                await ack.reply(chunk)
        except Exception:
            logger.exception("Не удалось отправить ответ в чат %s", message.chat.id)

    return router
