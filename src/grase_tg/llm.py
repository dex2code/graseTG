"""Обращение к OpenAI-совместимому backend'у."""

import logging

from openai import AsyncOpenAI

from .config import Config

logger = logging.getLogger(__name__)

EMPTY_ANSWER = "Модель вернула пустой ответ."


def make_client(config: Config) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=config.secrets.llm_api_key,
        base_url=config.secrets.llm_base_url,
        timeout=config.app.llm.timeout_seconds,
    )


async def ask_llm(client: AsyncOpenAI, config: Config, question: str) -> str:
    """Задаёт вопрос без истории переписки и возвращает текст ответа."""
    response = await client.chat.completions.create(
        model=config.app.llm.model,
        temperature=config.app.llm.temperature,
        max_tokens=config.app.llm.max_tokens,
        messages=[
            {"role": "system", "content": config.app.system_prompt},
            {"role": "user", "content": question},
        ],
    )
    answer = (response.choices[0].message.content or "").strip()
    return answer or EMPTY_ANSWER
