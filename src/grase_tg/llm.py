"""Calls to the OpenAI-compatible backend."""

from openai import AsyncOpenAI

from .config import Config

EMPTY_ANSWER = "The model returned an empty response."


def make_client(config: Config) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=config.secrets.llm_api_key.get_secret_value(),
        base_url=config.secrets.llm_base_url,
        timeout=config.app.llm.timeout_seconds,
    )


async def ask_llm(client: AsyncOpenAI, config: Config, question: str) -> str:
    """Ask a single question without chat history and return the answer text."""
    response = await client.chat.completions.create(
        model=config.app.llm.model,
        temperature=config.app.llm.temperature,
        max_tokens=config.app.llm.max_tokens,
        # Explicit: we always want one complete response, not an SSE stream.
        stream=False,
        messages=[
            {"role": "system", "content": config.app.system_prompt},
            {"role": "user", "content": question},
        ],
        # Provider-specific extension: model "thinking" toggle
        # (vLLM/SGLang, Qwen3-style models). Ignored by plain OpenAI backends.
        extra_body={"chat_template_kwargs": {"enable_thinking": config.app.llm.enable_thinking}},
    )
    answer = (response.choices[0].message.content or "").strip()
    return answer or EMPTY_ANSWER
