"""Calls to the OpenAI-compatible backend."""

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .config import Config
from .content import Question

EMPTY_ANSWER = "The model returned an empty response."


def make_client(config: Config) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=config.secrets.llm_api_key.get_secret_value(),
        base_url=config.secrets.llm_base_url,
        timeout=config.app.llm.timeout_seconds,
    )


def _build_messages(config: Config, question: Question) -> list[ChatCompletionMessageParam]:
    """Build the chat/completions payload.

    Extensible by question.kind: only plain text is implemented for now;
    multimodal kinds (photo, voice) attach their payloads here later.
    """
    if question.kind != "text":
        raise NotImplementedError(f"No request builder for content kind '{question.kind}'")
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": config.app.system_prompt},
        {"role": "user", "content": question.text},
    ]
    return messages


async def ask_llm(client: AsyncOpenAI, config: Config, question: Question) -> str:
    """Ask a single question without chat history and return the answer text."""
    # Model "thinking" toggle (vLLM/SGLang, Qwen3-style). Sent only when
    # enable_thinking is true/false; None means the field is omitted entirely.
    extra_body = None
    if config.app.llm.enable_thinking is not None:
        extra_body = {"chat_template_kwargs": {"enable_thinking": config.app.llm.enable_thinking}}
    response = await client.chat.completions.create(
        model=config.app.llm.model,
        temperature=config.app.llm.temperature,
        max_tokens=config.app.llm.max_tokens,
        # Explicit: we always want one complete response, not an SSE stream.
        stream=False,
        messages=_build_messages(config, question),
        extra_body=extra_body,
    )
    answer = (response.choices[0].message.content or "").strip()
    return answer or EMPTY_ANSWER
