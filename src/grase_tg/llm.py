"""Calls to the OpenAI-compatible backend."""

import logging
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
)

from .config import Config
from .content import Question
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

# Fixed English weekday names: the system locale must not change what the
# model is told (strftime with %A would be locale-dependent).
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def make_client(config: Config) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=config.secrets.llm_api_key.get_secret_value(),
        base_url=config.secrets.llm_base_url,
        timeout=config.app.llm.timeout_seconds,
    )


def _date_line(now: datetime | None = None) -> str:
    """The model has no clock: "tomorrow" must resolve to a real date.

    Injected into every request (docs/mcp-tools-spec.md §4), independent of
    tools: it alone fixes "which day is tomorrow" for knowledge-only models.
    """
    now = now or datetime.now().astimezone()
    offset = now.strftime("%z") or "+0000"  # "+0300"; naive input would be empty
    return f"Current date: {now:%Y-%m-%d} ({_WEEKDAYS[now.weekday()]}), UTC{offset[:3]}:{offset[3:]}."


def _user_content(question: Question) -> str:
    """The user message sent to the model: the question, optionally prefixed
    with the quoted message of another participant.

    The quote travels INSIDE the single user turn, clearly marked with "> "
    (chat-quoting convention the models know): one turn keeps the dialogue
    structure on every OpenAI-compatible backend, and the marker keeps the
    model from attributing the quote to the asker. Without a quote the
    content is exactly the question text — byte-for-byte the pre-quote
    request.
    """
    if question.quote is None:
        return question.text
    quoted = "\n".join(f"> {line}" for line in question.quote.splitlines())
    return f"Replying to another chat participant's message:\n{quoted}\n\n{question.text}"


def _build_messages(config: Config, question: Question) -> list[ChatCompletionMessageParam]:
    """Build the chat/completions payload.

    The payload is stateless (no chat history). The only context is
    Question.context — the bot answer the user replied to — sent as one
    assistant message, so follow-up questions ("and what about chicken?")
    keep their referent. A multi-chunk answer contributes only the chunk
    the user actually replied to.

    The current date is appended to the system prompt: the model has no
    clock, and both "tomorrow" in a question and any date-aware tool
    arguments depend on it.
    """
    if question.kind != "text":
        raise NotImplementedError(f"No request builder for content kind '{question.kind}'")
    # rstrip: a yaml block scalar (system_prompt: |) ends with a newline —
    # without it the date line would float two blank lines below the prompt.
    system_prompt = config.app.system_prompt.rstrip()
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": f"{system_prompt}\n\n{_date_line()}"},
    ]
    if question.context is not None:
        messages.append({"role": "assistant", "content": question.context})
    messages.append({"role": "user", "content": _user_content(question)})
    return messages


def _as_assistant_message(message: ChatCompletionMessage) -> ChatCompletionMessageParam:
    """Round-trip the SDK's assistant message into a plain dict for the next
    request of the tool loop.

    llama-server is strict about tool_call_id/type — an explicit dict avoids
    any surprises in how the SDK object serializes back into the payload.
    """
    tool_calls: list[ChatCompletionMessageToolCallParam] = [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }
        for call in message.tool_calls or []
        # Only function calls are re-sent: the SDK's union also admits
        # "custom" tool calls from other dialects, which llama-server does
        # not produce and the payload format cannot carry.
        if isinstance(call, ChatCompletionMessageFunctionToolCall)
    ]
    return {"role": "assistant", "content": message.content, "tool_calls": tool_calls}


async def ask_llm(
    client: AsyncOpenAI,
    config: Config,
    question: Question,
    tools: ToolRegistry | None = None,
) -> str:
    """Ask a single question without chat history (at most one level of
    replied-to context) and return the answer text.

    With a non-empty registry the call becomes a tool loop
    (docs/mcp-tools-spec.md §4): the model may request tool executions up to
    `llm.tool_rounds` times; every request goes through the registry and the
    result travels back as a role="tool" message. Tool execution never raises
    — errors are text for the model to react to. An empty registry (or
    tools=None) omits the `tools` field entirely: byte-for-byte the
    single-request behavior of earlier versions.
    """
    # Sampling overrides, None = the field is not sent: the backend applies
    # its own defaults. Both are standard OpenAI fields -> typed kwargs.
    sampling: dict[str, Any] = {
        key: value
        for key, value in (
            ("top_p", config.app.llm.top_p),
            ("presence_penalty", config.app.llm.presence_penalty),
        )
        if value is not None
    }
    # Model "thinking" toggle (vLLM/SGLang, Qwen3-style). Sent only when
    # enable_thinking is true/false; None means the field is omitted entirely.
    extra_body: dict[str, Any] | None = None
    if config.app.llm.enable_thinking is not None:
        extra_body = {"chat_template_kwargs": {"enable_thinking": config.app.llm.enable_thinking}}

    messages = _build_messages(config, question)
    # The field is either present with a non-empty schema or absent entirely —
    # an empty tools:[] confuses some OpenAI-compatible backends.
    schema = tools.openai_schema() if tools is not None else []
    rounds_left = config.app.llm.tool_rounds
    while True:
        kwargs: dict[str, Any] = {"tools": schema} if schema else {}
        response = await client.chat.completions.create(
            model=config.app.llm.model,
            temperature=config.app.llm.temperature,
            max_tokens=config.app.llm.max_tokens,
            # Explicit: we always want one complete response, not an SSE stream.
            stream=False,
            messages=messages,
            extra_body=extra_body,
            **sampling,
            **kwargs,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            answer = (message.content or "").strip()
            if not answer:
                # 200 OK with empty content is almost always a reasoning model
                # that spent the whole max_tokens budget on thinking: llama.cpp
                # counts reasoning tokens toward max_tokens and moves the text
                # into reasoning_content, leaving content empty with
                # finish_reason="length". Log the diagnosis — counts and
                # booleans only, never any content (privacy rule).
                extra = getattr(message, "model_extra", None) or {}
                logger.warning(
                    "Empty model answer: finish_reason=%s, completion_tokens=%s, prompt_tokens=%s, reasoning_content=%s",
                    response.choices[0].finish_reason,
                    getattr(response.usage, "completion_tokens", None),
                    getattr(response.usage, "prompt_tokens", None),
                    "present" if extra.get("reasoning_content") else "absent",
                )
            if response.choices[0].finish_reason == "length":
                # A truncated (or fully thinking-consumed) generation: the user
                # gets text that just stops — the log must say why.
                logger.warning(
                    "Generation hit llm.max_tokens=%s — reasoning tokens count toward the limit; "
                    "increase max_tokens or disable thinking (llm.enable_thinking)",
                    config.app.llm.max_tokens,
                )
            return answer or config.app.messages.empty_answer
        if rounds_left == 0:
            # The budget is spent while the model keeps asking for tools:
            # an honest failure beats a silent loop.
            return config.app.messages.tool_failure
        if tools is None:
            # Unreachable — tool_calls imply a non-empty schema, hence a
            # registry — but degrade to the same honest failure if it ever
            # happens, instead of a TypeError from execute() on None.
            return config.app.messages.tool_failure
        rounds_left -= 1
        messages.append(_as_assistant_message(message))
        for call in message.tool_calls:
            if not isinstance(call, ChatCompletionMessageFunctionToolCall):
                # The OpenAI dialect llama-server speaks carries function
                # calls only; a "custom" tool call would mean a different
                # dialect — tell the model instead of crashing.
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": "Error: unsupported custom tool call."}
                )
                continue
            result = await tools.execute(call.function.name, call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
