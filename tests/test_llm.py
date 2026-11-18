"""Tests for the LLM layer: request payload, thinking toggle, tool loop, empty-answer handling."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import SecretStr

from conftest import StubToolRegistry

from grase_tg.config import AppConfig, Config, Secrets, LlmSettings
from grase_tg.content import Question
from grase_tg.llm import _build_messages, _date_line, ask_llm, make_client


def _config(llm_settings: LlmSettings | None = None, base_url: str = "http://llm.test/v1") -> Config:
    return Config(
        secrets=Secrets(bot_token=SecretStr("t"), llm_api_key=SecretStr("k"), llm_base_url=base_url),
        app=AppConfig(llm=llm_settings or LlmSettings()),
    )


# --- _date_line ---

def test_date_line_format_with_offset():
    now = datetime(2026, 11, 18, 12, 0, tzinfo=timezone(timedelta(hours=3)))
    assert _date_line(now) == "Current date: 2026-11-18 (Wednesday), UTC+03:00."


def test_date_line_format_utc():
    now = datetime(2026, 1, 5, 23, 30, tzinfo=timezone.utc)
    assert _date_line(now) == "Current date: 2026-01-05 (Monday), UTC+00:00."


# --- _build_messages ---

def test_build_messages_prepends_system_prompt_with_current_date():
    cfg = _config()
    messages = _build_messages(cfg, Question(kind="text", text="hello"))
    assert messages[0]["role"] == "system"
    assert isinstance(messages[0]["content"], str)
    assert messages[0]["content"].startswith(cfg.app.system_prompt)
    assert "Current date:" in messages[0]["content"]  # the model has no clock
    assert messages[1] == {"role": "user", "content": "hello"}


def test_build_messages_includes_reply_context_as_assistant():
    cfg = _config()
    messages = _build_messages(
        cfg,
        Question(kind="text", text="and what about chicken?", context="Borscht takes 2 hours."),
    )
    assert messages[1] == {"role": "assistant", "content": "Borscht takes 2 hours."}
    assert messages[2] == {"role": "user", "content": "and what about chicken?"}


def test_build_messages_rejects_non_text():
    with pytest.raises(NotImplementedError):
        _build_messages(_config(), Question(kind="media", text="x"))


# --- quoted human message (reply to a participant + @mention) ---


def test_build_messages_without_quote_is_the_bare_question():
    # Byte-for-byte: without a quote the user message is exactly the question
    # text — the pre-quote request shape.
    messages = _build_messages(_config(), Question(kind="text", text="plain question"))
    assert messages[1:] == [{"role": "user", "content": "plain question"}]


def test_build_messages_renders_quote_inside_the_user_turn():
    # The quote is NOT a dialogue turn of its own: it is marked with "> "
    # inside the single user message, so the model cannot attribute it to
    # the asker and the turn structure stays valid on every backend.
    messages = _build_messages(
        _config(),
        Question(kind="text", text="что он имел в виду?", quote="борщ\nварить 2 часа"),
    )
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1:] == [
        {
            "role": "user",
            "content": "Replying to another chat participant's message:\n> борщ\n> варить 2 часа\n\nчто он имел в виду?",
        }
    ]


def test_build_messages_quote_and_context_coexist():
    # The extractors set at most one of the two (a reply points at either the
    # bot or a human), but the builder treats the fields independently: an
    # assistant context and a quote render in their own places regardless.
    messages = _build_messages(
        _config(),
        Question(kind="text", text="а теперь?", context="prev answer", quote="цитата"),
    )
    assert messages[1:] == [
        {"role": "assistant", "content": "prev answer"},
        {"role": "user", "content": "Replying to another chat participant's message:\n> цитата\n\nа теперь?"},
    ]


# --- ask_llm against the stub server ---

async def test_ask_llm_sends_full_payload(config_with_stub):
    cfg, stub = config_with_stub
    answer = await ask_llm(make_client(cfg), cfg, Question(kind="text", text="how to cook?"))
    assert answer == "ok"
    body = stub.requests[-1]
    assert body["stream"] is False
    assert body["model"] == cfg.app.llm.model
    assert body["messages"][-1] == {"role": "user", "content": "how to cook?"}


async def test_ask_llm_sends_reply_context(config_with_stub):
    cfg, stub = config_with_stub
    await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q2", context="prev answer"))
    body_messages = stub.requests[-1]["messages"]
    assert body_messages[0]["role"] == "system"
    assert body_messages[0]["content"].startswith(cfg.app.system_prompt)
    assert body_messages[1:] == [
        {"role": "assistant", "content": "prev answer"},
        {"role": "user", "content": "q2"},
    ]


# --- empty / truncated answers: diagnostics in the log, no content logged ---


async def test_empty_answer_logs_diagnosis_but_never_content(config_with_stub, caplog):
    # The prod failure report: 200 OK, 67s of generation, empty content —
    # a reasoning model spent the whole max_tokens budget on thinking
    # (reasoning moved to reasoning_content, finish_reason=length). The log
    # must say WHY, but must not contain the question text (privacy rule).
    import logging

    cfg, stub = config_with_stub
    stub.reply_content = ""
    with caplog.at_level(logging.WARNING, logger="grase_tg.llm"):
        answer = await ask_llm(make_client(cfg), cfg, Question(kind="text", text="TOPSECRET-QUESTION"))
    assert answer == cfg.app.messages.empty_answer
    assert "finish_reason=stop" in caplog.text
    assert "completion_tokens=1" in caplog.text
    assert "reasoning_content=absent" in caplog.text
    assert "TOPSECRET-QUESTION" not in caplog.text


async def test_reasoning_eaten_budget_is_visible_in_the_log(config_with_stub, caplog):
    # The exact mechanism behind the prod report: reasoning_content present,
    # content empty, finish_reason=length -> two warnings naming the knob
    # (max_tokens) and the fact that reasoning was there.
    import logging

    cfg, stub = config_with_stub
    stub.script = [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "long thinking about the tool result...",
            "_finish_reason": "length",
        }
    ]
    with caplog.at_level(logging.WARNING, logger="grase_tg.llm"):
        answer = await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"))
    assert answer == cfg.app.messages.empty_answer
    assert "finish_reason=length" in caplog.text
    assert "reasoning_content=present" in caplog.text
    assert "llm.max_tokens" in caplog.text


async def test_truncated_answer_with_content_warns(config_with_stub, caplog):
    # finish_reason=length WITH visible content: the answer is returned
    # (half an answer beats a failure notice), but the truncation is logged.
    import logging

    cfg, stub = config_with_stub
    stub.reply_content = "partial answer"
    stub.script = [{"role": "assistant", "content": "partial answer", "_finish_reason": "length"}]
    with caplog.at_level(logging.WARNING, logger="grase_tg.llm"):
        answer = await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"))
    assert answer == "partial answer"
    assert "hit llm.max_tokens" in caplog.text


# --- tools field presence ---


async def test_no_registry_omits_tools_field(config_with_stub):
    # Without a registry the request must be byte-for-byte the pre-tools
    # behavior: no "tools" key at all (an empty tools:[] confuses backends).
    cfg, stub = config_with_stub
    await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"))
    assert "tools" not in stub.requests[-1]


async def test_empty_registry_schema_omits_tools_field(config_with_stub):
    # A registry with zero tools (e.g. the MCP server exposed nothing after
    # filtering) behaves exactly like no registry.
    cfg, stub = config_with_stub
    await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"), StubToolRegistry())
    assert "tools" not in stub.requests[-1]


# --- the tool loop ---


def _tool_call_message(call_id: str = "call_1", name: str = "get_weather", arguments: str = '{"city": "Moscow"}'):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


async def test_tool_loop_roundtrip(config_with_stub):
    # 1st request -> tool_calls; the registry executes; the result travels
    # back as role="tool"; the 2nd request produces the final answer.
    cfg, stub = config_with_stub
    stub.script = [_tool_call_message(), {"role": "assistant", "content": "sunny and warm"}]
    registry = StubToolRegistry(schema=[{"type": "function", "function": {"name": "get_weather"}}], results=["sunny, +20C"])

    answer = await ask_llm(make_client(cfg), cfg, Question(kind="text", text="weather?"), registry)

    assert answer == "sunny and warm"
    assert registry.calls == [("get_weather", '{"city": "Moscow"}')]
    assert "tools" in stub.requests[0]
    second = stub.requests[1]["messages"]
    assert second[-2] == {  # the assistant message reconstructed as a plain dict
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'},
            }
        ],
    }
    assert second[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny, +20C"}


async def test_parallel_tool_calls_all_fed_back(config_with_stub):
    cfg, stub = config_with_stub
    stub.script = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
                {"id": "call_b", "type": "function", "function": {"name": "get_time", "arguments": "{}"}},
            ],
        },
        {"role": "assistant", "content": "done"},
    ]
    registry = StubToolRegistry(schema=[{"type": "function", "function": {"name": "x"}}], results=["r1", "r2"])

    answer = await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"), registry)

    assert answer == "done"
    assert registry.calls == [("get_weather", "{}"), ("get_time", "{}")]
    tool_messages = stub.requests[1]["messages"][-2:]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_a", "call_b"]
    assert [m["content"] for m in tool_messages] == ["r1", "r2"]


async def test_rounds_exhausted_returns_tool_failure(config_with_stub):
    # tool_rounds=1: the first round executes, but when the model asks for
    # tools again the budget is spent -> the honest failure text.
    cfg, stub = config_with_stub
    cfg.app.llm.tool_rounds = 1
    stub.script = [
        _tool_call_message(call_id="call_1"),
        _tool_call_message(call_id="call_2"),
    ]
    registry = StubToolRegistry(schema=[{"type": "function", "function": {"name": "get_weather"}}], results=["r1"])

    answer = await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"), registry)

    assert answer == cfg.app.messages.tool_failure
    assert registry.calls == [("get_weather", '{"city": "Moscow"}')]  # the 2nd call never happened


async def test_default_rounds_budget_is_two(config_with_stub):
    # The default budget (tool_rounds=2) allows at most two tool rounds:
    # three consecutive tool-call answers still stop at the failure notice.
    cfg, stub = config_with_stub
    stub.script = [
        _tool_call_message(call_id="call_1"),
        _tool_call_message(call_id="call_2"),
        _tool_call_message(call_id="call_3"),
    ]
    registry = StubToolRegistry(schema=[{"type": "function", "function": {"name": "x"}}], results=["r1", "r2"])

    answer = await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"), registry)

    assert answer == cfg.app.messages.tool_failure
    assert len(registry.calls) == 2


async def test_thinking_none_omits_field(config_with_stub):
    cfg, stub = config_with_stub
    assert cfg.app.llm.enable_thinking is None  # default from config.yaml
    await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"))
    assert "chat_template_kwargs" not in stub.requests[-1]


# --- sampling overrides ---


async def test_sampling_defaults_omitted(config_with_stub):
    # Unset overrides must not appear in the request at all: the backend
    # applies its own defaults (temperature is unconditional and stays).
    cfg, stub = config_with_stub
    await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"))
    body = stub.requests[-1]
    assert "top_p" not in body
    assert "presence_penalty" not in body
    assert body["temperature"] == cfg.app.llm.temperature


async def test_sampling_overrides_sent(config_with_stub):
    cfg, stub = config_with_stub
    cfg.app.llm.top_p = 0.9
    cfg.app.llm.presence_penalty = 0.5
    await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"))
    body = stub.requests[-1]
    assert body["top_p"] == 0.9
    assert body["presence_penalty"] == 0.5


async def test_thinking_true_and_false_sent_explicitly(config_with_stub):
    cfg, stub = config_with_stub
    for value in (True, False):
        cfg.app.llm.enable_thinking = value
        await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"))
        assert stub.requests[-1]["chat_template_kwargs"] == {"enable_thinking": value}


async def test_empty_model_answer_replaced_with_notice(config_with_stub):
    cfg, stub = config_with_stub
    stub.reply_content = ""
    answer = await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"))
    assert answer == cfg.app.messages.empty_answer


async def test_answer_is_stripped(config_with_stub):
    cfg, stub = config_with_stub
    stub.reply_content = "  padded answer  "
    answer = await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"))
    assert answer == "padded answer"


def test_make_client_uses_configured_base_url(config_with_stub):
    cfg, stub = config_with_stub
    client = make_client(cfg)
    assert str(client.base_url).rstrip("/").endswith("/v1")
