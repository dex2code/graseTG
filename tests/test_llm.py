"""Tests for the LLM layer: request payload, thinking toggle, empty-answer handling."""

import pytest
from pydantic import SecretStr

from grase_tg.config import AppConfig, Config, Secrets, LlmSettings
from grase_tg.content import Question
from grase_tg.llm import EMPTY_ANSWER, _build_messages, ask_llm, make_client


def _config(llm_settings: LlmSettings | None = None, base_url: str = "http://llm.test/v1") -> Config:
    return Config(
        secrets=Secrets(bot_token=SecretStr("t"), llm_api_key=SecretStr("k"), llm_base_url=base_url),
        app=AppConfig(llm=llm_settings or LlmSettings()),
    )


# --- _build_messages ---

def test_build_messages_text_shape():
    cfg = _config()
    messages = _build_messages(cfg, Question(kind="text", text="hello"))
    assert messages == [
        {"role": "system", "content": cfg.app.system_prompt},
        {"role": "user", "content": "hello"},
    ]


def test_build_messages_rejects_non_text():
    with pytest.raises(NotImplementedError):
        _build_messages(_config(), Question(kind="photo", text="x"))


# --- ask_llm against the stub server ---

async def test_ask_llm_sends_full_payload(config_with_stub):
    cfg, stub = config_with_stub
    answer = await ask_llm(make_client(cfg), cfg, Question(kind="text", text="how to cook?"))
    assert answer == "ok"
    body = stub.requests[-1]
    assert body["stream"] is False
    assert body["model"] == cfg.app.llm.model
    assert body["messages"][-1] == {"role": "user", "content": "how to cook?"}


async def test_thinking_none_omits_field(config_with_stub):
    cfg, stub = config_with_stub
    assert cfg.app.llm.enable_thinking is None  # default from config.yaml
    await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"))
    assert "chat_template_kwargs" not in stub.requests[-1]


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
    assert answer == EMPTY_ANSWER


async def test_answer_is_stripped(config_with_stub):
    cfg, stub = config_with_stub
    stub.reply_content = "  padded answer  "
    answer = await ask_llm(make_client(cfg), cfg, Question(kind="text", text="q"))
    assert answer == "padded answer"


def test_make_client_uses_configured_base_url(config_with_stub):
    cfg, stub = config_with_stub
    client = make_client(cfg)
    assert str(client.base_url).rstrip("/").endswith("/v1")
