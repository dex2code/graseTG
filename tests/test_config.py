"""Tests for configuration loading: .env secrets, config.yaml, defaults."""

import textwrap

import pytest
from pydantic import SecretStr, ValidationError

from grase_tg.config import load_config


ENV = {
    "BOT_TOKEN": "test-token-123",
    "LLM_API_KEY": "sk-test-key",
    "LLM_BASE_URL": "http://llm.test/v1",
}


@pytest.fixture
def env_isolated(monkeypatch):
    """Hide real environment variables so tests are hermetic."""
    for name in ("BOT_TOKEN", "LLM_API_KEY", "LLM_BASE_URL", "TELEGRAM_PROXY_URL"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _write(tmp_path, yaml_text: str | None = None, env_text: str | None = None):
    if yaml_text is not None:
        (tmp_path / "config.yaml").write_text(textwrap.dedent(yaml_text), encoding="utf-8")
    if env_text is not None:
        (tmp_path / ".env").write_text(textwrap.dedent(env_text), encoding="utf-8")


def test_load_reads_yaml_and_env(tmp_path, env_isolated):
    _write(
        tmp_path,
        yaml_text="""
        allowed_chat_ids: [-1001, -1002]
        system_prompt: Be brief.
        llm:
          model: my-model
          temperature: 0.2
          max_tokens: 512
          timeout_seconds: 30
          enable_thinking: true
        """,
        env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
        """,
    )
    cfg = load_config(tmp_path)
    assert cfg.app.allowed_chat_ids == [-1001, -1002]
    assert cfg.app.system_prompt == "Be brief."
    assert cfg.app.llm.model == "my-model"
    assert cfg.app.llm.temperature == 0.2
    assert cfg.app.llm.max_tokens == 512
    assert cfg.app.llm.timeout_seconds == 30
    assert cfg.app.llm.enable_thinking is True
    assert cfg.secrets.bot_token.get_secret_value() == ENV["BOT_TOKEN"]
    assert cfg.secrets.llm_api_key.get_secret_value() == ENV["LLM_API_KEY"]
    assert cfg.secrets.llm_base_url == ENV["LLM_BASE_URL"]


def test_missing_yaml_gives_defaults(tmp_path, env_isolated):
    _write(tmp_path, env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
    """)
    cfg = load_config(tmp_path)
    assert cfg.app.allowed_chat_ids == []
    assert cfg.app.llm.enable_thinking is None  # default: field is never sent
    assert cfg.secrets.telegram_proxy_url == ""


def test_yaml_null_thinking_maps_to_none(tmp_path, env_isolated):
    _write(
        tmp_path,
        yaml_text="""
        llm:
          enable_thinking: null
        """,
        env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
        """,
    )
    assert load_config(tmp_path).app.llm.enable_thinking is None


def test_proxy_url_read_from_env(tmp_path, env_isolated):
    env_isolated.setenv("TELEGRAM_PROXY_URL", "http://proxy.local:8080")
    _write(tmp_path, env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
    """)
    assert load_config(tmp_path).secrets.telegram_proxy_url == "http://proxy.local:8080"


def test_missing_secrets_fail_with_field_names(tmp_path, env_isolated):
    _write(tmp_path, yaml_text="allowed_chat_ids: []")  # no .env at all
    with pytest.raises(ValidationError) as exc:
        load_config(tmp_path)
    missing = str(exc.value)
    assert "bot_token" in missing and "llm_api_key" in missing and "llm_base_url" in missing


def test_secrets_are_masked_in_repr(tmp_path, env_isolated):
    _write(tmp_path, env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
    """)
    cfg = load_config(tmp_path)
    assert isinstance(cfg.secrets.bot_token, SecretStr)
    assert ENV["BOT_TOKEN"] not in repr(cfg.secrets)
    assert ENV["LLM_API_KEY"] not in str(cfg.secrets)


def test_unknown_yaml_keys_are_ignored(tmp_path, env_isolated):
    _write(
        tmp_path,
        yaml_text="""
        future_section:
          whatever: 1
        """,
        env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
        """,
    )
    assert load_config(tmp_path).app.allowed_chat_ids == []
