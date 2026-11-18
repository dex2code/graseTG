"""Tests for configuration loading: .env secrets, config.yaml, defaults."""

import textwrap
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from grase_tg.config import CONFIG_DIR_ENV, default_config_dir, env_value, load_config


ENV = {
    "BOT_TOKEN": "test-token-123",
    "LLM_API_KEY": "sk-test-key",
    "LLM_BASE_URL": "http://llm.test/v1",
}


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


def test_dotenv_values_are_taken_literally(tmp_path, env_isolated):
    # A secret containing $VAR / ${VAR} must NOT be interpolated (expanded or
    # emptied) — with the default interpolation on, the value would be silently
    # corrupted and fail later as a cryptic auth error.
    env_isolated.setenv("WHO", "intruder")
    env_text = (
        f"BOT_TOKEN={ENV['BOT_TOKEN']}\n"
        "LLM_API_KEY=sk-${WHO}-key-$1\n"
        f"LLM_BASE_URL={ENV['LLM_BASE_URL']}\n"
    )
    _write(tmp_path, env_text=env_text)
    cfg = load_config(tmp_path)
    assert cfg.secrets.llm_api_key.get_secret_value() == "sk-${WHO}-key-$1"


# --- config directory resolution ---


def test_config_dir_defaults_to_cwd(env_isolated, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert default_config_dir() == tmp_path


def test_config_dir_env_override_wins(env_isolated):
    env_isolated.setenv(CONFIG_DIR_ENV, "/somewhere/else")
    assert default_config_dir() == Path("/somewhere/else")


# --- user-facing messages config ---


def test_message_defaults_are_english(tmp_path, env_isolated):
    _write(tmp_path, env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
    """)
    messages = load_config(tmp_path).app.messages
    assert messages.ack == "Got it, thinking…"
    assert "{username}" in messages.format_hint  # the template placeholder is required at runtime
    assert messages.empty_answer


def test_custom_messages_loaded(tmp_path, env_isolated):
    _write(
        tmp_path,
        yaml_text="""
        messages:
          ack: Понял, думаю…
          error: Не получилось — попробуй позже.
        """,
        env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
        """,
    )
    messages = load_config(tmp_path).app.messages
    assert messages.ack == "Понял, думаю…"
    assert messages.error == "Не получилось — попробуй позже."
    assert messages.media_stub  # unset fields keep their defaults


def test_unknown_message_key_rejected(tmp_path, env_isolated):
    _write(
        tmp_path,
        yaml_text="""
        messages:
          acck: typo
        """,
        env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
        """,
    )
    with pytest.raises(ValidationError) as exc:
        load_config(tmp_path)
    assert "acck" in str(exc.value)


def test_empty_message_text_rejected(tmp_path, env_isolated):
    # An empty user-facing text is a config error, not an invisible bot reply.
    _write(
        tmp_path,
        yaml_text="""
        messages:
          ack: ""
        """,
        env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
        """,
    )
    with pytest.raises(ValidationError) as exc:
        load_config(tmp_path)
    assert "ack" in str(exc.value)


def test_format_hint_unknown_placeholder_rejected(tmp_path, env_isolated):
    # {usrname} is a typo: .format(username=...) would raise KeyError inside
    # the handler — it must fail at startup instead.
    _write(
        tmp_path,
        yaml_text="""
        messages:
          format_hint: "Hi {usrname}, ask me anything"
        """,
        env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
        """,
    )
    with pytest.raises(ValidationError) as exc:
        load_config(tmp_path)
    assert "placeholder" in str(exc.value)


@pytest.mark.parametrize(
    ("llm_yaml", "bad_field"),
    [
        ("temperature: -0.1", "temperature"),
        ("max_tokens: 0", "max_tokens"),
        ("timeout_seconds: 0", "timeout_seconds"),
        ("model: ''", "model"),
        ("top_p: 0", "top_p"),
        ("top_p: 1.5", "top_p"),
        ("presence_penalty: -3", "presence_penalty"),
        ("presence_penalty: 3", "presence_penalty"),
    ],
)
def test_out_of_range_llm_settings_rejected_at_startup(tmp_path, env_isolated, llm_yaml, bad_field):
    # Nonsense values must fail at startup, not as a runtime API error.
    _write(
        tmp_path,
        yaml_text=f"""
        llm:
          {llm_yaml}
        """,
        env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
        """,
    )
    with pytest.raises(ValidationError) as exc:
        load_config(tmp_path)
    assert bad_field in str(exc.value)


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


def test_unknown_yaml_keys_are_rejected(tmp_path, env_isolated):
    # A typo in a key must fail at startup, not silently disable the setting.
    _write(
        tmp_path,
        yaml_text="""
        allowed_chat_idz: [-1001]
        """,
        env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
        """,
    )
    with pytest.raises(ValidationError) as exc:
        load_config(tmp_path)
    assert "allowed_chat_idz" in str(exc.value)
    assert "Extra inputs are not permitted" in str(exc.value)


def test_unknown_llm_keys_are_rejected(tmp_path, env_isolated):
    _write(
        tmp_path,
        yaml_text="""
        llm:
          modell: typo-model
        """,
        env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
        """,
    )
    with pytest.raises(ValidationError) as exc:
        load_config(tmp_path)
    assert "modell" in str(exc.value)


def test_empty_secret_values_are_rejected(tmp_path, env_isolated):
    # "BOT_TOKEN=" in .env must fail at startup, not at the first Telegram call.
    _write(
        tmp_path,
        env_text="""
        BOT_TOKEN=
        LLM_API_KEY=
        LLM_BASE_URL=
        """,
    )
    with pytest.raises(ValidationError) as exc:
        load_config(tmp_path)
    message = str(exc.value)
    assert "bot_token" in message and "llm_api_key" in message and "llm_base_url" in message


# --- tools (MCP) ---


def test_tools_default_to_disabled(tmp_path, env_isolated):
    # No tools section = tools fully disabled: the loop must send no `tools`
    # field at all (behavior identical to the pre-tools bot).
    _write(tmp_path, env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
    """)
    cfg = load_config(tmp_path)
    assert cfg.app.tools.mcp is None
    assert cfg.app.llm.tool_rounds == 2
    assert cfg.app.messages.tool_failure  # the honest-failure notice has a default


def test_mcp_settings_loaded(tmp_path, env_isolated):
    _write(
        tmp_path,
        yaml_text="""
        tools:
          mcp:
            url: http://mcp.test:8080/mcp
            auth_env: MCP_TOKEN
            timeout_seconds: 10
            allow: [get_forecast, get_exchange_rate]
        """,
        env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
        """,
    )
    mcp = load_config(tmp_path).app.tools.mcp
    assert mcp is not None
    assert mcp.url == "http://mcp.test:8080/mcp"
    assert mcp.auth_env == "MCP_TOKEN"
    assert mcp.timeout_seconds == 10
    assert mcp.allow == ["get_forecast", "get_exchange_rate"]


@pytest.mark.parametrize(
    ("tools_yaml", "bad_piece"),
    [
        ("tools:\n  mcpp:\n    url: http://x/mcp", "mcpp"),  # typo in the section name
        ("tools:\n  mcp:\n    urll: http://x/mcp", "urll"),  # typo in a field name
        ("tools:\n  mcp:\n    url: not-a-url", "http(s) URL"),  # no scheme/host
        ("tools:\n  mcp:\n    url: ftp://mcp.test/mcp", "http(s) URL"),  # wrong scheme
        ("tools:\n  mcp:\n    timeout_seconds: 0", "timeout_seconds"),
        ("llm:\n  tool_rounds: 0", "tool_rounds"),  # 0 would advertise tools but never run them
    ],
)
def test_tools_config_errors_fail_at_startup(tmp_path, env_isolated, tools_yaml, bad_piece):
    _write(
        tmp_path,
        yaml_text=tools_yaml,
        env_text=f"""
        BOT_TOKEN={ENV['BOT_TOKEN']}
        LLM_API_KEY={ENV['LLM_API_KEY']}
        LLM_BASE_URL={ENV['LLM_BASE_URL']}
        """,
    )
    with pytest.raises(ValidationError) as exc:
        load_config(tmp_path)
    assert bad_piece in str(exc.value)


# --- env_value: the one precedence rule behind secrets AND mcp.auth_env ---


def test_env_value_environment_wins_over_dotenv_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("PROBE_A=from-file\n", encoding="utf-8")
    monkeypatch.setenv("PROBE_A", "from-os")
    assert env_value("PROBE_A", tmp_path) == "from-os"


def test_env_value_falls_back_to_dotenv_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("PROBE_B=from-file\n", encoding="utf-8")
    monkeypatch.delenv("PROBE_B", raising=False)
    assert env_value("PROBE_B", tmp_path) == "from-file"


def test_env_value_empty_means_unset(tmp_path, monkeypatch):
    # "X=" in .env or an empty exported value must behave like absent:
    # pydantic's min_length on Secrets relies on this.
    (tmp_path / ".env").write_text("PROBE_C=\n", encoding="utf-8")
    monkeypatch.setenv("PROBE_C", "")
    assert env_value("PROBE_C", tmp_path) is None


def test_env_value_absent_everywhere_is_none(tmp_path, monkeypatch):
    monkeypatch.delenv("PROBE_D", raising=False)
    assert env_value("PROBE_D", tmp_path) is None


def test_env_value_is_literal_not_interpolated(tmp_path, monkeypatch):
    # A token containing $ must survive verbatim: dotenv's default
    # interpolation would expand (or empty) it — same rule as _read_secrets.
    (tmp_path / ".env").write_text("PROBE_E=ghp_$urely_${nope}\n", encoding="utf-8")
    monkeypatch.delenv("PROBE_E", raising=False)
    assert env_value("PROBE_E", tmp_path) == "ghp_$urely_${nope}"
