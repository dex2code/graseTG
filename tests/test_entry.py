"""Tests for entry-point helpers: proxy session building, log masking, startup failures."""

import logging

import pytest
from aiogram.client.session.aiohttp import AiohttpSession

from grase_tg import SUPPORTED_PROXY_SCHEMES, _build_session, _masked_proxy, run_bot


# --- _masked_proxy ---

def test_proxy_without_credentials_is_untouched():
    assert _masked_proxy("http://proxy.local:8080") == "http://proxy.local:8080"


def test_proxy_credentials_are_stripped():
    assert _masked_proxy("http://user:secret@proxy.local:8080") == "http://proxy.local:8080"


def test_proxy_username_only_is_stripped():
    assert _masked_proxy("http://user@proxy.local:8080") == "http://proxy.local:8080"


def test_proxy_without_port():
    assert _masked_proxy("http://user:secret@proxy.local") == "http://proxy.local:"


def test_socks_proxy_credentials_are_stripped():
    assert _masked_proxy("socks5://user:secret@proxy.local:1080") == "socks5://proxy.local:1080"


# --- _build_session ---

@pytest.mark.parametrize(
    "url",
    [
        "http://proxy.local:8080",
        "http://user:secret@proxy.local:8080",
        "socks4://proxy.local:1080",
        "socks5://proxy.local:1080",
        "socks5://user:secret@proxy.local:1080",
    ],
)
def test_build_session_accepts_http_and_socks(url):
    session = _build_session(url)
    assert isinstance(session, AiohttpSession)
    assert session.proxy == url


@pytest.mark.parametrize(
    "url",
    [
        "ftp://proxy.local:8080",   # unsupported scheme
        "proxy.local:8080",         # no scheme
        "socks5://proxy.local",     # no port -> rejected by the connector parser
        "",
    ],
)
def test_build_session_rejects_invalid_urls(url):
    with pytest.raises(ValueError, match="proxy scheme|port"):
        _build_session(url)


def test_supported_schemes_are_exactly_the_connector_ones():
    # Mirror of python-socks ProxyType: no https, no socks4a/socks5h.
    assert SUPPORTED_PROXY_SCHEMES == {"http", "socks4", "socks5"}


# --- startup failures are clean (a one-line error + SystemExit(1), no traceback) ---


def _write_env(base, token: str = "123456:ABC-test-token") -> None:
    (base / ".env").write_text(
        f"BOT_TOKEN={token}\nLLM_API_KEY=k\nLLM_BASE_URL=http://llm.test/v1\n", encoding="utf-8"
    )


async def test_invalid_config_exits_cleanly(tmp_path, monkeypatch, caplog, env_isolated):
    # A typo in config.yaml must produce a readable one-line error naming the
    # field — not a raw pydantic traceback.
    _write_env(tmp_path)
    (tmp_path / "config.yaml").write_text("llm:\n  temperature: oops\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.ERROR, logger="grase_tg"):
        with pytest.raises(SystemExit) as exc:
            await run_bot()
    assert exc.value.code == 1
    assert "Invalid configuration" in caplog.text
    assert "temperature" in caplog.text


async def test_broken_yaml_exits_cleanly(tmp_path, monkeypatch, caplog, env_isolated):
    _write_env(tmp_path)
    (tmp_path / "config.yaml").write_text("llm:\n\ttemperature: 5\n", encoding="utf-8")  # tab = YAML error
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.ERROR, logger="grase_tg"):
        with pytest.raises(SystemExit) as exc:
            await run_bot()
    assert exc.value.code == 1
    assert "Invalid configuration" in caplog.text


async def test_missing_secrets_exit_cleanly(tmp_path, monkeypatch, caplog, env_isolated):
    # No .env at all: the error must name the missing secrets.
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.ERROR, logger="grase_tg"):
        with pytest.raises(SystemExit) as exc:
            await run_bot()
    assert exc.value.code == 1
    assert "Invalid configuration" in caplog.text
    assert "bot_token" in caplog.text


async def test_malformed_bot_token_exits_cleanly(tmp_path, monkeypatch, caplog, env_isolated):
    # aiogram rejects a malformed token at Bot() construction: a clear message
    # instead of a TokenValidationError traceback.
    _write_env(tmp_path, token="garbage-not-a-token")
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.ERROR, logger="grase_tg"):
        with pytest.raises(SystemExit) as exc:
            await run_bot()
    assert exc.value.code == 1
    assert "BOT_TOKEN" in caplog.text
