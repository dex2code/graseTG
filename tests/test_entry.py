"""Tests for entry-point helpers: proxy session building, log masking, startup failures,
MCP registry wiring."""

import logging
from contextlib import AsyncExitStack

import pytest
from aiogram.client.session.aiohttp import AiohttpSession

from grase_tg import SUPPORTED_PROXY_SCHEMES, _build_mcp_registry, _build_session, _masked_proxy, run_bot
from grase_tg.config import load_config


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


# --- MCP registry wiring (docs/mcp-tools-spec.md §6.1) ---


def _write_mcp_config(base, yaml_text: str) -> None:
    _write_env(base)
    (base / "config.yaml").write_text(yaml_text, encoding="utf-8")


async def test_missing_mcp_auth_env_exits_cleanly(tmp_path, monkeypatch, caplog, env_isolated):
    # A missing auth_env variable is a CONFIG error: clean startup exit
    # (one-line error, SystemExit(1)) — connecting unauthenticated by
    # accident would be worse than not connecting.
    _write_mcp_config(tmp_path, "tools:\n  mcp:\n    url: http://mcp.test/mcp\n    auth_env: MCP_ABSENT_TOKEN\n")
    monkeypatch.chdir(tmp_path)
    config = load_config(tmp_path)
    async with AsyncExitStack() as stack:
        with caplog.at_level(logging.ERROR, logger="grase_tg"):
            with pytest.raises(SystemExit) as exc:
                await _build_mcp_registry(config, stack)
    assert exc.value.code == 1
    assert "Invalid tools.mcp configuration" in caplog.text
    assert "MCP_ABSENT_TOKEN" in caplog.text


async def test_unreachable_mcp_server_degrades_to_none(tmp_path, monkeypatch, caplog, env_isolated):
    # An unreachable server is INFRASTRUCTURE, not config: the bot must keep
    # starting (registry=None, warning in the log) — a restart race with the
    # MCP server must not take the whole bot down.
    _write_mcp_config(tmp_path, "tools:\n  mcp:\n    url: http://127.0.0.1:1/mcp\n    timeout_seconds: 2\n")
    monkeypatch.chdir(tmp_path)
    config = load_config(tmp_path)
    async with AsyncExitStack() as stack:
        with caplog.at_level(logging.WARNING, logger="grase_tg"):
            registry = await _build_mcp_registry(config, stack)
    assert registry is None
    assert "running without tools" in caplog.text


async def test_no_mcp_section_builds_no_registry(tmp_path, monkeypatch, caplog, env_isolated):
    # Default config (no tools section): no MCP machinery at all.
    _write_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    config = load_config(tmp_path)
    async with AsyncExitStack() as stack:
        registry = await _build_mcp_registry(config, stack)
    assert registry is None
    assert "MCP" not in caplog.text
