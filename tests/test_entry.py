"""Tests for entry-point helpers: proxy session building and log masking."""

import pytest
from aiogram.client.session.aiohttp import AiohttpSession

from grase_tg import SUPPORTED_PROXY_SCHEMES, _build_session, _masked_proxy


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
