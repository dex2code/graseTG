"""Tests for entry-point helpers."""

from grase_tg import _masked_proxy


def test_proxy_without_credentials_is_untouched():
    assert _masked_proxy("http://proxy.local:8080") == "http://proxy.local:8080"


def test_proxy_credentials_are_stripped():
    assert _masked_proxy("http://user:secret@proxy.local:8080") == "http://proxy.local:8080"


def test_proxy_username_only_is_stripped():
    assert _masked_proxy("http://user@proxy.local:8080") == "http://proxy.local:8080"


def test_proxy_without_port():
    assert _masked_proxy("http://user:secret@proxy.local") == "http://proxy.local:"
