"""Tests for handlers: whitelist filter, routing, the text pipeline, stub replies."""

import logging
from datetime import datetime, timezone

import pytest
from aiogram.types import Chat, Message

import grase_tg.handlers as handlers_module
from conftest import BOT_USERNAME, BOT_MENTION, FakeMessage, mention_entities_for

from grase_tg.handlers import AllowedChatsFilter, split_text


# --- split_text ---

def test_split_short_text_is_single_chunk():
    assert split_text("hello") == ["hello"]


def test_split_exact_limit_is_single_chunk():
    assert split_text("a" * 4096) == ["a" * 4096]


def test_split_long_text_into_chunks():
    chunks = split_text("a" * 10_000)
    assert chunks == ["a" * 4096, "a" * 4096, "a" * 1808]


def test_split_empty_text_gives_one_empty_chunk():
    assert split_text("") == [""]


# --- AllowedChatsFilter ---

def _group_message(chat_id: int) -> Message:
    chat = Chat(id=chat_id, type="supergroup", title="Our Family")
    return Message(message_id=1, date=datetime.now(timezone.utc), chat=chat)


async def test_filter_passes_whitelisted_chat():
    f = AllowedChatsFilter(frozenset({-100}))
    assert await f(_group_message(-100)) is True


async def test_filter_rejects_unknown_chat_and_logs_once(caplog):
    f = AllowedChatsFilter(frozenset({-100}))
    with caplog.at_level(logging.INFO, logger="grase_tg.handlers"):
        assert await f(_group_message(-200)) is False
        assert await f(_group_message(-200)) is False  # same chat: no second log
    records = [r for r in caplog.records if "not whitelisted" in r.message]
    assert len(records) == 1


# --- routing through the handler ---

async def test_text_question_goes_to_llm_pipeline(handler, monkeypatch):
    async def ok(client, cfg, question):
        return "A" * 9000  # forces chunking

    monkeypatch.setattr(handlers_module, "ask_llm", ok)
    text = f"{BOT_MENTION} how to cook?"
    m = FakeMessage(text, mention_entities_for(text))
    await handler(m)
    assert m.sent == ["Got it, thinking…"]
    ack = m.acks[0]
    assert ack.edits == ["A" * 4096]
    assert ack.replies == ["A" * 4096, "A" * 808]


async def test_llm_failure_edits_ack_with_error_text(handler, monkeypatch):
    async def fail(client, cfg, question):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(handlers_module, "ask_llm", fail)
    text = f"{BOT_MENTION} hi"
    m = FakeMessage(text, mention_entities_for(text))
    await handler(m)
    assert m.sent == ["Got it, thinking…"]
    assert m.acks[0].edits == ["Failed to get a response from the model. Please try again later."]
    assert m.acks[0].replies == []


async def test_mention_without_question_gets_format_hint(handler):
    m = FakeMessage(BOT_MENTION, mention_entities_for(BOT_MENTION))
    await handler(m)
    assert m.sent == [handlers_module.FORMAT_HINT.format(username=BOT_USERNAME)]


async def test_unaddressed_messages_are_silent(handler):
    for m in (
        FakeMessage("plain chatter"),
        FakeMessage(photo=True, caption="just a photo"),
        FakeMessage(voice=True),
        FakeMessage(sticker=True),
    ):
        await handler(m)
        assert not m.sent


async def test_addressed_photo_gets_photo_stub(handler, monkeypatch):
    async def boom(client, cfg, question):  # the stub path must not touch the LLM
        raise AssertionError("LLM must not be called for stubs")

    monkeypatch.setattr(handlers_module, "ask_llm", boom)
    caption = f"{BOT_MENTION} what is this?"
    m = FakeMessage(photo=True, caption=caption, caption_entities=mention_entities_for(caption))
    await handler(m)
    assert m.sent == [handlers_module.PHOTO_STUB_TEXT]


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("photo", handlers_module.PHOTO_STUB_TEXT),
        ("audio", handlers_module.AUDIO_STUB_TEXT),
        ("sticker", handlers_module.STICKER_STUB_TEXT),
        ("something_new", handlers_module.OTHER_STUB_TEXT),  # fallback for kinds without a handler
    ],
)
async def test_stub_routing_by_kind(handler, monkeypatch, kind, expected):
    # Dormant kinds cannot be produced by extractors yet — inject directly to test the routing table.
    from grase_tg.content import Question

    monkeypatch.setattr(handlers_module, "extract_question", lambda m, u: Question(kind=kind, text=""))
    m = FakeMessage()
    await handler(m)
    assert m.sent == [expected]
