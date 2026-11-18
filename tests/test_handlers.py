"""Tests for handlers: whitelist filter, routing, the text pipeline, stub replies.

User-facing texts are asserted via the config fixture (the English defaults),
not via literals: the texts are configurable in config.yaml.
"""

import logging
from datetime import datetime, timezone
from typing import cast

import pytest
from aiogram.types import Chat, Message

import grase_tg.handlers as handlers_module
from conftest import BOT, BOT_MENTION, BOT_USERNAME, FakeAck, bot_message, mention_entities_for

from grase_tg.config import MessagesSettings
from grase_tg.content import QuestionKind
from grase_tg.handlers import AllowedChatsFilter, split_text

# Default texts: the single source is the config model, not test literals.
DEFAULTS = MessagesSettings()


# --- split_text ---

def test_split_short_text_is_single_chunk():
    assert split_text("hello") == ["hello"]


def test_split_exact_limit_is_single_chunk():
    assert split_text("a" * 4096) == ["a" * 4096]


def test_split_long_text_into_chunks():
    chunks = split_text("a" * 10_000)
    assert chunks == ["a" * 4096, "a" * 4096, "a" * 1808]


def test_split_counts_emoji_as_two_utf16_units():
    # Telegram measures length in UTF-16 code units: one noodle emoji = 2 units.
    # 3000 emoji = 6000 units -> 2048 emoji (4096 units) + 952 emoji (1904 units).
    chunks = split_text("🍜" * 3000)
    assert [len(c.encode("utf-16-le")) // 2 for c in chunks] == [4096, 1904]


def test_split_never_breaks_an_astral_character():
    # The 4096th unit boundary falls between 'a'*4095 and the 2-unit emoji:
    # the emoji must move entirely into the second chunk, never be split.
    text = "a" * 4095 + "🍜b"
    assert split_text(text) == ["a" * 4095, "🍜b"]


def test_split_empty_text_gives_no_chunks():
    # ask_llm never returns empty text (empty_answer fallback), so this cannot
    # happen in practice; the contract is honest: empty input -> empty output.
    assert split_text("") == []


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

async def test_text_question_goes_to_llm_pipeline(handler, config, make_message, monkeypatch):
    async def ok(client, cfg, question):
        return "A" * 9000  # forces chunking

    monkeypatch.setattr(handlers_module, "ask_llm", ok)
    text = f"{BOT_MENTION} how to cook?"
    m = make_message(text, mention_entities_for(text))
    await handler(m)
    assert m.sent == [config.app.messages.ack]
    ack = m.acks[0]
    assert ack.edits == ["A" * 4096]
    assert ack.replies == ["A" * 4096, "A" * 808]


async def test_llm_failure_edits_ack_with_error_text(handler, config, make_message, monkeypatch):
    async def fail(client, cfg, question):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(handlers_module, "ask_llm", fail)
    text = f"{BOT_MENTION} hi"
    m = make_message(text, mention_entities_for(text))
    await handler(m)
    assert m.sent == [config.app.messages.ack]
    assert m.acks[0].edits == [config.app.messages.error]
    assert m.acks[0].replies == []


async def test_mention_without_question_gets_format_hint(handler, config, make_message):
    m = make_message(BOT_MENTION, mention_entities_for(BOT_MENTION))
    await handler(m)
    assert m.sent == [config.app.messages.format_hint.format(username=BOT_USERNAME)]


async def test_unaddressed_messages_are_silent(handler, make_message):
    for m in (
        make_message("plain chatter"),
        make_message(photo=True, caption="just a photo"),
        make_message(voice=True),
        make_message(sticker=True),
    ):
        await handler(m)
        assert not m.sent


async def test_reply_to_bot_text_goes_to_llm_pipeline(handler, config, make_message, monkeypatch):
    captured: dict = {}

    async def ok(client, cfg, question):
        captured["question"] = question
        return "the answer"

    monkeypatch.setattr(handlers_module, "ask_llm", ok)
    m = make_message("and what about dinner?", reply_to_message=bot_message())
    await handler(m)
    assert m.sent == [config.app.messages.ack]
    assert captured["question"].kind == "text"
    assert captured["question"].text == "and what about dinner?"
    assert m.acks[0].edits == ["the answer"]


async def test_replied_bot_answer_reaches_llm_as_context(handler, make_message, monkeypatch):
    captured: dict = {}

    async def ok(client, cfg, question):
        captured["question"] = question
        return "answer"

    monkeypatch.setattr(handlers_module, "ask_llm", ok)
    m = make_message("а что насчёт курицы?", reply_to_message=bot_message(text="Борщ варится 2 часа."))
    await handler(m)
    assert captured["question"].context == "Борщ варится 2 часа."


async def test_reply_with_mention_keeps_context_up_to_llm(handler, make_message, monkeypatch):
    # Mention + reply together: the question text comes from the mention
    # semantics, the replied-to answer still travels as context.
    captured: dict = {}

    async def ok(client, cfg, question):
        captured["question"] = question
        return "answer"

    monkeypatch.setattr(handlers_module, "ask_llm", ok)
    text = f"{BOT_MENTION} а что насчёт курицы?"
    m = make_message(text, mention_entities_for(text), reply_to_message=bot_message(text="Борщ варится 2 часа."))
    await handler(m)
    assert captured["question"].text == "а что насчёт курицы?"
    assert captured["question"].context == "Борщ варится 2 часа."


async def test_reply_with_mention_to_hint_strips_context(handler, config, make_message, monkeypatch):
    # The hint is a service text: even a mention + reply to it must not leak
    # the hint into the LLM request as fake dialogue context.
    captured: dict = {}

    async def ok(client, cfg, question):
        captured["question"] = question
        return "answer"

    monkeypatch.setattr(handlers_module, "ask_llm", ok)
    hint = config.app.messages.format_hint.format(username=BOT_USERNAME)
    text = f"{BOT_MENTION} как спросить?"
    m = make_message(text, mention_entities_for(text), reply_to_message=bot_message(text=hint))
    await handler(m)
    assert captured["question"].context is None


async def test_reply_to_service_texts_carries_no_context(handler, config, make_message, monkeypatch):
    # Replying to an error, a stub or the format hint must not feed those
    # texts to the model as fake dialogue context. (The ack is not here:
    # replies to the pending ack are skipped entirely — see the tests below.)
    captured: dict = {}

    async def ok(client, cfg, question):
        captured["question"] = question
        return "answer"

    monkeypatch.setattr(handlers_module, "ask_llm", ok)
    messages = config.app.messages
    service_texts = [
        messages.error,
        messages.empty_answer,
        messages.media_stub,
        messages.format_hint.format(username=BOT_USERNAME),
    ]
    for service_text in service_texts:
        m = make_message("try again", reply_to_message=bot_message(text=service_text))
        await handler(m)
        assert captured["question"].context is None, service_text
        assert captured["question"].text == "try again", service_text


async def test_ack_skip_uses_configured_text(handler, config, make_message, monkeypatch):
    # The ack-skip must use the CONFIGURED ack text, not the English default:
    # a reply to the translated pending ack is skipped before the LLM is called.
    captured: dict = {}

    async def ok(client, cfg, question):
        captured["question"] = question
        return "answer"

    monkeypatch.setattr(handlers_module, "ask_llm", ok)
    config.app.messages.ack = "Понял, думаю…"
    # Rebuild the handler: the ack text is read at router build time.
    from grase_tg.handlers import build_router
    from grase_tg.llm import make_client

    translated = build_router(config, make_client(config), BOT).observers["message"].handlers[0].callback

    # Reply to the translated (configured) ack: skipped entirely.
    m = make_message("ещё раз", reply_to_message=bot_message(text="Понял, думаю…"))
    await translated(m)
    assert m.sent == []
    assert "question" not in captured

    # The English default is not this deployment's ack: it passes through as
    # (harmless one-line) context — the rules cover exactly the current texts.
    m = make_message("и ещё", reply_to_message=bot_message(text="Got it, thinking…"))
    await translated(m)
    assert captured["question"].context == "Got it, thinking…"


async def test_reply_to_pending_ack_is_skipped(handler, config, make_message, monkeypatch):
    # While the model is generating, the user's addition to the pending ack
    # is skipped by design: no second LLM call, no reply — wait for the answer.
    async def boom(client, cfg, question):
        raise AssertionError("LLM must not be called for a reply to the pending ack")

    monkeypatch.setattr(handlers_module, "ask_llm", boom)
    m = make_message("ой, ещё вопрос!", reply_to_message=bot_message(text=config.app.messages.ack))
    await handler(m)
    assert m.sent == []


async def test_reply_to_pending_ack_with_mention_is_skipped(handler, config, make_message, monkeypatch):
    async def boom(client, cfg, question):
        raise AssertionError("LLM must not be called for a reply to the pending ack")

    monkeypatch.setattr(handlers_module, "ask_llm", boom)
    text = f"{BOT_MENTION} ой, ещё вопрос!"
    m = make_message(
        text,
        mention_entities_for(text),
        reply_to_message=bot_message(text=config.app.messages.ack),
    )
    await handler(m)
    assert m.sent == []


async def test_voice_reply_to_pending_ack_is_skipped(handler, config, make_message, monkeypatch):
    async def boom(client, cfg, question):
        raise AssertionError("LLM must not be called for a reply to the pending ack")

    monkeypatch.setattr(handlers_module, "ask_llm", boom)
    m = make_message(voice=True, reply_to_message=bot_message(text=config.app.messages.ack))
    await handler(m)
    assert m.sent == []  # not even the audio stub: the rule covers all kinds


async def test_reply_to_error_text_still_asks_the_llm(handler, config, make_message, monkeypatch):
    # The error text is not an ack: a reply to it is a legitimate retry.
    captured: dict = {}

    async def ok(client, cfg, question):
        captured["question"] = question
        return "answer"

    monkeypatch.setattr(handlers_module, "ask_llm", ok)
    m = make_message("try again please", reply_to_message=bot_message(text=config.app.messages.error))
    await handler(m)
    assert captured["question"].text == "try again please"
    assert captured["question"].context is None
    assert m.acks[0].edits == ["answer"]


def test_strip_service_context_keeps_real_answers():
    from grase_tg.content import Question
    from grase_tg.handlers import strip_service_context

    non_answers = frozenset({"service one", "service two"})
    stripped = strip_service_context(Question(kind="text", text="q", context="service one"), non_answers)
    assert stripped.context is None
    kept = strip_service_context(Question(kind="text", text="q", context="real answer"), non_answers)
    assert kept.context == "real answer"
    untouched = Question(kind="text", text="q")
    assert strip_service_context(untouched, non_answers) is untouched


async def test_reply_to_bot_with_voice_gets_audio_stub(handler, config, make_message, monkeypatch):
    async def boom(client, cfg, question):  # stubs must not touch the LLM
        raise AssertionError("LLM must not be called for stubs")

    monkeypatch.setattr(handlers_module, "ask_llm", boom)
    m = make_message(voice=True, reply_to_message=bot_message())
    await handler(m)
    assert m.sent == [config.app.messages.audio_stub]


async def test_reply_to_bot_with_sticker_gets_sticker_stub(handler, config, make_message, monkeypatch):
    async def boom(client, cfg, question):  # stubs must not touch the LLM
        raise AssertionError("LLM must not be called for stubs")

    monkeypatch.setattr(handlers_module, "ask_llm", boom)
    m = make_message(sticker=True, reply_to_message=bot_message())
    await handler(m)
    assert m.sent == [config.app.messages.sticker_stub]


async def test_addressed_media_gets_stub(handler, config, make_message, monkeypatch):
    async def boom(client, cfg, question):  # the stub path must not touch the LLM
        raise AssertionError("LLM must not be called for stubs")

    monkeypatch.setattr(handlers_module, "ask_llm", boom)
    caption = f"{BOT_MENTION} what is this?"
    m = make_message(photo=True, caption=caption, caption_entities=mention_entities_for(caption))
    await handler(m)
    assert m.sent == [config.app.messages.media_stub]


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("media", DEFAULTS.media_stub),
        ("audio", DEFAULTS.audio_stub),
        ("sticker", DEFAULTS.sticker_stub),
        ("something_new", DEFAULTS.other_stub),  # fallback for kinds without a handler
    ],
)
async def test_stub_routing_by_kind(handler, make_message, monkeypatch, kind, expected):
    # The "other" fallback is the safety net for kinds the extractors do not
    # produce yet: a deliberately outside-the-set kind is cast in to exercise it.
    from grase_tg.content import Question

    monkeypatch.setattr(
        handlers_module,
        "extract_question",
        lambda m, u: Question(kind=cast(QuestionKind, kind), text=""),
    )
    m = make_message()
    await handler(m)
    assert m.sent == [expected]


# --- delivery failures are contained (no exception may escape the handler) ---


async def test_ack_send_failure_skips_llm_and_is_contained(handler, make_message, monkeypatch):
    async def boom(client, cfg, question):
        raise AssertionError("LLM must not be called when the ack cannot be sent")

    monkeypatch.setattr(handlers_module, "ask_llm", boom)

    async def broken_reply(_text: str):
        raise RuntimeError("telegram is down")

    text = f"{BOT_MENTION} hi"
    m = make_message(text, mention_entities_for(text))
    m.reply = broken_reply
    await handler(m)  # must not raise
    assert m.sent == []


async def test_error_edit_failure_falls_back_to_plain_reply(handler, config, make_message, monkeypatch):
    async def fail(client, cfg, question):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(handlers_module, "ask_llm", fail)

    async def broken_edit(self, _text: str) -> None:
        raise RuntimeError("edit rejected")

    monkeypatch.setattr(FakeAck, "edit_text", broken_edit)
    text = f"{BOT_MENTION} hi"
    m = make_message(text, mention_entities_for(text))
    await handler(m)  # must not raise
    # The error text still reaches the user — as a fresh reply, not a stale ack.
    assert m.sent == [config.app.messages.ack, config.app.messages.error]
    assert m.acks[0].edits == []
    assert m.acks[0].replies == []


async def test_answer_edit_failure_resends_all_chunks(handler, config, make_message, monkeypatch):
    async def ok(client, cfg, question):
        return "A" * 9000  # forces chunking

    monkeypatch.setattr(handlers_module, "ask_llm", ok)
    monkeypatch.setattr(FakeAck, "fail_edit", True)
    text = f"{BOT_MENTION} hi"
    m = make_message(text, mention_entities_for(text))
    await handler(m)
    chunks = split_text("A" * 9000)
    # The ack could not be edited with the answer -> everything is re-sent
    # as plain replies to the original message.
    assert m.sent == [config.app.messages.ack, *chunks]
    assert m.acks[0].edits == []


async def test_partial_delivery_failure_resends_only_missing_chunks(handler, config, make_message, monkeypatch):
    async def ok(client, cfg, question):
        return "A" * 9000  # forces chunking

    monkeypatch.setattr(handlers_module, "ask_llm", ok)
    monkeypatch.setattr(FakeAck, "fail_reply", True)  # ack.reply fails, edit_text works
    text = f"{BOT_MENTION} hi"
    m = make_message(text, mention_entities_for(text))
    await handler(m)
    chunks = split_text("A" * 9000)
    # Chunk 0 was delivered via edit; the follow-up replies failed, so only
    # chunks 1 and 2 are re-sent — chunk 0 is not duplicated.
    assert m.acks[0].edits == [chunks[0]]
    assert m.acks[0].replies == []
    assert m.sent == [config.app.messages.ack, chunks[1], chunks[2]]


async def test_fallback_send_failure_is_contained(handler, config, make_message, monkeypatch):
    async def ok(client, cfg, question):
        return "answer"

    monkeypatch.setattr(handlers_module, "ask_llm", ok)
    monkeypatch.setattr(FakeAck, "fail_edit", True)  # forces the fallback path

    text = f"{BOT_MENTION} hi"
    m = make_message(text, mention_entities_for(text))
    real_reply = m.reply

    async def reply_until_ack(_text: str):
        if not m.acks:
            return await real_reply(_text)  # the ack still goes through
        raise RuntimeError("telegram is down")  # the fallback send fails

    m.reply = reply_until_ack
    await handler(m)  # must not raise
    assert m.sent == [config.app.messages.ack]


async def test_stub_reply_failure_is_contained(handler, make_message, monkeypatch):
    async def boom(client, cfg, question):
        raise AssertionError("LLM must not be called for stubs")

    monkeypatch.setattr(handlers_module, "ask_llm", boom)

    async def broken_reply(_text: str):
        raise RuntimeError("telegram is down")

    caption = f"{BOT_MENTION} what is this?"
    m = make_message(photo=True, caption=caption, caption_entities=mention_entities_for(caption))
    m.reply = broken_reply
    await handler(m)  # must not raise
    assert m.sent == []
