"""Tests for content extraction: UTF-16 offsets, addressing, per-type dispatch."""

from types import SimpleNamespace

import pytest
from aiogram.enums import MessageEntityType

from conftest import BOT, BOT_MENTION, bot_message, human_message, mention_entities_for, mention_entity

from grase_tg.content import (
    _utf16_to_py_index,
    extract_audio,
    extract_media,
    extract_question,
    extract_sticker,
    extract_text,
)


# --- UTF-16 -> Python index conversion ---

def test_utf16_plain_ascii_passthrough():
    text = "@bot hello"
    assert _utf16_to_py_index(text, 0) == 0
    assert _utf16_to_py_index(text, 4) == 4


def test_utf16_emoji_counts_double():
    text = "🍜🍜 @bot"
    # two emoji = 4 UTF-16 units, plus one space -> '@' starts at unit 5, python index 3
    assert _utf16_to_py_index(text, 5) == 3


def test_utf16_position_beyond_text_clamps():
    assert _utf16_to_py_index("hi", 100) == 2


def test_utf16_zero_and_negative_are_start():
    assert _utf16_to_py_index("abc", 0) == 0
    assert _utf16_to_py_index("abc", -5) == 0


# --- extract_text: addressing via @mention ---

def test_text_mention_at_start(make_message):
    text = f"{BOT_MENTION} how to cook?"
    q = extract_text(make_message(text, mention_entities_for(text)), BOT)
    assert q is not None
    assert q.kind == "text" and q.text == "how to cook?"


def test_text_mention_at_end_question_before(make_message):
    text = f"what is the weather? {BOT_MENTION}"
    q = extract_text(make_message(text, mention_entities_for(text)), BOT)
    assert q is not None and q.text == "what is the weather?"


def test_text_mention_case_insensitive(make_message):
    text = "@FAMILY_BOT hi"
    q = extract_text(make_message(text, mention_entities_for(text)), BOT)
    assert q is not None and q.text == "hi"


def test_text_mention_mid_message_takes_text_after(make_message):
    text = f"hey {BOT_MENTION} what about dinner?"
    q = extract_text(make_message(text, mention_entities_for(text)), BOT)
    assert q is not None and q.text == "what about dinner?"


def test_text_mention_only_gives_empty_question(make_message):
    q = extract_text(make_message(BOT_MENTION, mention_entities_for(BOT_MENTION)), BOT)
    assert q is not None and q.text == ""


def test_text_emoji_before_mention_offset_correct(make_message):
    text = f"🍜🍜 {BOT_MENTION} question"
    q = extract_text(make_message(text, mention_entities_for(text)), BOT)
    assert q is not None and q.text == "question"


def test_text_wrong_username_not_addressed(make_message):
    m = make_message("@other_bot hi", [mention_entity(0, 10)])
    assert extract_text(m, BOT) is None


def test_text_no_entities_not_addressed(make_message):
    assert extract_text(make_message("plain chatter"), BOT) is None


def test_text_no_text_not_addressed(make_message):
    assert extract_text(make_message(text=None), BOT) is None


def test_text_mention_is_not_prefix_match(make_message):
    # '@family_bot_x' is a different user: the entity covers the longer mention
    m = make_message("@family_bot_x hi", [mention_entity(0, 13)])
    assert extract_text(m, BOT) is None


def test_text_non_mention_entity_ignored(make_message):
    url = SimpleNamespace(type=MessageEntityType.URL, offset=0, length=4)
    assert extract_text(make_message(BOT_MENTION, [url]), BOT) is None


# --- extract_text: addressing via a reply to the bot ---

def test_text_reply_to_bot_whole_text_is_the_question(make_message):
    m = make_message("what about dinner?", reply_to_message=bot_message())
    q = extract_text(m, BOT)
    assert q is not None
    assert q.kind == "text" and q.text == "what about dinner?"


def test_text_reply_carries_replied_answer_as_context(make_message):
    m = make_message("а что насчёт курицы?", reply_to_message=bot_message(text="Борщ варится 2 часа."))
    q = extract_text(m, BOT)
    assert q is not None
    assert q.text == "а что насчёт курицы?"
    assert q.context == "Борщ варится 2 часа."


def test_reply_with_mention_keeps_mention_text_and_reply_context(make_message):
    # Both at once: the mention shapes the question, the reply still
    # contributes its context — the user explicitly referenced that answer.
    text = f"{BOT_MENTION} а что насчёт курицы?"
    m = make_message(text, mention_entities_for(text), reply_to_message=bot_message(text="Борщ: 2 часа."))
    q = extract_text(m, BOT)
    assert q is not None
    assert q.text == "а что насчёт курицы?"
    assert q.context == "Борщ: 2 часа."


def test_mention_without_reply_has_no_context(make_message):
    text = f"{BOT_MENTION} hi"
    q = extract_text(make_message(text, mention_entities_for(text)), BOT)
    assert q is not None and q.context is None


def test_mention_in_reply_to_human_has_no_context(make_message):
    text = f"{BOT_MENTION} hi"
    m = make_message(text, mention_entities_for(text), reply_to_message=human_message(text="old"))
    q = extract_text(m, BOT)
    assert q is not None and q.context is None


def test_reply_to_blank_bot_message_has_no_context(make_message):
    m = make_message("hi", reply_to_message=bot_message(text="   "))
    q = extract_text(m, BOT)
    assert q is not None and q.context is None


def test_text_reply_to_another_user_not_addressed(make_message):
    m = make_message("hi", reply_to_message=human_message())
    assert extract_text(m, BOT) is None


def test_reply_that_also_mentions_prefers_mention_semantics(make_message):
    text = f"{BOT_MENTION} cook borscht"
    m = make_message(text, mention_entities_for(text), reply_to_message=bot_message())
    q = extract_text(m, BOT)
    assert q is not None and q.text == "cook borscht"


def test_topic_thread_link_to_bot_message_is_not_addressing(make_message):
    # In forum topics every message carries an implicit "reply" to the topic
    # root; that is threading, not addressing.
    m = make_message(
        "chat chatter",
        reply_to_message=bot_message(message_id=500),
        is_topic_message=True,
        message_thread_id=500,
    )
    assert extract_text(m, BOT) is None


def test_explicit_reply_to_bot_inside_a_topic_is_addressed(make_message):
    m = make_message(
        "a real question",
        reply_to_message=bot_message(message_id=501),
        is_topic_message=True,
        message_thread_id=500,
    )
    q = extract_text(m, BOT)
    assert q is not None and q.text == "a real question"


# --- extract_media: photos, videos, GIFs, documents ---

def test_photo_caption_mention_addressed(make_message):
    caption = f"{BOT_MENTION} what is on this photo?"
    q = extract_media(
        make_message(photo=True, caption=caption, caption_entities=mention_entities_for(caption)),
        BOT,
    )
    assert q is not None
    assert q.kind == "media"
    assert q.text == "what is on this photo?"
    assert q.image is None  # payload is not attached yet


def test_photo_caption_with_emoji_offset(make_message):
    caption = f"🍜 {BOT_MENTION} guess"
    q = extract_media(
        make_message(photo=True, caption=caption, caption_entities=mention_entities_for(caption)),
        BOT,
    )
    assert q is not None and q.text == "guess"


@pytest.mark.parametrize("media_flag", ["photo", "document", "video", "animation"])
def test_document_video_gif_caption_mention_addressed(make_message, media_flag):
    # Any caption-bearing attachment is addressed by an @mention in the caption.
    caption = f"{BOT_MENTION} what is this?"
    q = extract_media(
        make_message(**{media_flag: True, "caption": caption, "caption_entities": mention_entities_for(caption)}),
        BOT,
    )
    assert q is not None and q.kind == "media" and q.text == "what is this?"


@pytest.mark.parametrize("media_flag", ["photo", "document", "video", "animation"])
def test_document_video_gif_reply_to_bot_addressed(make_message, media_flag):
    q = extract_question(make_message(**{media_flag: True, "reply_to_message": bot_message()}), BOT)
    assert q is not None and q.kind == "media" and q.text == ""


def test_media_reply_with_caption_mention_keeps_context(make_message):
    caption = f"{BOT_MENTION} what is this?"
    m = make_message(
        photo=True,
        caption=caption,
        caption_entities=mention_entities_for(caption),
        reply_to_message=bot_message(text="prev answer"),
    )
    q = extract_media(m, BOT)
    assert q is not None and q.text == "what is this?" and q.context == "prev answer"


def test_media_without_caption_not_addressed(make_message):
    assert extract_media(make_message(photo=True), BOT) is None


def test_media_caption_without_mention_not_addressed(make_message):
    assert extract_media(make_message(document=True, caption="just a file"), BOT) is None


def test_not_a_media_message_not_addressed(make_message):
    assert extract_media(make_message(text="not a media message"), BOT) is None


def test_media_reply_to_bot_without_caption_addressed(make_message):
    m = make_message(photo=True, reply_to_message=bot_message())
    q = extract_media(m, BOT)
    assert q is not None and q.kind == "media" and q.text == ""


def test_media_reply_to_bot_with_caption_keeps_caption(make_message):
    m = make_message(video=True, caption="what is this?", reply_to_message=bot_message())
    q = extract_media(m, BOT)
    assert q is not None and q.kind == "media" and q.text == "what is this?"


def test_media_reply_to_another_user_not_addressed(make_message):
    m = make_message(document=True, reply_to_message=human_message())
    assert extract_media(m, BOT) is None


# --- audio / sticker: addressed only via a reply to the bot ---

def test_voice_reply_to_bot_addressed_as_audio(make_message):
    m = make_message(voice=True, reply_to_message=bot_message(text="old answer"))
    q = extract_audio(m, BOT)
    assert q is not None and q.kind == "audio" and q.text == ""
    assert q.context == "old answer"


def test_audio_file_reply_caption_kept_as_text(make_message):
    m = make_message(audio=True, caption="listen to this", reply_to_message=bot_message())
    q = extract_audio(m, BOT)
    assert q is not None and q.kind == "audio" and q.text == "listen to this"


def test_video_note_reply_addressed_as_audio(make_message):
    m = make_message(video_note=True, reply_to_message=bot_message())
    q = extract_audio(m, BOT)
    assert q is not None and q.kind == "audio"


def test_voice_without_reply_is_unaddressed(make_message):
    assert extract_audio(make_message(voice=True), BOT) is None


def test_voice_reply_to_another_user_unaddressed(make_message):
    m = make_message(voice=True, reply_to_message=human_message())
    assert extract_audio(m, BOT) is None


def test_sticker_reply_to_bot_addressed(make_message):
    m = make_message(sticker=True, reply_to_message=bot_message())
    q = extract_sticker(m, BOT)
    assert q is not None and q.kind == "sticker" and q.text == ""


def test_sticker_without_reply_is_unaddressed(make_message):
    assert extract_sticker(make_message(sticker=True), BOT) is None


def test_sticker_reply_to_another_user_unaddressed(make_message):
    m = make_message(sticker=True, reply_to_message=human_message())
    assert extract_sticker(m, BOT) is None


# --- dispatch ---

def test_dispatch_text_first(make_message):
    text = f"{BOT_MENTION} hi"
    q = extract_question(make_message(text, mention_entities_for(text)), BOT)
    assert q is not None and q.kind == "text"


def test_dispatch_media(make_message):
    caption = f"{BOT_MENTION} hi"
    q = extract_question(
        make_message(photo=True, caption=caption, caption_entities=mention_entities_for(caption)),
        BOT,
    )
    assert q is not None and q.kind == "media"


def test_dispatch_document(make_message):
    caption = f"{BOT_MENTION} what is in this file?"
    q = extract_question(
        make_message(document=True, caption=caption, caption_entities=mention_entities_for(caption)),
        BOT,
    )
    assert q is not None and q.kind == "media"


def test_dispatch_text_reply(make_message):
    q = extract_question(make_message("hi", reply_to_message=bot_message()), BOT)
    assert q is not None and q.kind == "text"


def test_dispatch_media_reply(make_message):
    q = extract_question(make_message(photo=True, reply_to_message=bot_message()), BOT)
    assert q is not None and q.kind == "media"


def test_dispatch_voice_reply(make_message):
    q = extract_question(make_message(voice=True, reply_to_message=bot_message()), BOT)
    assert q is not None and q.kind == "audio"


def test_dispatch_sticker_reply(make_message):
    q = extract_question(make_message(sticker=True, reply_to_message=bot_message()), BOT)
    assert q is not None and q.kind == "sticker"


def test_dispatch_unaddressed_is_none(make_message):
    for m in (
        make_message("plain chatter"),
        make_message(voice=True),
        make_message(sticker=True),
        make_message(photo=True, caption="no mention"),
        make_message("reply to a human", reply_to_message=human_message()),
    ):
        assert extract_question(m, BOT) is None
