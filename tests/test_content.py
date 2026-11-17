"""Tests for content extraction: UTF-16 offsets, addressing, per-type dispatch."""

from types import SimpleNamespace

from aiogram.enums import MessageEntityType

from conftest import BOT_USERNAME, BOT_MENTION, mention_entities_for, mention_entity

from grase_tg.content import (
    _utf16_to_py_index,
    extract_audio,
    extract_photo,
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


# --- extract_text ---

def test_text_mention_at_start(make_message):
    text = f"{BOT_MENTION} how to cook?"
    q = extract_text(make_message(text, mention_entities_for(text)), BOT_USERNAME)
    assert q is not None
    assert q.kind == "text" and q.text == "how to cook?"


def test_text_mention_at_end_question_before(make_message):
    text = f"what is the weather? {BOT_MENTION}"
    q = extract_text(make_message(text, mention_entities_for(text)), BOT_USERNAME)
    assert q is not None and q.text == "what is the weather?"


def test_text_mention_case_insensitive(make_message):
    text = "@FAMILY_BOT hi"
    q = extract_text(make_message(text, mention_entities_for(text)), BOT_USERNAME)
    assert q is not None and q.text == "hi"


def test_text_mention_mid_message_takes_text_after(make_message):
    text = f"hey {BOT_MENTION} what about dinner?"
    q = extract_text(make_message(text, mention_entities_for(text)), BOT_USERNAME)
    assert q is not None and q.text == "what about dinner?"


def test_text_mention_only_gives_empty_question(make_message):
    q = extract_text(make_message(BOT_MENTION, mention_entities_for(BOT_MENTION)), BOT_USERNAME)
    assert q is not None and q.text == ""


def test_text_emoji_before_mention_offset_correct(make_message):
    text = f"🍜🍜 {BOT_MENTION} question"
    q = extract_text(make_message(text, mention_entities_for(text)), BOT_USERNAME)
    assert q is not None and q.text == "question"


def test_text_wrong_username_not_addressed(make_message):
    m = make_message("@other_bot hi", [mention_entity(0, 10)])
    assert extract_text(m, BOT_USERNAME) is None


def test_text_no_entities_not_addressed(make_message):
    assert extract_text(make_message("plain chatter"), BOT_USERNAME) is None


def test_text_no_text_not_addressed(make_message):
    assert extract_text(make_message(text=None), BOT_USERNAME) is None


def test_text_mention_is_not_prefix_match(make_message):
    # '@family_bot_x' is a different user: the entity covers the longer mention
    m = make_message("@family_bot_x hi", [mention_entity(0, 13)])
    assert extract_text(m, BOT_USERNAME) is None


def test_text_non_mention_entity_ignored(make_message):
    url = SimpleNamespace(type=MessageEntityType.URL, offset=0, length=4)
    assert extract_text(make_message(BOT_MENTION, [url]), BOT_USERNAME) is None


# --- extract_photo ---

def test_photo_caption_mention_addressed(make_message):
    caption = f"{BOT_MENTION} what is on this photo?"
    q = extract_photo(
        make_message(photo=True, caption=caption, caption_entities=mention_entities_for(caption)),
        BOT_USERNAME,
    )
    assert q is not None
    assert q.kind == "photo"
    assert q.text == "what is on this photo?"
    assert q.image is None  # payload is not attached yet


def test_photo_caption_with_emoji_offset(make_message):
    caption = f"🍜 {BOT_MENTION} guess"
    q = extract_photo(
        make_message(photo=True, caption=caption, caption_entities=mention_entities_for(caption)),
        BOT_USERNAME,
    )
    assert q is not None and q.text == "guess"


def test_photo_without_caption_not_addressed(make_message):
    assert extract_photo(make_message(photo=True), BOT_USERNAME) is None


def test_photo_caption_without_mention_not_addressed(make_message):
    assert extract_photo(make_message(photo=True, caption="just a photo"), BOT_USERNAME) is None


def test_photo_not_a_photo_not_addressed(make_message):
    assert extract_photo(make_message(text="not a photo"), BOT_USERNAME) is None


# --- dormant extractors ---

def test_audio_extractor_is_dormant(make_message):
    assert extract_audio(make_message(voice=True), BOT_USERNAME) is None


def test_sticker_extractor_is_dormant(make_message):
    assert extract_sticker(make_message(sticker=True), BOT_USERNAME) is None


# --- dispatch ---

def test_dispatch_text_first(make_message):
    text = f"{BOT_MENTION} hi"
    q = extract_question(make_message(text, mention_entities_for(text)), BOT_USERNAME)
    assert q is not None and q.kind == "text"


def test_dispatch_photo(make_message):
    caption = f"{BOT_MENTION} hi"
    q = extract_question(
        make_message(photo=True, caption=caption, caption_entities=mention_entities_for(caption)),
        BOT_USERNAME,
    )
    assert q is not None and q.kind == "photo"


def test_dispatch_unaddressed_is_none(make_message):
    for m in (
        make_message("plain chatter"),
        make_message(voice=True),
        make_message(sticker=True),
        make_message(photo=True, caption="no mention"),
    ):
        assert extract_question(m, BOT_USERNAME) is None
