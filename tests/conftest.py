"""Shared test fixtures: fake Telegram objects, a router under test, a stub LLM server."""

import json
import threading
from typing import Any, Callable, cast
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
from aiogram.enums import MessageEntityType
from pydantic import SecretStr

from grase_tg.config import AppConfig, Config, Secrets
from grase_tg.content import BotIdentity
from grase_tg.handlers import build_router
from grase_tg.llm import make_client

BOT_USERNAME = "family_bot"
BOT_MENTION = f"@{BOT_USERNAME}"
BOT_USER_ID = 4242  # the bot's Telegram user id in tests
BOT = BotIdentity(username=BOT_USERNAME, user_id=BOT_USER_ID)


def mention_entity(offset: int = 0, length: int = 4) -> SimpleNamespace:
    """A duck-typed MENTION entity, like the real MessageEntity."""
    return SimpleNamespace(type=MessageEntityType.MENTION, offset=offset, length=length)


def mention_entities_for(text: str, username: str = BOT_USERNAME) -> list:
    """Build MENTION entities for every @username in text, with real Telegram semantics:
    offsets in UTF-16 code units (emoji count as two units)."""
    target = f"@{username}".lower()
    lowered = text.lower()
    entities: list = []
    pos = 0
    while (idx := lowered.find(target, pos)) != -1:
        utf16_offset = len(text[:idx].encode("utf-16-le")) // 2
        entities.append(mention_entity(utf16_offset, len(target)))
        pos = idx + len(target)
    return entities


def bot_message(message_id: int = 1, text: str = "", caption: str = "") -> SimpleNamespace:
    """A duck-typed message authored by the bot, for use as reply_to_message."""
    return SimpleNamespace(message_id=message_id, from_user=SimpleNamespace(id=BOT_USER_ID), text=text, caption=caption)


def human_message(message_id: int = 1, text: str = "", caption: str = "") -> SimpleNamespace:
    """A duck-typed message authored by some other user."""
    return SimpleNamespace(message_id=message_id, from_user=SimpleNamespace(id=999), text=text, caption=caption)


class FakeAck:
    """Records edit_text / reply calls, like the aiogram Message returned by reply().

    The fail_edit / fail_reply switches (set per test, usually via monkeypatch
    on the class) make the corresponding call raise, to exercise failure paths.
    """

    fail_edit = False
    fail_reply = False

    def __init__(self) -> None:
        self.edits: list[str] = []
        self.replies: list[str] = []

    async def edit_text(self, text: str) -> None:
        if self.fail_edit:
            raise RuntimeError("edit rejected")
        self.edits.append(text)

    async def reply(self, text: str) -> None:
        if self.fail_reply:
            raise RuntimeError("reply rejected")
        self.replies.append(text)


class FakeMessage:
    """Duck-typed message mimicking the fields the bot touches. No network involved."""

    def __init__(
        self,
        text: str | None = None,
        entities: list | None = None,
        chat_id: int = -100,
        photo: bool = False,
        document: bool = False,
        video: bool = False,
        animation: bool = False,
        caption: str | None = None,
        caption_entities: list | None = None,
        voice: bool = False,
        audio: bool = False,
        video_note: bool = False,
        sticker: bool = False,
        reply_to_message: Any = None,
        is_topic_message: bool = False,
        message_thread_id: int | None = None,
    ) -> None:
        self.chat = SimpleNamespace(id=chat_id, title="Test Group", full_name="Test User")
        self.text = text
        self.entities = entities
        self.photo = object() if photo else None
        self.document = object() if document else None
        self.video = object() if video else None
        self.animation = object() if animation else None
        self.caption = caption
        self.caption_entities = caption_entities
        self.voice = object() if voice else None
        self.audio = object() if audio else None
        self.video_note = object() if video_note else None
        self.sticker = object() if sticker else None
        self.reply_to_message = reply_to_message
        self.is_topic_message = is_topic_message
        self.message_thread_id = message_thread_id
        self.sent: list[str] = []
        self.acks: list[FakeAck] = []

    async def reply(self, text: str) -> FakeAck:
        self.sent.append(text)
        ack = FakeAck()
        self.acks.append(ack)
        return ack


@pytest.fixture
def make_message() -> Callable[..., Any]:
    """Factory for duck-typed messages. Typed as Any: fakes intentionally
    implement only the attributes the bot touches, so strict Message typing
    does not apply to them."""
    return FakeMessage


@pytest.fixture
def config() -> Config:
    """An app config with one allowed group and no secrets."""
    return Config(
        secrets=Secrets(
            bot_token=SecretStr("test-token"),
            llm_api_key=SecretStr("test-key"),
            llm_base_url="http://llm.test/v1",
        ),
        app=AppConfig(allowed_chat_ids=[-100]),
    )


@pytest.fixture
def env_isolated(monkeypatch):
    """Hide real environment variables so tests are hermetic."""
    for name in (
        "BOT_TOKEN",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "TELEGRAM_PROXY_URL",
        "GRASE_TG_CONFIG_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def handler(config):
    """The inner message handler extracted from the built router."""
    # A real client is fine here: every test that reaches ask_llm monkeypatches it.
    router = build_router(config, make_client(config), BOT)
    return router.observers["message"].handlers[0].callback


class StubLLMServer:
    """A local OpenAI-compatible /chat/completions endpoint that captures request bodies.

    Default: answers with `reply_content`. For tool-loop tests, `script`
    holds raw assistant message dicts popped one per request (e.g. a
    tool_calls message followed by a plain-content message). A dict may
    carry "_finish_reason" ("length" etc.) — popped before serialization,
    never sent inside the message."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.reply_content = "ok"
        self.script: list[dict] = []
        self._server = HTTPServer(("127.0.0.1", 0), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = cast(tuple[str, int], self._server.server_address)
        return f"http://{host}:{port}/v1"

    def _make_handler(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                stub.requests.append(json.loads(body))
                if stub.script:
                    message = stub.script.pop(0)
                    finish_reason = message.pop("_finish_reason", None) or (
                        "tool_calls" if message.get("tool_calls") else "stop"
                    )
                else:
                    message = {"role": "assistant", "content": stub.reply_content}
                    finish_reason = "stop"
                resp = json.dumps(
                    {
                        "id": "x",
                        "object": "chat.completion",
                        "created": 0,
                        "model": stub.requests[-1].get("model", "m"),
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": finish_reason,
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - base class name
                pass  # keep test output clean

        return Handler

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def llm_stub():
    stub = StubLLMServer()
    yield stub
    stub.stop()


@pytest.fixture
def config_with_stub(config, llm_stub):
    """Config whose LLM endpoint points at the local stub server."""
    config.secrets.llm_base_url = llm_stub.base_url
    return config, llm_stub


class StubToolRegistry:
    """Offline ToolRegistry for the tool loop: a fixed schema and canned
    execute() results, recording every (name, arguments) call."""

    def __init__(self, schema: list[dict] | None = None, results: list[str] | None = None) -> None:
        self._schema = list(schema or [])
        self._results = list(results or [])
        self.calls: list[tuple[str, str]] = []

    def openai_schema(self) -> list[dict]:
        return list(self._schema)

    async def execute(self, name: str, arguments_json: str) -> str:
        self.calls.append((name, arguments_json))
        return self._results.pop(0) if self._results else "stub tool result"
