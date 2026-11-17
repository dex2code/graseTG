"""Shared test fixtures: fake Telegram objects, a router under test, a stub LLM server."""

import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
from aiogram.enums import MessageEntityType
from pydantic import SecretStr

from grase_tg.config import AppConfig, Config, Secrets
from grase_tg.handlers import build_router

BOT_USERNAME = "family_bot"
BOT_MENTION = f"@{BOT_USERNAME}"


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


class FakeAck:
    """Records edit_text / reply calls, like the aiogram Message returned by reply()."""

    def __init__(self) -> None:
        self.edits: list[str] = []
        self.replies: list[str] = []

    async def edit_text(self, text: str) -> None:
        self.edits.append(text)

    async def reply(self, text: str) -> None:
        self.replies.append(text)


class FakeMessage:
    """Duck-typed message mimicking the fields the bot touches. No network involved."""

    def __init__(
        self,
        text: str | None = None,
        entities: list | None = None,
        chat_id: int = -100,
        photo: bool = False,
        caption: str | None = None,
        caption_entities: list | None = None,
        voice: bool = False,
        sticker: bool = False,
    ) -> None:
        self.chat = SimpleNamespace(id=chat_id, title="Test Group", full_name="Test User")
        self.text = text
        self.entities = entities
        self.photo = object() if photo else None
        self.caption = caption
        self.caption_entities = caption_entities
        self.voice = object() if voice else None
        self.sticker = object() if sticker else None
        self.sent: list[str] = []
        self.acks: list[FakeAck] = []

    async def reply(self, text: str) -> FakeAck:
        self.sent.append(text)
        ack = FakeAck()
        self.acks.append(ack)
        return ack


@pytest.fixture
def make_message():
    """Factory fixture for FakeMessage."""
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
def handler(config):
    """The inner message handler extracted from the built router."""
    router = build_router(config, llm_client=None, bot_username=BOT_USERNAME)
    return router.observers["message"].handlers[0].callback


class StubLLMServer:
    """A local OpenAI-compatible /chat/completions endpoint that captures request bodies."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.reply_content = "ok"
        self._server = HTTPServer(("127.0.0.1", 0), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def _make_handler(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                stub.requests.append(json.loads(body))
                resp = json.dumps(
                    {
                        "id": "x",
                        "object": "chat.completion",
                        "created": 0,
                        "model": stub.requests[-1].get("model", "m"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": stub.reply_content},
                                "finish_reason": "stop",
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

            def log_message(self, *args):
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
