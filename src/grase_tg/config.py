"""Configuration loading: secrets from .env + settings from config.yaml -> one Config object."""

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

# Where the bot looks for .env and config.yaml: $GRASE_TG_CONFIG_DIR when set,
# otherwise the working directory. Deliberately NOT the installed package
# location: an installed wheel must not go looking for secrets in site-packages.
CONFIG_DIR_ENV = "GRASE_TG_CONFIG_DIR"


def default_config_dir() -> Path:
    """Directory holding .env and config.yaml: $GRASE_TG_CONFIG_DIR, else the CWD."""
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.cwd()


class Secrets(BaseModel):
    """Secrets, collected from the OS environment and the project-root .env file."""

    # min_length: an empty value (e.g. "BOT_TOKEN=" in .env) must fail at
    # startup, not as a cryptic error on the first request.
    bot_token: SecretStr = Field(min_length=1)  # bot token from @BotFather
    llm_api_key: SecretStr = Field(min_length=1)  # key for the OpenAI-compatible API
    llm_base_url: str = Field(min_length=1)  # API base URL, usually ends with /v1
    # Optional HTTP/SOCKS proxy for the Telegram API. Empty value = direct connection.
    telegram_proxy_url: str = ""


class LlmSettings(BaseModel):
    """Parameters for calls to the OpenAI-compatible backend."""

    # Unknown keys are a config typo: fail at startup instead of silently
    # keeping the default for that setting.
    model_config = ConfigDict(extra="forbid")

    model: str = Field(default="gpt-5-mini", min_length=1)  # model name on the backend
    temperature: float = Field(default=0.7, ge=0.0)  # creativity: 0 = deterministic, higher = more varied
    max_tokens: int = Field(default=1024, gt=0)  # upper bound for a single answer, in tokens
    timeout_seconds: float = Field(default=60.0, gt=0)  # per-request HTTP timeout
    # Model "thinking" toggle, sent as chat_template_kwargs={"enable_thinking": ...}
    # (vLLM/SGLang, Qwen3-style models).
    # true / false — send the field; None — do not send it at all
    # (safe for vanilla OpenAI-compatible APIs like api.openai.com).
    enable_thinking: bool | None = None
    # Sampling overrides, None = the field is not sent at all: the backend
    # then applies its own defaults (values baked in here would silently
    # change behavior for anyone who has not set them). Both are standard
    # OpenAI fields — vanilla-API safe, sent as typed kwargs.
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)  # nucleus sampling cutoff
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    # Max tool-call rounds per question (docs/mcp-tools-spec.md §4): the model
    # may request tool executions up to this many times. ge=1 on purpose: 0
    # would mean "advertise tools, but never run them" — a trap. Tools on/off
    # is governed solely by the tools: section below.
    tool_rounds: int = Field(default=2, ge=1)


class MessagesSettings(BaseModel):
    """User-facing texts of the bot — translate them to your language in config.yaml.

    Every value must be non-empty. format_hint is a template: {username} is
    replaced with the bot's @username; unknown placeholders fail at startup
    (see the validator below) instead of crashing the handler at runtime.
    """

    # Unknown keys are a config typo: fail at startup instead of silently
    # keeping the default for that setting.
    model_config = ConfigDict(extra="forbid")

    ack: str = Field(default="Got it, thinking…", min_length=1)
    error: str = Field(
        default="Failed to get a response from the model. Please try again later.",
        min_length=1,
    )
    format_hint: str = Field(
        default=(
            "To ask a question, write it in the same message after the mention:\n"
            '"@{username} how do I cook borscht?"\n'
            "Or simply reply to any of my messages."
        ),
        min_length=1,
    )
    media_stub: str = Field(
        default="I can see the attachment, but I can't analyze it yet — ask about it with text.",
        min_length=1,
    )
    audio_stub: str = Field(
        default="Voice and audio messages are not supported yet — please type your question.",
        min_length=1,
    )
    sticker_stub: str = Field(
        default="Nice sticker! I can't reply to stickers meaningfully yet.",
        min_length=1,
    )
    other_stub: str = Field(
        default="This message type is not supported yet — ask me with a text @mention.",
        min_length=1,
    )
    empty_answer: str = Field(default="The model returned an empty response.", min_length=1)
    # The tool-call budget (llm.tool_rounds) ran out while the model kept
    # asking for tools — an honest failure notice, not a silent stop.
    tool_failure: str = Field(
        default="I couldn't fetch the external data needed to answer this. Please try again later.",
        min_length=1,
    )

    @model_validator(mode="after")
    def _check_format_hint_placeholders(self) -> "MessagesSettings":
        """format_hint is .format()-ed with username={bot username}: an unknown
        placeholder must fail at startup, not raise KeyError in the handler."""
        try:
            self.format_hint.format(username="x")
        except (KeyError, IndexError) as error:
            raise ValueError(f"messages.format_hint has an unknown placeholder: {error}") from error
        return self


class McpSettings(BaseModel):
    """The single MCP server (streamable HTTP) exposing tools to the model.

    Architecture (docs/mcp-tools-spec.md §1.1): exactly one server. There is
    deliberately NO list here — describing a second server must be impossible
    in the config, not merely unhandled. Tool names are passed to the model
    as the server publishes them (no rename, no prefixes).

    The token value lives in .env; the config holds only the env var NAME
    (auth_env), read by the bridge at startup. Secrets never enter config.yaml.
    """

    # Unknown keys are a config typo: fail at startup instead of silently
    # keeping the default for that setting.
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)  # streamable HTTP endpoint of the MCP server
    # Name of the environment variable holding the bearer token (.env).
    # Empty = connect without auth.
    auth_env: str = ""
    timeout_seconds: float = Field(default=30.0, gt=0)  # per call_tool request
    # Optional whitelist of tool names; empty = expose every discovered tool.
    # Every tool description goes into every request — filtering saves tokens.
    allow: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_url_is_http(self) -> "McpSettings":
        """A non-http(s) url can only be a typo (the bridge speaks streamable
        HTTP): fail at startup, not on the first connection attempt."""
        parts = urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"tools.mcp.url must be an http(s) URL, got {self.url!r}")
        return self


class ToolsSettings(BaseModel):
    """External tools for the LLM. Absent/empty = tools are fully disabled:
    requests carry no `tools` field at all and the bot behaves exactly as
    it did before tools existed (docs/mcp-tools-spec.md §7)."""

    # Unknown keys are a config typo: fail at startup instead of silently
    # ignoring the misspelled setting.
    model_config = ConfigDict(extra="forbid")

    mcp: McpSettings | None = None


class AppConfig(BaseModel):
    """Non-secret settings from config.yaml."""

    # Unknown keys are a config typo: fail at startup instead of silently
    # ignoring the misspelled setting (see LlmSettings too).
    model_config = ConfigDict(extra="forbid")

    # Whitelisted group chat ids; messages from other chats are ignored.
    allowed_chat_ids: list[int] = Field(default_factory=list)
    # Default used when config.yaml omits system_prompt.
    system_prompt: str = (
        "You are an assistant in a family group chat. "
        "Answer in User's language: briefly, clearly and to the point."
    )
    llm: LlmSettings = LlmSettings()
    messages: MessagesSettings = MessagesSettings()
    tools: ToolsSettings = ToolsSettings()


class Config(BaseModel):
    """Final application configuration."""

    secrets: Secrets
    app: AppConfig


def _read_secrets(base_dir: Path) -> Secrets:
    """Collect secret values: OS environment wins over the .env file.

    An empty OS environment variable counts as unset (falls back to .env).
    """
    # interpolate=False: secrets are taken literally. With dotenv's default
    # interpolation ON, a value containing $VAR or ${VAR} would be silently
    # rewritten (expanded or emptied) and fail later as a cryptic auth error.
    file_values = dotenv_values(base_dir / ".env", interpolate=False)
    values: dict[str, str] = {}
    for field in Secrets.model_fields:
        # Field names map to UPPER_CASE environment variable names.
        value = os.environ.get(field.upper()) or file_values.get(field.upper())
        if value is not None:
            values[field] = value
    return Secrets.model_validate(values)


def load_config(base_dir: Path | None = None) -> Config:
    """Load settings from the given directory (defaults to default_config_dir())."""
    base = base_dir or default_config_dir()
    raw: dict[str, Any] = {}
    yaml_path = base / "config.yaml"
    if yaml_path.exists():
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return Config(secrets=_read_secrets(base), app=AppConfig.model_validate(raw))
