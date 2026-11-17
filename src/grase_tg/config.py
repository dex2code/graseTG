"""Configuration loading: secrets from .env + settings from config.yaml -> one Config object."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr

# Project root: src/grase_tg/config.py -> parents[2]
BASE_DIR = Path(__file__).resolve().parents[2]


class Secrets(BaseModel):
    """Secrets, collected from the OS environment and the project-root .env file."""

    bot_token: SecretStr
    llm_api_key: SecretStr
    llm_base_url: str
    # Optional HTTP proxy for the Telegram API. Empty value = direct connection.
    telegram_proxy_url: str = ""


class LlmSettings(BaseModel):
    """Parameters for calls to the OpenAI-compatible backend."""

    model: str = "gpt-5-mini"
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    # Model "thinking" toggle, sent as chat_template_kwargs={"enable_thinking": ...}
    # (vLLM/SGLang, Qwen3-style models).
    # true / false — send the field; None — do not send it at all
    # (safe for vanilla OpenAI-compatible APIs like api.openai.com).
    enable_thinking: bool | None = None


class AppConfig(BaseModel):
    """Non-secret settings from config.yaml."""

    allowed_chat_ids: list[int] = Field(default_factory=list)
    system_prompt: str = "You are an assistant in a family group chat. Answer in User's language: briefly, clearly and to the point."
    llm: LlmSettings = LlmSettings()


class Config(BaseModel):
    """Final application configuration."""

    secrets: Secrets
    app: AppConfig


def _read_secrets(base_dir: Path) -> Secrets:
    """Collect secret values: OS environment wins over the .env file.

    An empty OS environment variable counts as unset (falls back to .env).
    """
    file_values = dotenv_values(base_dir / ".env")
    values: dict[str, str] = {}
    for field in Secrets.model_fields:
        # Field names map to UPPER_CASE environment variable names.
        value = os.environ.get(field.upper()) or file_values.get(field.upper())
        if value is not None:
            values[field] = value
    return Secrets.model_validate(values)


def load_config(base_dir: Path | None = None) -> Config:
    """Load settings from the given directory (defaults to the project root)."""
    base = base_dir or BASE_DIR
    raw: dict[str, Any] = {}
    yaml_path = base / "config.yaml"
    if yaml_path.exists():
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return Config(secrets=_read_secrets(base), app=AppConfig.model_validate(raw))
