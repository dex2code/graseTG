"""Configuration loading: secrets from .env + settings from config.yaml -> one Config object."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: src/grase_tg/config.py -> parents[2]
BASE_DIR = Path(__file__).resolve().parents[2]


class Secrets(BaseSettings):
    """Secrets, read from .env in the project root (not committed to git)."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

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
    # (supported by vLLM/SGLang and Qwen3-style models).
    enable_thinking: bool = False


class AppConfig(BaseModel):
    """Non-secret settings from config.yaml."""

    allowed_chat_ids: list[int] = Field(default_factory=list)
    system_prompt: str = "You are an assistant in a family group chat. Answer in User's language: briefly, clearly and to the point."
    llm: LlmSettings = LlmSettings()


class Config(BaseModel):
    """Final application configuration."""

    secrets: Secrets
    app: AppConfig


def load_config() -> Config:
    raw: dict[str, Any] = {}
    yaml_path = BASE_DIR / "config.yaml"
    if yaml_path.exists():
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    # pydantic-settings fills Secrets from .env at runtime, but pyright cannot see
    # that and reports the missing constructor arguments — a known false positive.
    return Config(secrets=Secrets(), app=AppConfig.model_validate(raw))  # pyright: ignore[reportCallIssue]
