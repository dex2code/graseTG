"""Загрузка конфигурации: секреты из .env + настройки из config.yaml -> один объект Config."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Secrets(BaseSettings):
    """Секреты. Читаются из .env в корне проекта (в git не коммитится)."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: str
    llm_api_key: str
    llm_base_url: str
    # HTTP-прокси для Telegram API, опционально. Пусто — прямое подключение.
    telegram_proxy_url: str = ""


class LlmSettings(BaseModel):
    """Параметры обращения к OpenAI-совместимому backend'у."""

    model: str = "gpt-5-mini"
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_seconds: float = 60.0


class AppConfig(BaseModel):
    """Настройки из config.yaml (несекретные)."""

    allowed_chat_ids: list[int] = Field(default_factory=list)
    system_prompt: str = "Ты — полезный ассистент. Отвечай по-русски, кратко и по делу."
    llm: LlmSettings = LlmSettings()


class Config(BaseModel):
    """Итоговая конфигурация приложения."""

    secrets: Secrets
    app: AppConfig


def load_config() -> Config:
    raw: dict[str, Any] = {}
    yaml_path = BASE_DIR / "config.yaml"
    if yaml_path.exists():
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return Config(secrets=Secrets(), app=AppConfig.model_validate(raw))
