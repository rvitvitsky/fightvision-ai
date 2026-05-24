"""Конфиг Telegram-бота. Все секреты читаются из переменных окружения/.env."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    telegram_bot_token: str = ""
    fv_backend_url: str = "http://localhost:8000"

    fv_bot_poll_timeout: int = 1800   # 30 минут: 10-минутные видео на CPU могут считаться долго
    fv_bot_poll_interval: int = 4     # каждые 4 секунды
    fv_bot_max_video_mb: int = 49     # лимит безопасной загрузки через Telegram Bot API
    fv_bot_max_result_mb: int = 49    # если нарезка больше — бот пришлёт сводку и попросит скачать с сайта


bot_settings = BotSettings()
