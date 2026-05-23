"""Конфиг Telegram-бота. Читает токен и адрес backend'а из переменных окружения."""
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

    fv_bot_poll_timeout: int = 900   # 15 минут
    fv_bot_poll_interval: int = 4    # каждые 4 секунды

    # Telegram-лимит на upload через bot api = 50 MB
    max_video_mb: int = 49


bot_settings = BotSettings()
