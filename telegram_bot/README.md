# FightVision Telegram Bot

Бот на aiogram 3, который принимает видео и возвращает нарезку ударов + сводку.
Сам по себе модели не запускает — ходит в gateway (`backend/`).

## Локальный запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export TELEGRAM_BOT_TOKEN=xxx:xxx
export FV_BACKEND_URL=http://localhost:8000

python bot.py
```

## Команды

- `/start` — приветствие
- `/help` — справка
- `/health` — проверка связи с backend'ом (полезно когда бот молчит)

## Ограничения

- Telegram Bot API ограничивает файлы загрузкой до **50 МБ**. Если у юзера видео больше — попросим залить на сайт.
- Опрос статуса работает до `FV_BOT_POLL_TIMEOUT` секунд (по умолчанию 15 минут).

## Архитектура

```
User → [video] → bot.py
                  ↓
              api_client.py
                  ↓
              POST /api/analyze → {job_id}
                  ↓
              GET /api/jobs/{id} (poll) → progress
                  ↓
              GET /api/jobs/{id}/result
                  ↓
              GET /media/.../highlights.mp4
                  ↓
User ← [highlights.mp4 + сводка]
```

То есть в боте нет ни pyTorch, ни YOLO, ни opencv — только httpx и aiogram.
