# FightVision Telegram Bot

Бот на aiogram 3 принимает видео, отправляет его в FastAPI backend, ждёт обработку, возвращает нарезку ударов и сводку.

## Docker-запуск вместе с сайтом

1. В корне проекта создай `.env` из `.env.example`.
2. Впиши `TELEGRAM_BOT_TOKEN`.
3. Запусти весь стек с bot-профилем:

```bash
docker compose --profile bot up --build
```

Если сайт/worker уже собраны, быстрее:

```bash
docker compose --profile bot build telegram_bot
docker compose --profile bot up
```

## Проверка

В Telegram открой своего бота и отправь:

```text
/start
/health
```

`/health` должен ответить, что backend на связи.

## Команды

- `/start` — приветствие
- `/help` — справка
- `/health` — проверка связи с backend

## Ограничения

- Видео через Telegram лучше держать до 49 МБ.
- Если нарезка получится слишком большой, бот пришлёт сводку и попросит забрать файл через сайт/папку `media/results`.
- Для больших 5–10 минутных видео удобнее сайт: Telegram будет долго ждать и может упереться в лимиты размера.
