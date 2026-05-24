# FightVision AI — локальный запуск

## Что уже включено в этот архив

В `ml_worker/models/` уже положены веса из загруженных файлов:

- `full_pipeline.joblib`
- `visual_reranker.joblib`
- `visual_reranker.cbm`
- `visual_pca.joblib`
- `mlp_reranker.pt`
- `mlp_scaler.joblib`
- `inference_config.json`

Код worker'а поправлен под фактический формат этих артефактов: notebook `AttributeModels`, отдельный PCA/scaler, CatBoost как raw model, MLP с BatchNorm/512/256/128.

## Самый простой запуск сайта через Docker

```bash
cd fightvision-ai
cp .env.example .env
docker compose up --build
```

Открой: http://localhost:5173

Проверка backend:

```bash
curl http://127.0.0.1:8000/health
```

## Запуск Telegram-бота через Docker

В `.env` впиши токен:

```env
TELEGRAM_BOT_TOKEN=123456:ABC...
```

Запуск всего вместе с ботом:

```bash
docker compose --profile bot up --build
```

## Локальный запуск без Docker

### 1. Redis

```bash
docker run -d --name fv_redis -p 6379:6379 redis:7-alpine
```

### 2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. ML worker

```bash
cd ml_worker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python worker.py
```

### 4. Frontend

```bash
cd frontend
npm ci
npm run dev
```

Открой: http://localhost:5173

### 5. Telegram bot

```bash
cd telegram_bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN=123456:ABC...
python bot.py
```

## Важные замечания

- Первый запуск `ml_worker` может быть долгим: ставится/инициализируется PyTorch, Ultralytics и YOLO-pose модель.
- Если YOLO не сможет скачать `yolov8n-pose.pt`, скачай файл отдельно и положи его в `ml_worker/` или укажи путь через `FV_YOLO_POSE_MODEL`.
- Telegram Bot API принимает файлы до ~50 МБ, поэтому бот ограничен 49 МБ. Для больших видео используй сайт.
- Если запускаешь на CPU, обработка видео может быть медленной. Для демо лучше короткие ролики 10–30 секунд.
