# FightVision AI

AI-разбор боксёрских поединков: загружаешь видео раунда — получаешь хайлайты с ударами и статистику.

Сделано на базе хакатонового решения RASCAR (YOLOv8-pose + event model + visual reranker + MLP reranker + rank blend).

## Архитектура

```
┌──────────┐  ┌──────────┐  ┌──────────────┐
│  Web UI  │  │ TG Bot   │  │ Mobile (TBD) │
│ (React)  │  │(aiogram3)│  │              │
└────┬─────┘  └─────┬────┘  └──────┬───────┘
     └──────────────┴──────────────┘
                    ▼
         ┌──────────────────┐
         │ Gateway (FastAPI)│  POST /api/analyze       → {job_id}
         │   тонкий слой    │  GET  /api/jobs/{id}     → status, progress
         └────────┬─────────┘  GET  /api/jobs/{id}/result
                  ▼
         ┌──────────────────┐
         │ Redis (queue +   │
         │  job state)      │
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ ML Worker (RQ)   │  pose → features → event model
         │ — где живёт      │  → visual rerank → mlp rerank
         │   torch и YOLO   │  → rank blend → topK+NMS
         └────────┬─────────┘  → attributes → overlay → highlights
                  ▼
         ┌──────────────────┐
         │ Storage (FS)     │  /media/uploads/{job}.mp4
         │                  │  /media/results/{job}/highlights.mp4
         └──────────────────┘  /media/results/{job}/result.json
```

Frontend и бот — два разных клиента **одного** API. Бизнес-логика не дублируется.

## Структура репозитория

```
fightvision-ai/
├── backend/        FastAPI gateway (лёгкий, без torch)
├── ml_worker/      ML-пайплайн (тяжёлый процесс с torch/YOLO/opencv)
│   ├── pipeline/   модули из ноутбука RASCAR, разнесённые по файлам
│   ├── visualization/  отрисовка bbox/лейблов + нарезка хайлайтов
│   └── models/     сюда кладутся обученные веса (.joblib/.pt)
├── telegram_bot/   aiogram 3
├── frontend/       React + Vite (уже был, доработан под поллинг)
├── docker-compose.yml
└── .env.example
```

## Быстрый старт (без Docker, локальная разработка)

Нужно: Python 3.10+, Node 18+, Redis.

### 1. Redis

```bash
# macOS
brew install redis && brew services start redis
# Linux
sudo apt install redis-server && sudo systemctl start redis
# или просто docker run -d -p 6379:6379 redis:7-alpine
```

### 2. Backend gateway

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env
uvicorn app.main:app --reload --port 8000
```

### 3. ML worker (в отдельном терминале)

```bash
cd ml_worker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# веса (опционально, иначе будет heuristic-fallback):
#   models/full_pipeline.joblib
#   models/visual_reranker.joblib
#   models/mlp_reranker.pt
python worker.py
```

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### 5. Telegram bot (опционально)

```bash
cd telegram_bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN=xxx:xxx
python bot.py
```

## Быстрый старт через Docker

```bash
cp .env.example .env
# впиши TELEGRAM_BOT_TOKEN если нужен бот
docker compose up --build
```

Поднимет: redis, backend, ml_worker, frontend (на :5173), telegram_bot.

## Режимы работы

ML worker имеет **два режима**:

1. **Полный режим (с весами).** Если в `ml_worker/models/` лежат `full_pipeline.joblib` + `visual_reranker.joblib` + `mlp_reranker.pt` — pipeline работает как в ноутбуке: YOLO → pose → event model → visual rerank → MLP rerank → blend.

2. **Heuristic fallback (без весов).** Если каких-то весов нет — недостающие этапы заменяются простой эвристикой по скорости wrist'ов. Это менее точно, но позволяет крутить весь стек end-to-end и тестировать UI.

Проверить какой режим активен:

```bash
curl http://127.0.0.1:8000/api/health
```

## API

| Метод | Путь                          | Описание                                                |
|-------|-------------------------------|---------------------------------------------------------|
| POST  | `/api/analyze`                | Загрузка видео, возвращает `{job_id, status: "queued"}` |
| GET   | `/api/jobs/{job_id}`          | Статус и прогресс job'а                                 |
| GET   | `/api/jobs/{job_id}/result`   | Финальный результат (только когда `status=done`)        |
| GET   | `/media/results/{job}/highlights.mp4` | Видео-нарезка с ударами                         |
| GET   | `/health`                     | Health check                                            |

Формат результата:

```json
{
  "video_url": "/media/results/{job}/highlights.mp4",
  "highlights_url": "/media/results/{job}/highlights.mp4",
  "events": [
    {"time": 11.2, "frame": 336, "fighter": "red", "hand": "left",
     "punch_type": "jab", "target": "head", "effectiveness": "landed",
     "confidence": 0.92}
  ],
  "summary": {
    "red":  {"punches": 23, "accuracy": 0.61, "landed": 14, "blocked": 5, "missed": 4},
    "blue": {"punches": 18, "accuracy": 0.53, "landed":  9, "blocked": 6, "missed": 3}
  },
  "dna": {
    "red":  {"aggression": 82, "defense": 47, "mobility": 64, "pressure": 73},
    "blue": {"aggression": 69, "defense": 58, "mobility": 71, "pressure": 62}
  }
}
```

## Как положить веса

После того как обученные веса появились (из тренировки в Colab/ноутбуке), кладёшь сюда:

```
ml_worker/models/
├── full_pipeline.joblib     # event_model + attr_models + feature_cols
├── visual_reranker.joblib   # PCA + CatBoost + ResNet18 weights ref
└── mlp_reranker.pt          # state_dict финального MLP
```

После этого worker автоматически переключится в полный режим, рестарт не нужен (`has_*_model()` проверяется на каждый job).

## TODO / roadmap

- [ ] Положить реальные веса и проверить end-to-end на тестовом видео
- [ ] Добавить визуализацию skeleton'а на видео (сейчас только bbox + лейбл удара)
- [ ] Добавить отдельный эндпоинт `/api/jobs/{job}/highlights` со списком отдельных клипов
- [ ] Telegram бот: отправка нарезки в виде нескольких коротких клипов с подписями
- [ ] Tauri/Electron обёртка для десктопа
- [ ] React Native клиент

## Лицензия

Внутренний проект.
