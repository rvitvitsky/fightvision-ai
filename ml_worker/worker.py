"""
RQ worker. Слушает очередь fightvision и исполняет ``tasks.process_video``.

Запуск:
    python worker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import redis
from rq import Worker, Queue

# Делаем себя импортируемым: tasks.py и pipeline/ в одном каталоге
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import ml_settings  # noqa: E402


def main() -> None:
    conn = redis.from_url(ml_settings.redis_url)
    queue = Queue(ml_settings.queue_name, connection=conn)

    print(f"[worker] connected to {ml_settings.redis_url}")
    print(f"[worker] queue={ml_settings.queue_name}")
    print(f"[worker] models_dir={ml_settings.models_path}")
    print(f"[worker] media_dir={ml_settings.media_path}")
    print(f"[worker] yolo_model={ml_settings.yolo_pose_model}")

    worker = Worker([queue], connection=conn)
    worker.work(with_scheduler=False, logging_level="INFO")


if __name__ == "__main__":
    main()
