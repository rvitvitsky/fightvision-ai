"""
JobManager — обёртка над Redis для хранения состояния обработки видео.

Каждый job хранится по ключу `job:{uuid}` как JSON.
TTL ограничивает сколько Redis держит запись (по умолчанию сутки).
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis

from app.core.config import settings


class JobManager:
    """Простой синхронный клиент для отслеживания jobs.

    Используется одновременно gateway'ем (создание, чтение)
    и ml_worker'ом (обновление прогресса, запись результата).
    """

    def __init__(self) -> None:
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)

    # ---- internal ----------------------------------------------------------

    @staticmethod
    def _key(job_id: str) -> str:
        return f"job:{job_id}"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ---- public API --------------------------------------------------------

    def create_job(self, job_id: str, video_filename: str) -> Dict[str, Any]:
        job = {
            "job_id": job_id,
            "status": "queued",        # queued | processing | done | failed
            "stage": "queued",         # human-readable stage name
            "progress": 0.0,           # 0.0 — 1.0
            "video_filename": video_filename,
            "created_at": self._now(),
            "updated_at": self._now(),
            "error": None,
            "result": None,
        }
        self._save(job_id, job)
        return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        raw = self._redis.get(self._key(job_id))
        return json.loads(raw) if raw else None

    def update_job(self, job_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        job = self.get_job(job_id)
        if job is None:
            return None
        job.update(kwargs)
        job["updated_at"] = self._now()
        self._save(job_id, job)
        return job

    def delete_job(self, job_id: str) -> None:
        self._redis.delete(self._key(job_id))

    # ---- helpers -----------------------------------------------------------

    def _save(self, job_id: str, job: Dict[str, Any]) -> None:
        self._redis.setex(
            self._key(job_id),
            settings.job_ttl_seconds,
            json.dumps(job, ensure_ascii=False, default=str),
        )

    def ping(self) -> bool:
        try:
            return bool(self._redis.ping())
        except Exception:
            return False


job_manager = JobManager()
