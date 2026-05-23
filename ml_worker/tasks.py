"""
RQ-таска, которую дёргает воркер. Импортируется по имени ``tasks.process_video``.
Здесь же мы дублируем минимальный JobManager — чтобы воркеру не нужно было
импортировать код gateway'а (могут быть разные виртуалки).
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone

import redis

from config import ml_settings
from pipeline.inference import run_inference


# ----------------------------------------------- мини-JobManager на стороне worker'а


_redis = redis.from_url(ml_settings.redis_url, decode_responses=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _load(job_id: str) -> dict | None:
    raw = _redis.get(_job_key(job_id))
    return json.loads(raw) if raw else None


def _save(job_id: str, job: dict) -> None:
    _redis.setex(
        _job_key(job_id),
        ml_settings.job_ttl_seconds,
        json.dumps(job, ensure_ascii=False, default=str),
    )


def _update(job_id: str, **kwargs) -> None:
    job = _load(job_id)
    if job is None:
        return
    job.update(kwargs)
    job["updated_at"] = _now()
    _save(job_id, job)


# ----------------------------------------------- сама таска


def process_video(job_id: str, video_path: str) -> dict:
    """Главный entry-point worker'а. Обрабатывает одно видео и пишет результат в Redis."""
    print(f"[worker] start job={job_id} video={video_path}")

    def progress(stage: str, value: float) -> None:
        _update(
            job_id,
            status="processing",
            stage=stage,
            progress=float(max(0.0, min(1.0, value))),
        )

    try:
        result = run_inference(
            video_path=video_path,
            job_id=job_id,
            progress_callback=progress,
        )
        _update(
            job_id,
            status="done",
            stage="finished",
            progress=1.0,
            result=result,
            error=None,
        )
        print(f"[worker] done job={job_id} events={len(result.get('events', []))}")
        return result
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        print(f"[worker] FAILED job={job_id}: {err_msg}")
        traceback.print_exc()
        _update(
            job_id,
            status="failed",
            stage="error",
            error=err_msg,
        )
        raise
