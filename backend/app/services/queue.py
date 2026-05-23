"""
Thin wrapper around RQ.

Gateway не импортирует код ml_worker'а напрямую — он только кладёт задачу
в очередь по полному имени функции. Worker подбирает её и исполняет
в своём процессе со своим окружением (torch и т.д.).
"""
import redis
from rq import Queue

from app.core.config import settings

_redis_conn = redis.from_url(settings.redis_url)
_queue = Queue(settings.queue_name, connection=_redis_conn)


def enqueue_video_processing(job_id: str, video_path: str, *, timeout_seconds: int = 3600) -> None:
    """Поставить задачу обработки видео в очередь.

    Worker должен импортировать `ml_worker.tasks.process_video` по этому имени.
    """
    _queue.enqueue(
        "tasks.process_video",
        job_id,
        video_path,
        job_id=job_id,
        job_timeout=timeout_seconds,
        result_ttl=settings.job_ttl_seconds,
        failure_ttl=settings.job_ttl_seconds,
    )


def queue_health() -> dict:
    try:
        return {
            "connected": _redis_conn.ping(),
            "queue": settings.queue_name,
            "queued_jobs": _queue.count,
            "failed_jobs": _queue.failed_job_registry.count,
        }
    except Exception as exc:
        return {"connected": False, "error": str(exc)}
