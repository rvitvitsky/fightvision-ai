"""GET /health — проверка что gateway жив и видит Redis."""
from fastapi import APIRouter

from app.services.job_manager import job_manager
from app.services.queue import queue_health

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "redis": job_manager.ping(),
        "queue": queue_health(),
    }
