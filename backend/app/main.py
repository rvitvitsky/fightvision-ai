"""FightVision AI — FastAPI gateway."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import analyze, jobs, health
from app.core.config import settings


def create_app() -> FastAPI:
    application = FastAPI(
        title="FightVision AI",
        version="0.2.0",
        description="AI-разбор боксёрских поединков. Загружаешь видео — получаешь хайлайты и статистику.",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    media_path = Path(settings.media_dir)
    media_path.mkdir(parents=True, exist_ok=True)
    (media_path / "uploads").mkdir(exist_ok=True)
    (media_path / "results").mkdir(exist_ok=True)
    application.mount("/media", StaticFiles(directory=str(media_path)), name="media")

    application.include_router(health.router)
    application.include_router(analyze.router, prefix="/api", tags=["analyze"])
    application.include_router(jobs.router, prefix="/api", tags=["jobs"])

    return application


app = create_app()
