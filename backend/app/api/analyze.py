"""POST /api/analyze — приём видео, создание job, постановка в очередь."""
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.schemas import JobCreated
from app.services.job_manager import job_manager
from app.services.queue import enqueue_video_processing

router = APIRouter()

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}


@router.post("/analyze", response_model=JobCreated, status_code=202)
async def analyze(file: UploadFile = File(...)) -> JobCreated:
    """Принимает видео, кладёт на диск и в очередь, отдаёт job_id.

    Реальная обработка идёт асинхронно в ml_worker'е.
    Опрашивай статус через GET /api/jobs/{job_id}.
    """
    filename = file.filename or ""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Стримим файл, не загружая всё в память сразу
    job_id = str(uuid4())
    saved_filename = f"{job_id}.{extension}"
    save_path: Path = settings.uploads_dir / saved_filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    with save_path.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                f.close()
                save_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: limit is {settings.max_upload_mb} MB",
                )
            f.write(chunk)

    if written == 0:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty file")

    job_manager.create_job(job_id, saved_filename)
    enqueue_video_processing(job_id, str(save_path.resolve()))

    return JobCreated(job_id=job_id, status="queued")
