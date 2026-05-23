"""GET /api/jobs/{id} и /api/jobs/{id}/result."""
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.schemas import JobStatus
from app.services.job_manager import job_manager

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        stage=job.get("stage", "unknown"),
        progress=float(job.get("progress", 0.0)),
        error=job.get("error"),
    )


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> Dict[str, Any]:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=job.get("error") or "Processing failed",
        )

    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job not ready (status={job['status']})",
        )

    result = job.get("result")
    if result is None:
        raise HTTPException(status_code=500, detail="Job is marked done but has no result")

    return result
