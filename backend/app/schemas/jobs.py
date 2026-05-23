"""Pydantic-схемы для API. Только то что отдаётся наружу."""
from typing import Optional, List, Dict, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------- events


class Event(BaseModel):
    time: float = Field(..., description="Время удара в секундах")
    frame: int = Field(..., description="Номер кадра в исходном видео")
    fighter: Literal["red", "blue"]
    hand: Literal["left", "right"]
    punch_type: Literal["jab", "hook", "cross", "uppercut"]
    target: Literal["head", "body"]
    effectiveness: Literal["landed", "blocked", "miss"]
    confidence: float = Field(..., ge=0.0, le=1.0)


class FighterStats(BaseModel):
    punches: int
    accuracy: float
    landed: int = 0
    blocked: int = 0
    missed: int = 0


class FighterDNA(BaseModel):
    aggression: int = Field(..., ge=0, le=100)
    defense: int = Field(..., ge=0, le=100)
    mobility: int = Field(..., ge=0, le=100)
    pressure: int = Field(..., ge=0, le=100)


class AnalysisResult(BaseModel):
    video_url: str
    highlights_url: str
    events: List[Event]
    summary: Dict[Literal["red", "blue"], FighterStats]
    dna: Dict[Literal["red", "blue"], FighterDNA]


# ----------------------------------------------------------------------- jobs


class JobCreated(BaseModel):
    job_id: str
    status: Literal["queued"]


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "done", "failed"]
    stage: str
    progress: float = Field(..., ge=0.0, le=1.0)
    error: Optional[str] = None
