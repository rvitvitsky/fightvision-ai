from pydantic import BaseModel


class Event(BaseModel):
    time: float
    frame: int
    fighter: str
    hand: str
    punch_type: str
    target: str
    confidence: float


class FighterStats(BaseModel):
    punches: int
    accuracy: float


class FighterDNA(BaseModel):
    aggression: int
    defense: int
    mobility: int
    pressure: int


class AnalysisResponse(BaseModel):
    video_url: str
    events: list[Event]
    summary: dict
    dna: dict