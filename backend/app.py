from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ml_service import analyze_video

MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="FightVision AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory="media"), name="media")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1]

    filename = f"{uuid4()}.{extension}"
    save_path = MEDIA_DIR / filename

    with open(save_path, "wb") as f:
        f.write(await file.read())

    result = analyze_video(str(save_path))

    return result
