"""Конфиг ML worker'а. Грузится из тех же FV_* переменных что и gateway."""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class MLSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FV_",
        case_sensitive=False,
        extra="ignore",
    )

    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "fightvision"

    media_dir: str = "./media"
    models_dir: str = "./models"

    # YOLO pose
    yolo_pose_model: str = "yolov8n-pose.pt"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.50
    frame_stride: int = 1  # обрабатывать каждый N-й кадр

    # Highlights
    highlight_pad_before: float = 0.6  # секунды до удара
    highlight_pad_after: float = 0.6   # секунды после удара

    # NMS / post-processing
    nms_distance_frames: int = 10
    peak_threshold: float = 0.35
    peak_distance_frames: int = 12
    smooth_window: int = 5

    job_ttl_seconds: int = 60 * 60 * 24

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir).expanduser().resolve()

    @property
    def media_path(self) -> Path:
        return Path(self.media_dir).expanduser().resolve()

    def model_file(self, name: str) -> Optional[Path]:
        """Вернёт абсолютный путь до файла модели если он существует, иначе None."""
        p = self.models_path / name
        return p if p.exists() else None


ml_settings = MLSettings()
