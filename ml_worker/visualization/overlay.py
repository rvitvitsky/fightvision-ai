"""
Видео-overlay: на исходное видео рисуем bbox-ы бойцов в их цвете
и в моменты ударов — всплывающие лейблы (RED · JAB · LEFT · HEAD).

Сохраняет в out_path как .mp4 (mp4v fourcc).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd


COLORS = {
    "red": (0, 0, 255),
    "blue": (255, 80, 0),
}


def _bbox_at_frame(pose_df: pd.DataFrame, frame: int, fighter: str):
    """Ближайший bbox данного бойца не дальше чем на 3 кадра. Возвращает (x1,y1,x2,y2) или None."""
    g = pose_df[(pose_df["fighter"] == fighter)]
    if g.empty:
        return None
    diffs = (g["frame"] - frame).abs()
    idx = diffs.idxmin()
    if int(diffs.loc[idx]) > 3:
        return None
    row = g.loc[idx]
    return int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"])


def _draw_label(frame: np.ndarray, text: str, anchor: tuple, color: tuple, scale: float = 0.7):
    x, y = anchor
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    cv2.rectangle(
        frame,
        (x - pad, y - h - pad - baseline),
        (x + w + pad, y + pad),
        color,
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y - baseline // 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_overlay(
    video_path: str,
    out_path: str,
    pose_df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    label_persist_frames: int = 12,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> str:
    """Создаёт mp4 с отрисованными bbox'ами и лейблами ударов."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter for {out_path}")

    # быстрая lookup-структура: какие лейблы активны на каждом кадре
    label_by_frame: dict[int, list] = {}
    for _, ev in events.iterrows():
        start = int(ev["frame"])
        end = start + label_persist_frames
        text = " · ".join([
            ev["fighter"].upper(),
            str(ev.get("punch_type", "punch")).upper(),
            str(ev.get("hand", "")).upper(),
            str(ev.get("target", "")).upper(),
        ])
        for f in range(start, end + 1):
            label_by_frame.setdefault(f, []).append({
                "text": text,
                "fighter": ev["fighter"],
            })

    last_reported = -1
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # рисуем bbox-ы обоих бойцов (берём последний известный поз-ряд)
            for fighter in ("red", "blue"):
                bbox = _bbox_at_frame(pose_df, frame_idx, fighter)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[fighter], 2)
                    _draw_label(frame, fighter.upper(), (x1, max(25, y1 - 10)), COLORS[fighter], scale=0.6)

            # рисуем лейблы активных событий снизу по центру
            labels = label_by_frame.get(frame_idx, [])
            for i, lbl in enumerate(labels):
                anchor_x = 30
                anchor_y = height - 30 - i * 50
                _draw_label(frame, lbl["text"], (anchor_x, anchor_y), COLORS[lbl["fighter"]], scale=0.8)

            writer.write(frame)
            frame_idx += 1

            if progress_callback is not None and frame_idx - last_reported >= 30:
                progress_callback(min(1.0, frame_idx / max(1, n_frames)))
                last_reported = frame_idx
    finally:
        cap.release()
        writer.release()

    if progress_callback is not None:
        progress_callback(1.0)

    return out_path
