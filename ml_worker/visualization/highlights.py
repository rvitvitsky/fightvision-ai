"""
Нарезка хайлайтов: вокруг каждого удара берём окно ``[t - pad_before, t + pad_after]``,
склеиваем все окна в один итоговый файл.

Используем opencv для чтения/записи (proще ffmpeg-зависимости, всегда есть).
Если соседние окна пересекаются — мерджим в одно длинное окно.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional
import shutil
import subprocess

import cv2
import pandas as pd


def _merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not windows:
        return []
    windows.sort()
    merged = [windows[0]]
    for start, end in windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def build_highlights_reel(
    video_path: str,
    out_path: str,
    events: pd.DataFrame,
    *,
    pad_before: float = 0.6,
    pad_after: float = 0.6,
    progress_callback: Optional[Callable[[float], None]] = None,
    overlay_pose_df: Optional[pd.DataFrame] = None,
    draw_boxes: bool = False,
    draw_event_labels: bool = True,
) -> Optional[str]:
    """Создаёт mp4-нарезку вокруг ударов.

    Если в events нет ни одного удара — возвращает None и файл не создаётся.

    По умолчанию нарезка чистая: без bbox-прямоугольников.
    ``draw_event_labels=True`` оставляет аккуратную подпись удара снизу.
    ``draw_boxes=True`` можно включить для debug-режима, если нужен overlay bbox.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if events.empty:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pad_b_frames = max(1, int(round(pad_before * fps)))
    pad_a_frames = max(1, int(round(pad_after * fps)))

    raw_windows = [
        (max(0, int(ev["frame"]) - pad_b_frames), min(n_frames - 1, int(ev["frame"]) + pad_a_frames))
        for _, ev in events.iterrows()
    ]
    windows = _merge_windows(raw_windows)
    if not windows:
        cap.release()
        return None

    # OpenCV часто пишет mp4v, который нормально открывается в VLC,
    # но может не воспроизводиться в браузере. Поэтому сначала пишем
    # временный mp4, а в конце перекодируем в H.264 + yuv420p через ffmpeg.
    out_file = Path(out_path)
    tmp_out_path = str(out_file.with_name(out_file.stem + "_raw.mp4"))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter for {tmp_out_path}")

    # lookup events по кадру: подпись удара можно рисовать без bbox-разметки
    event_by_frame: dict[int, dict] = {}
    if draw_event_labels:
        for _, ev in events.iterrows():
            f = int(ev["frame"])
            for off in range(-3, pad_a_frames + 1):
                event_by_frame[f + off] = {
                    "text": " · ".join([
                        ev["fighter"].upper(),
                        str(ev.get("punch_type", "")).upper(),
                        str(ev.get("hand", "")).upper(),
                        str(ev.get("target", "")).upper(),
                    ]).strip(" ·"),
                    "fighter": ev["fighter"],
                }

    from .overlay import COLORS, _draw_label
    if draw_boxes:
        from .overlay import _bbox_at_frame

    total_window_frames = sum(end - start + 1 for start, end in windows)
    processed = 0
    last_reported = -1

    try:
        for start, end in windows:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for frame_idx in range(start, end + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                if draw_boxes and overlay_pose_df is not None:
                    for fighter in ("red", "blue"):
                        bbox = _bbox_at_frame(overlay_pose_df, frame_idx, fighter)
                        if bbox is not None:
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[fighter], 2)

                lbl = event_by_frame.get(frame_idx)
                if lbl is not None:
                    _draw_label(frame, lbl["text"], (30, height - 30), COLORS[lbl["fighter"]], scale=0.85)

                writer.write(frame)

                processed += 1
                if progress_callback is not None and processed - last_reported >= 30:
                    progress_callback(min(1.0, processed / max(1, total_window_frames)))
                    last_reported = processed
    finally:
        cap.release()
        writer.release()

    # Финальный web-friendly encode. Это чинит воспроизведение <video> в Chrome/Edge/Safari.
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            tmp_out_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_path,
        ]
        subprocess.run(cmd, check=True)
        Path(tmp_out_path).unlink(missing_ok=True)
    except Exception as exc:
        # Если ffmpeg по какой-то причине не сработал, не валим весь pipeline:
        # оставляем сырой mp4, который обычно можно открыть локально/VLC.
        print(f"[highlights] ffmpeg web transcode failed, using raw mp4: {exc}")
        shutil.move(tmp_out_path, out_path)

    if progress_callback is not None:
        progress_callback(1.0)

    return out_path
