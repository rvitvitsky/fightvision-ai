"""
Pose extraction для одного видео.

Перенесено из ноутбука RASCAR (ячейки 6, 7, 8) с адаптациями для inference:
- работаем с одним видео, без parquet-кеша
- держим всё в памяти (DataFrame в конце)
- сообщаем прогресс по кадрам через callback

Главные сущности:
- COCO_KP_NAMES, KP — индексы keypoints YOLOv8-pose
- pose_color_scores() — red/blue score по зонам экипировки (перчатки, торс, шорты, голова)
- PairwiseRoleAssigner — назначает паре кандидатов роли red/blue с учётом геометрии
- extract_poses() — основной entry, возвращает DataFrame [frame, fighter, bbox, keypoints, ...]
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------- consts

COCO_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(COCO_KP_NAMES)}


# ----------------------------------------------------------------- low-level helpers


def _safe_crop(img: np.ndarray, x1: float, y1: float, x2: float, y2: float):
    h, w = img.shape[:2]
    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w, x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h, y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def _point_is_valid(kpts: np.ndarray, name: str, min_conf: float = 0.25) -> bool:
    idx = KP[name]
    return (
        idx < len(kpts)
        and np.isfinite(kpts[idx, 0])
        and np.isfinite(kpts[idx, 1])
        and kpts[idx, 2] >= min_conf
    )


def _get_point(kpts: np.ndarray, name: str) -> np.ndarray:
    return np.array([float(kpts[KP[name], 0]), float(kpts[KP[name], 1])], dtype=float)


def _bbox_diag(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(1.0, math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


# ---------------------------------------------------------- red/blue color scoring


def _color_fraction_in_crop(crop: Optional[np.ndarray]):
    """Возвращает (red_frac, blue_frac, valid_pixel_count) для HSV crop'а."""
    if crop is None or crop.size == 0:
        return 0.0, 0.0, 0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Красный в HSV разорван вокруг 0 — две полосы
    red1 = cv2.inRange(hsv, np.array([0, 45, 35]), np.array([13, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 45, 35]), np.array([180, 255, 255]))
    red_mask = (red1 > 0) | (red2 > 0)

    blue_mask = cv2.inRange(hsv, np.array([88, 40, 30]), np.array([138, 255, 255])) > 0

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # отсекаем серое/чёрное/белое (форма судьи, фон ринга)
    valid = (sat > 35) & (val > 35)

    denom = int(valid.sum())
    if denom <= 0:
        return 0.0, 0.0, 0

    red_score = float((red_mask & valid).sum() / denom)
    blue_score = float((blue_mask & valid).sum() / denom)
    return red_score, blue_score, denom


def _add_region(regions: list, frame: np.ndarray, x1, y1, x2, y2, weight: float):
    crop = _safe_crop(frame, x1, y1, x2, y2)
    red, blue, denom = _color_fraction_in_crop(crop)
    if denom > 15:
        regions.append({"red": red, "blue": blue, "weight": float(weight), "pixels": int(denom)})


def pose_color_scores(frame: np.ndarray, kpts: np.ndarray, bbox) -> tuple[float, float, float]:
    """Red/blue по зонам экипировки (перчатки, торс, шорты, голова), а не по всему bbox.

    Возвращает (red_score, blue_score, coverage).
    """
    x1, y1, x2, y2 = map(float, bbox)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    diag = _bbox_diag(bbox)

    regions: list = []
    glove_r = max(10, min(32, int(0.045 * diag)))
    head_r = max(12, min(38, int(0.055 * diag)))

    # 1) Перчатки — самый важный сигнал
    for wrist_name in ("left_wrist", "right_wrist"):
        if _point_is_valid(kpts, wrist_name, min_conf=0.20):
            px, py = _get_point(kpts, wrist_name)
            _add_region(regions, frame, px - glove_r, py - glove_r, px + glove_r, py + glove_r, 2.0)

    # 2) Торс
    torso_pts = [
        _get_point(kpts, n)
        for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if _point_is_valid(kpts, n, min_conf=0.20)
    ]
    if len(torso_pts) >= 2:
        pts = np.vstack(torso_pts)
        tx1, ty1 = pts.min(axis=0)
        tx2, ty2 = pts.max(axis=0)
        _add_region(regions, frame, tx1 - 0.08 * bw, ty1 - 0.08 * bh, tx2 + 0.08 * bw, ty2 + 0.08 * bh, 1.4)

    # 3) Шорты / пояс
    hip_pts = [_get_point(kpts, n) for n in ("left_hip", "right_hip") if _point_is_valid(kpts, n, 0.20)]
    knee_pts = [_get_point(kpts, n) for n in ("left_knee", "right_knee") if _point_is_valid(kpts, n, 0.20)]
    if hip_pts:
        hips = np.vstack(hip_pts)
        hx1, hy1 = hips.min(axis=0)
        hx2, hy2 = hips.max(axis=0)
        if knee_pts:
            bottom = float(np.nanmean(np.vstack(knee_pts)[:, 1]))
        else:
            bottom = hy2 + 0.22 * bh
        _add_region(regions, frame, hx1 - 0.18 * bw, hy1 - 0.04 * bh, hx2 + 0.18 * bw, bottom, 1.2)

    # 4) Голова / шлем
    head_pts = [
        _get_point(kpts, n)
        for n in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
        if _point_is_valid(kpts, n, 0.20)
    ]
    if head_pts:
        cx, cy = np.vstack(head_pts).mean(axis=0)
        _add_region(regions, frame, cx - head_r, cy - head_r, cx + head_r, cy + head_r, 0.8)

    # Fallback: центральная зона bbox (не весь bbox — фон не нужен)
    if not regions:
        _add_region(regions, frame, x1 + 0.25 * bw, y1 + 0.20 * bh, x2 - 0.25 * bw, y2 - 0.15 * bh, 0.6)

    total_w = sum(r["weight"] for r in regions)
    if total_w <= 0:
        return 0.0, 0.0, 0.0

    red_score = sum(r["red"] * r["weight"] for r in regions) / total_w
    blue_score = sum(r["blue"] * r["weight"] for r in regions) / total_w
    return float(red_score), float(blue_score), float(sum(r["pixels"] for r in regions))


def pose_quality(kpts: np.ndarray) -> float:
    """Средняя уверенность важных keypoints (нос, плечи, локти, кисти, бёдра, колени)."""
    useful = (
        "nose",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
    )
    confs = []
    for name in useful:
        idx = KP[name]
        if idx < len(kpts) and np.isfinite(kpts[idx, 2]):
            confs.append(float(kpts[idx, 2]))
    return float(np.mean(confs)) if confs else 0.0


# ----------------------------------------------------------- pair role assigner


class PairwiseRoleAssigner:
    """Назначает паре кандидатов роли red/blue.

    Главная фишка vs независимого выбора red и blue:
    мы оптимизируем ПАРУ (одинаковая высота bbox, похожий bottom_y,
    разумная горизонтальная дистанция), что отсекает фоновых людей.
    """

    def __init__(
        self,
        max_lost: int = 35,
        max_candidates: int = 8,
        min_area_frac: float = 0.004,
        min_height_frac: float = 0.18,
        min_bottom_frac: float = 0.50,
        min_pose_quality: float = 0.10,
        min_new_color: float = 0.06,
        min_new_margin: float = 0.020,
        min_pair_height_ratio: float = 0.45,
        max_pair_bottom_diff_frac: float = 0.22,
        max_pair_center_dx_frac: float = 0.75,
        color_weight: float = 5.0,
        continuity_weight: float = 1.8,
        pair_geometry_weight: float = 3.0,
        area_weight: float = 0.25,
        ema_alpha: float = 0.10,
    ):
        self.prev_centers = {"red": None, "blue": None}
        self.last_seen = {"red": -10**9, "blue": -10**9}
        self.role_track_id = {"red": None, "blue": None}
        self.track_stats: dict = {}

        self.max_lost = max_lost
        self.max_candidates = max_candidates
        self.min_area_frac = min_area_frac
        self.min_height_frac = min_height_frac
        self.min_bottom_frac = min_bottom_frac
        self.min_pose_quality = min_pose_quality
        self.min_new_color = min_new_color
        self.min_new_margin = min_new_margin
        self.min_pair_height_ratio = min_pair_height_ratio
        self.max_pair_bottom_diff_frac = max_pair_bottom_diff_frac
        self.max_pair_center_dx_frac = max_pair_center_dx_frac
        self.color_weight = color_weight
        self.continuity_weight = continuity_weight
        self.pair_geometry_weight = pair_geometry_weight
        self.area_weight = area_weight
        self.ema_alpha = ema_alpha

    @staticmethod
    def _center(c):
        x1, y1, x2, y2 = c["bbox"]
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)

    @staticmethod
    def _height(c):
        return max(1.0, float(c["bbox"][3] - c["bbox"][1]))

    @staticmethod
    def _bottom(c):
        return float(c["bbox"][3])

    def _stats(self, c):
        h = max(1.0, float(c["frame_h"]))
        w = max(1.0, float(c["frame_w"]))
        area = max(1.0, float(c["area"]))
        bbox_h = self._height(c)
        return {
            "area_frac": area / (w * h),
            "height_frac": bbox_h / h,
            "bottom_frac": self._bottom(c) / h,
        }

    def _update_track_ema(self, candidates):
        a = self.ema_alpha
        for c in candidates:
            tid = int(c.get("track_id", -1))
            if tid < 0:
                c["red_score_smooth"] = c.get("red_score", 0.0)
                c["blue_score_smooth"] = c.get("blue_score", 0.0)
                c["track_count"] = 1
                continue

            st = self.track_stats.get(tid)
            if st is None:
                st = {
                    "red_ema": float(c.get("red_score", 0.0)),
                    "blue_ema": float(c.get("blue_score", 0.0)),
                    "count": 0,
                }
            else:
                st["red_ema"] = (1 - a) * st["red_ema"] + a * float(c.get("red_score", 0.0))
                st["blue_ema"] = (1 - a) * st["blue_ema"] + a * float(c.get("blue_score", 0.0))
            st["count"] += 1
            self.track_stats[tid] = st

            c["red_score_smooth"] = st["red_ema"]
            c["blue_score_smooth"] = st["blue_ema"]
            c["track_count"] = st["count"]

    def _is_candidate_reasonable(self, c) -> bool:
        s = self._stats(c)
        if s["area_frac"] < self.min_area_frac:
            return False
        if s["height_frac"] < self.min_height_frac:
            return False
        if s["bottom_frac"] < self.min_bottom_frac:
            return False
        if float(c.get("pose_quality", 0.0)) < self.min_pose_quality:
            return False
        return True

    def _color_score_for_role(self, c, role):
        red = float(c.get("red_score_smooth", c.get("red_score", 0.0)))
        blue = float(c.get("blue_score_smooth", c.get("blue_score", 0.0)))
        main, other = (red, blue) if role == "red" else (blue, red)
        return main, main - other, abs(red - blue)

    def _can_be_new_role(self, c, role) -> bool:
        main, margin, abs_margin = self._color_score_for_role(c, role)
        if main < self.min_new_color:
            return False
        if abs_margin < self.min_new_margin:
            return False
        return margin > 0

    def _continuity_score(self, c, role, frame_idx) -> float:
        center = self._center(c)
        frame_diag = max(1.0, math.sqrt(c["frame_w"] ** 2 + c["frame_h"] ** 2))
        score = 0.0
        prev = self.prev_centers[role]
        if prev is not None and frame_idx - self.last_seen[role] <= self.max_lost:
            dist = np.linalg.norm(center - prev) / frame_diag
            score += max(0.0, 1.0 - 10.0 * dist)
        tid = int(c.get("track_id", -1))
        if self.role_track_id[role] is not None and tid == self.role_track_id[role]:
            score += 0.8
        return score

    def _pair_is_reasonable(self, red_c, blue_c) -> bool:
        frame_h = max(1.0, float(red_c["frame_h"]))
        frame_w = max(1.0, float(red_c["frame_w"]))
        red_h = self._height(red_c)
        blue_h = self._height(blue_c)
        if min(red_h, blue_h) / max(red_h, blue_h) < self.min_pair_height_ratio:
            return False
        if abs(self._bottom(red_c) - self._bottom(blue_c)) / frame_h > self.max_pair_bottom_diff_frac:
            return False
        red_center = self._center(red_c)
        blue_center = self._center(blue_c)
        if abs(red_center[0] - blue_center[0]) / frame_w > self.max_pair_center_dx_frac:
            return False
        return True

    def _pair_geometry_score(self, red_c, blue_c) -> float:
        frame_h = max(1.0, float(red_c["frame_h"]))
        frame_w = max(1.0, float(red_c["frame_w"]))
        red_h = self._height(red_c)
        blue_h = self._height(blue_c)
        height_ratio = min(red_h, blue_h) / max(red_h, blue_h)
        bottom_diff = abs(self._bottom(red_c) - self._bottom(blue_c)) / frame_h
        dx = abs(self._center(red_c)[0] - self._center(blue_c)[0]) / frame_w
        score = height_ratio
        score += max(0.0, 1.0 - bottom_diff / self.max_pair_bottom_diff_frac)
        score += max(0.0, 1.0 - dx / self.max_pair_center_dx_frac)
        return score

    def _role_candidate_score(self, c, role, frame_idx) -> float:
        _, margin, abs_margin = self._color_score_for_role(c, role)
        s = self._stats(c)
        score = self.color_weight * margin
        score += self.area_weight * math.log1p(100.0 * s["area_frac"])
        score += self.continuity_weight * self._continuity_score(c, role, frame_idx)
        if abs_margin < self.min_new_margin:
            score -= 1.0
        return score

    def assign(self, candidates: list, frame_idx: int) -> dict:
        if not candidates:
            return {}

        candidates = [c for c in candidates if self._is_candidate_reasonable(c)]
        if not candidates:
            return {}

        candidates = sorted(candidates, key=lambda c: c["area"], reverse=True)[: self.max_candidates]
        self._update_track_ema(candidates)

        best_pair, best_score = None, -1e18

        for i, red_c in enumerate(candidates):
            if not self._can_be_new_role(red_c, "red"):
                red_tid = int(red_c.get("track_id", -1))
                is_old_red = (
                    self.role_track_id["red"] is not None
                    and red_tid == self.role_track_id["red"]
                    and frame_idx - self.last_seen["red"] <= self.max_lost
                )
                if not is_old_red:
                    continue

            for j, blue_c in enumerate(candidates):
                if i == j:
                    continue

                if not self._can_be_new_role(blue_c, "blue"):
                    blue_tid = int(blue_c.get("track_id", -1))
                    is_old_blue = (
                        self.role_track_id["blue"] is not None
                        and blue_tid == self.role_track_id["blue"]
                        and frame_idx - self.last_seen["blue"] <= self.max_lost
                    )
                    if not is_old_blue:
                        continue

                if not self._pair_is_reasonable(red_c, blue_c):
                    continue

                score = (
                    self._role_candidate_score(red_c, "red", frame_idx)
                    + self._role_candidate_score(blue_c, "blue", frame_idx)
                    + self.pair_geometry_weight * self._pair_geometry_score(red_c, blue_c)
                )
                if score > best_score:
                    best_score = score
                    best_pair = (red_c, blue_c)

        assigned: dict = {}
        if best_pair is not None:
            red_c, blue_c = best_pair
            assigned["red"] = red_c
            assigned["blue"] = blue_c
            for role, cand in assigned.items():
                self.prev_centers[role] = self._center(cand)
                self.last_seen[role] = frame_idx
                self.role_track_id[role] = int(cand.get("track_id", -1))
            return assigned

        # fallback: продолжаем только уже известные треки
        for role in ("red", "blue"):
            role_tid = self.role_track_id[role]
            if role_tid is None or frame_idx - self.last_seen[role] > self.max_lost:
                continue
            same_track = [c for c in candidates if int(c.get("track_id", -1)) == role_tid]
            if not same_track:
                continue
            cand = max(same_track, key=lambda c: self._role_candidate_score(c, role, frame_idx))
            assigned[role] = cand
            self.prev_centers[role] = self._center(cand)
            self.last_seen[role] = frame_idx

        return assigned


# ---------------------------------------------------------------- main entry


def _flatten_pose_row(frame_idx: int, fighter: str, cand: dict) -> dict:
    row = {
        "frame": int(frame_idx),
        "fighter": fighter,
        "track_id": cand.get("track_id", -1),
        "bbox_x1": cand["bbox"][0],
        "bbox_y1": cand["bbox"][1],
        "bbox_x2": cand["bbox"][2],
        "bbox_y2": cand["bbox"][3],
        "red_score": cand.get("red_score", 0.0),
        "blue_score": cand.get("blue_score", 0.0),
        "red_score_smooth": cand.get("red_score_smooth", cand.get("red_score", 0.0)),
        "blue_score_smooth": cand.get("blue_score_smooth", cand.get("blue_score", 0.0)),
        "color_coverage": cand.get("color_coverage", 0.0),
        "pose_quality": cand.get("pose_quality", np.nan),
        "det_conf": cand.get("det_conf", np.nan),
    }
    kpts = cand["keypoints"]
    for idx, name in enumerate(COCO_KP_NAMES):
        row[f"{name}_x"] = float(kpts[idx, 0])
        row[f"{name}_y"] = float(kpts[idx, 1])
        row[f"{name}_conf"] = float(kpts[idx, 2])
    return row


def extract_poses(
    video_path: str,
    yolo_model_name: str = "yolov8n-pose.pt",
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.50,
    frame_stride: int = 1,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> pd.DataFrame:
    """Прогоняет YOLO-pose по всему видео и возвращает DataFrame с pose-фичами.

    Колонки:
        frame, fighter, track_id, bbox_*, red_score*, blue_score*,
        pose_quality, det_conf, и для каждого keypoint — *_x, *_y, *_conf.
    """
    from ultralytics import YOLO  # импорт здесь — torch стартует только когда нужно

    video_path = str(Path(video_path).resolve())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    model = YOLO(yolo_model_name)
    assigner = PairwiseRoleAssigner()
    rows: list = []

    frame_idx = 0
    last_reported = -1
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                try:
                    result = model.track(
                        frame, persist=True, verbose=False,
                        conf=yolo_conf, iou=yolo_iou, classes=[0],
                    )[0]
                except Exception:
                    result = model.predict(
                        frame, verbose=False, conf=yolo_conf, iou=yolo_iou, classes=[0],
                    )[0]

                candidates: list = []
                if result.boxes is not None and result.keypoints is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = (
                        result.boxes.conf.cpu().numpy()
                        if result.boxes.conf is not None
                        else np.ones(len(boxes))
                    )
                    ids = (
                        result.boxes.id.cpu().numpy().astype(int)
                        if getattr(result.boxes, "id", None) is not None
                        else None
                    )
                    kpts_xy = result.keypoints.xy.cpu().numpy()
                    kpts_conf = (
                        result.keypoints.conf.cpu().numpy()
                        if getattr(result.keypoints, "conf", None) is not None
                        else np.ones(kpts_xy.shape[:2], dtype=float)
                    )

                    for i, bbox in enumerate(boxes):
                        x1, y1, x2, y2 = bbox.astype(float)
                        area = max(0.0, (x2 - x1) * (y2 - y1))
                        if area < 500:
                            continue
                        kpts = np.concatenate([kpts_xy[i], kpts_conf[i, :, None]], axis=1)
                        red_s, blue_s, cov = pose_color_scores(frame, kpts, (x1, y1, x2, y2))
                        candidates.append({
                            "bbox": [x1, y1, x2, y2],
                            "area": area,
                            "red_score": red_s,
                            "blue_score": blue_s,
                            "color_coverage": cov,
                            "pose_quality": pose_quality(kpts),
                            "det_conf": float(confs[i]),
                            "track_id": int(ids[i]) if ids is not None else i,
                            "keypoints": kpts,
                            "frame_w": width,
                            "frame_h": height,
                        })

                assigned = assigner.assign(candidates, frame_idx)
                for fighter, cand in assigned.items():
                    rows.append(_flatten_pose_row(frame_idx, fighter, cand))

            frame_idx += 1

            # прогресс раз в ~30 кадров (1 секунду при 30fps)
            if progress_callback is not None and frame_idx - last_reported >= 30:
                progress_callback(min(1.0, frame_idx / max(1, n_frames)))
                last_reported = frame_idx
    finally:
        cap.release()

    if progress_callback is not None:
        progress_callback(1.0)

    df = pd.DataFrame(rows)
    df.attrs["fps"] = float(fps)
    df.attrs["width"] = int(width)
    df.attrs["height"] = int(height)
    df.attrs["n_frames"] = int(n_frames)
    df.attrs["video_path"] = video_path
    return df
