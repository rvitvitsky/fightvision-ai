"""
Frame-level features из pose DataFrame.

Берёт сырые keypoints (которые лежат в pose_df) и считает derived-фичи:
- скорость и ускорение wrist'ов и elbow'ов
- расстояние между бойцами
- углы локтя (вытянутая рука = большой угол)
- движение корпуса
- симметричные фичи vs противник

Эти фичи идут на вход event_model. Точная схема фич должна совпадать
с той что в обученном бандле (feature_cols).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


KP_NAMES_FOR_FEATURES = (
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
)


def _smooth(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def _velocity(series: pd.Series) -> pd.Series:
    return series.diff().fillna(0.0)


def _angle_3pt(ax, ay, bx, by, cx, cy) -> pd.Series:
    """Угол в точке B, образованный векторами BA и BC. В радианах."""
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    dot = v1x * v2x + v1y * v2y
    n1 = np.sqrt(v1x ** 2 + v1y ** 2)
    n2 = np.sqrt(v2x ** 2 + v2y ** 2)
    denom = (n1 * n2).replace(0.0, np.nan)
    cos_a = (dot / denom).clip(-1.0, 1.0)
    return np.arccos(cos_a).fillna(0.0)


def _bbox_center(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    cx = (df["bbox_x1"] + df["bbox_x2"]) / 2.0
    cy = (df["bbox_y1"] + df["bbox_y2"]) / 2.0
    return cx, cy


def build_frame_features(pose_df: pd.DataFrame) -> pd.DataFrame:
    """Из pose_df строит frame-level фичи на каждую (frame, fighter) запись.

    Возвращает DataFrame с теми же ключевыми колонками + derived-фичи.
    Сохраняет ``.attrs`` от исходного pose_df (fps, dims, etc).
    """
    if pose_df.empty:
        return pose_df.copy()

    df = pose_df.sort_values(["fighter", "frame"]).reset_index(drop=True).copy()

    # bbox features
    cx, cy = _bbox_center(df)
    df["bbox_cx"] = cx
    df["bbox_cy"] = cy
    df["bbox_w"] = df["bbox_x2"] - df["bbox_x1"]
    df["bbox_h"] = df["bbox_y2"] - df["bbox_y1"]
    df["bbox_area"] = df["bbox_w"] * df["bbox_h"]

    # per-fighter derived features
    feature_blocks = []
    for fighter, group in df.groupby("fighter", sort=False):
        g = group.sort_values("frame").copy()

        # smooth keypoint coordinates by fighter timeline
        for kp in KP_NAMES_FOR_FEATURES:
            g[f"{kp}_x_smooth"] = _smooth(g[f"{kp}_x"])
            g[f"{kp}_y_smooth"] = _smooth(g[f"{kp}_y"])

        # wrist velocity / acceleration (the most useful signal for punches)
        for side in ("left", "right"):
            wx = f"{side}_wrist_x_smooth"
            wy = f"{side}_wrist_y_smooth"
            g[f"{side}_wrist_vx"] = _velocity(g[wx])
            g[f"{side}_wrist_vy"] = _velocity(g[wy])
            g[f"{side}_wrist_speed"] = np.sqrt(g[f"{side}_wrist_vx"] ** 2 + g[f"{side}_wrist_vy"] ** 2)
            g[f"{side}_wrist_acc"] = _velocity(g[f"{side}_wrist_speed"])

            # elbow angle: shoulder–elbow–wrist
            g[f"{side}_elbow_angle"] = _angle_3pt(
                g[f"{side}_shoulder_x"], g[f"{side}_shoulder_y"],
                g[f"{side}_elbow_x"], g[f"{side}_elbow_y"],
                g[f"{side}_wrist_x"], g[f"{side}_wrist_y"],
            )

            # extension: distance wrist–shoulder (нормировано на bbox высоту)
            dx = g[f"{side}_wrist_x"] - g[f"{side}_shoulder_x"]
            dy = g[f"{side}_wrist_y"] - g[f"{side}_shoulder_y"]
            g[f"{side}_arm_extension"] = np.sqrt(dx ** 2 + dy ** 2) / g["bbox_h"].clip(lower=1.0)

        # body center movement
        g["body_vx"] = _velocity(g["bbox_cx"])
        g["body_vy"] = _velocity(g["bbox_cy"])
        g["body_speed"] = np.sqrt(g["body_vx"] ** 2 + g["body_vy"] ** 2)

        feature_blocks.append(g)

    df = pd.concat(feature_blocks, axis=0, ignore_index=True).sort_values(["frame", "fighter"])

    # opponent distance — между bbox центрами red и blue в одном кадре
    pivot_x = df.pivot_table(index="frame", columns="fighter", values="bbox_cx").rename(
        columns={"red": "red_cx", "blue": "blue_cx"}
    )
    pivot_y = df.pivot_table(index="frame", columns="fighter", values="bbox_cy").rename(
        columns={"red": "red_cy", "blue": "blue_cy"}
    )
    dist = np.sqrt(
        (pivot_x.get("red_cx", 0) - pivot_x.get("blue_cx", 0)) ** 2
        + (pivot_y.get("red_cy", 0) - pivot_y.get("blue_cy", 0)) ** 2
    )
    df = df.merge(dist.rename("opponent_distance").reset_index(), on="frame", how="left")
    df["opponent_distance"] = df["opponent_distance"].fillna(df["opponent_distance"].median())

    df = df.fillna(0.0)
    df.attrs.update(pose_df.attrs)
    return df


def get_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Список колонок, пригодных для модели (числовые, без id-полей)."""
    exclude = {
        "frame", "fighter", "track_id",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    }
    cols = []
    for c in features_df.columns:
        if c in exclude:
            continue
        if features_df[c].dtype.kind in "fiub":
            cols.append(c)
    return cols
