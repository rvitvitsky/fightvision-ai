"""
Postprocessing — rank blend, NMS, финальный отбор top-K.

Финальное скоринг из ноутбука (cell 29):
    rank_475mlp_525vis = 0.475 * mlp_prob_rank + 0.525 * vis_prob_rank

NMS: внутри каждого видео не должно быть двух событий одного и того же бойца
ближе чем на ``nms_distance_frames`` друг к другу.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ml_settings


def add_rank_blend(events: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку ``final_score`` = 0.475 * mlp_rank + 0.525 * vis_rank."""
    out = events.copy()

    for c in ("event_score", "vis_prob", "mlp_prob"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
            out[f"{c}_rank"] = out[c].rank(method="average", pct=True)

    has_mlp = "mlp_prob_rank" in out.columns
    has_vis = "vis_prob_rank" in out.columns

    if has_mlp and has_vis:
        out["final_score"] = 0.475 * out["mlp_prob_rank"] + 0.525 * out["vis_prob_rank"]
    elif has_vis:
        out["final_score"] = out["vis_prob_rank"]
    elif has_mlp:
        out["final_score"] = out["mlp_prob_rank"]
    else:
        out["final_score"] = out.get("event_score", 0.0)

    return out


def nms_within_fighter(events: pd.DataFrame, distance_frames: int) -> pd.DataFrame:
    """Внутри каждого бойца: если два события ближе чем distance_frames — оставляем
    то у которого final_score выше."""
    if events.empty:
        return events.copy()

    keep_idx = []
    for fighter, g in events.groupby("fighter"):
        g = g.sort_values("final_score", ascending=False)
        chosen_frames: list = []
        for idx, row in g.iterrows():
            frame = int(row["frame"])
            if all(abs(frame - cf) >= distance_frames for cf in chosen_frames):
                chosen_frames.append(frame)
                keep_idx.append(idx)

    return events.loc[keep_idx].sort_values("frame").reset_index(drop=True)


def postprocess_events(
    events: pd.DataFrame,
    distance_frames: int | None = None,
    keep_top_quantile: float = 1.0,
) -> pd.DataFrame:
    """Полный постпроцессинг: rank blend → NMS → опционально топ-q квантиль."""
    if events.empty:
        return events.copy()

    out = add_rank_blend(events)
    dist = distance_frames if distance_frames is not None else ml_settings.nms_distance_frames
    out = nms_within_fighter(out, dist)

    if 0.0 < keep_top_quantile < 1.0 and len(out) > 0:
        thr = float(out["final_score"].quantile(1.0 - keep_top_quantile))
        out = out[out["final_score"] >= thr].reset_index(drop=True)

    return out
