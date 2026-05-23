"""
Расчёт summary (количество, точность) и Fight DNA (4 метрики на радар) для фронта.

DNA — это эвристические агрегаты от 0 до 100:
  - aggression  — количество ударов / минуту
  - defense     — доля заблокированных + пропущенных ударов противника
  - mobility    — средняя скорость движения корпуса (body_speed)
  - pressure    — доля времени когда боец ближе к противнику чем средняя дистанция
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _safe_round(x: float, lo: int = 0, hi: int = 100) -> int:
    if np.isnan(x) or np.isinf(x):
        return lo
    return int(max(lo, min(hi, round(float(x)))))


def compute_summary(events: pd.DataFrame) -> Dict[str, dict]:
    """Возвращает {"red": {...}, "blue": {...}} со статами по ударам."""
    out: Dict[str, dict] = {}
    for fighter in ("red", "blue"):
        g = events[events["fighter"] == fighter] if not events.empty else events.iloc[0:0]

        n = len(g)
        landed = int((g["effectiveness"] == "landed").sum()) if "effectiveness" in g.columns else 0
        blocked = int((g["effectiveness"] == "blocked").sum()) if "effectiveness" in g.columns else 0
        missed = int((g["effectiveness"] == "miss").sum()) if "effectiveness" in g.columns else 0
        accuracy = round(landed / n, 3) if n > 0 else 0.0

        out[fighter] = {
            "punches": n,
            "accuracy": accuracy,
            "landed": landed,
            "blocked": blocked,
            "missed": missed,
        }
    return out


def compute_dna(events: pd.DataFrame, features_df: pd.DataFrame) -> Dict[str, dict]:
    """4 метрики для радар-чарта на фронте. Чисто эвристически из событий и движения."""
    fps = float(features_df.attrs.get("fps", 30.0)) if not features_df.empty else 30.0
    n_frames = int(features_df.attrs.get("n_frames", len(features_df))) if not features_df.empty else 1
    duration_sec = n_frames / max(fps, 1.0)
    duration_min = max(duration_sec / 60.0, 1e-6)

    out: Dict[str, dict] = {}

    for fighter in ("red", "blue"):
        ev = events[events["fighter"] == fighter] if not events.empty else events.iloc[0:0]
        feats = features_df[features_df["fighter"] == fighter]

        # aggression: 4-5 ударов в минуту = ~70 баллов; калибровка через clip
        punches_per_min = len(ev) / duration_min if duration_min > 0 else 0.0
        aggression = _safe_round(punches_per_min * 15.0)

        # defense: доля заблокированных в ударах противника + 1 - доля landed против себя
        opponent = "blue" if fighter == "red" else "red"
        opp_ev = events[events["fighter"] == opponent] if not events.empty else events.iloc[0:0]
        if len(opp_ev) > 0:
            opp_landed_share = float((opp_ev["effectiveness"] == "landed").sum()) / len(opp_ev)
            defense = _safe_round((1.0 - opp_landed_share) * 100.0)
        else:
            defense = 50

        # mobility: средняя скорость body_speed (px/frame), нормирована по 95-му перцентилю
        if not feats.empty and "body_speed" in feats.columns:
            speed = feats["body_speed"].fillna(0.0)
            if len(speed) > 0 and float(speed.quantile(0.95)) > 0:
                mobility = _safe_round(float(speed.mean()) / float(speed.quantile(0.95)) * 100.0)
            else:
                mobility = 50
        else:
            mobility = 50

        # pressure: доля кадров когда opponent_distance меньше своей медианы
        if not feats.empty and "opponent_distance" in feats.columns:
            d = feats["opponent_distance"].dropna()
            if len(d) > 0:
                median_d = float(d.median())
                pressure_share = float((d < median_d).sum()) / len(d)
                pressure = _safe_round(pressure_share * 100.0)
            else:
                pressure = 50
        else:
            pressure = 50

        out[fighter] = {
            "aggression": aggression,
            "defense": defense,
            "mobility": mobility,
            "pressure": pressure,
        }

    return out
