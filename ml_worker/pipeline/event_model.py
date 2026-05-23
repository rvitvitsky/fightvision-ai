"""
Event model — предсказывает вероятность удара на каждом кадре для каждого бойца.

Два режима:
1. Если ``models/full_pipeline.joblib`` есть — грузим обученный HistGradientBoostingClassifier
   из ноутбука и его feature_cols.
2. Если файла нет — fallback на эвристику по скорости wrist'ов
   (находим пики ``wrist_speed`` через scipy.signal.find_peaks).

В обоих случаях возвращаем кандидатов в едином формате:
    [{frame, fighter, time, event_score, hand_hint, ...}, ...]
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from config import ml_settings


# ----------------------------------------------------------- weights loading


_BUNDLE: Optional[dict] = None


def _load_bundle() -> Optional[dict]:
    """Лениво грузит full_pipeline.joblib один раз на процесс."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    path = ml_settings.model_file("full_pipeline.joblib")
    if path is None:
        return None

    try:
        from joblib import load
        _BUNDLE = load(path)
        print(f"[event_model] loaded bundle from {path}")
        return _BUNDLE
    except Exception as exc:
        print(f"[event_model] failed to load bundle: {exc}")
        return None


def has_event_model() -> bool:
    return _load_bundle() is not None


# ----------------------------------------------------------- real predictor


def _predict_with_model(features_df: pd.DataFrame) -> pd.DataFrame:
    bundle = _load_bundle()
    if bundle is None:
        raise RuntimeError("Event model bundle is not loaded")

    model = bundle["event_model"]
    feature_cols = bundle["feature_cols"]

    X = features_df.reindex(columns=feature_cols, fill_value=0.0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    probs = model.predict_proba(X)[:, 1]

    df = features_df[["frame", "fighter"]].copy()
    df["event_prob"] = probs
    return df


# ------------------------------------------------------ heuristic fallback


def _heuristic_event_prob(features_df: pd.DataFrame) -> pd.DataFrame:
    """Простая эвристика: вероятность удара ~ speed wrist'а (нормированная по бойцу).

    Используется когда обученной модели нет — для smoke-теста всего pipeline.
    """
    df = features_df[["frame", "fighter"]].copy()

    # суммарная скорость wrist'ов как сырой сигнал
    s = (
        features_df.get("left_wrist_speed", pd.Series(0, index=features_df.index)).fillna(0)
        + features_df.get("right_wrist_speed", pd.Series(0, index=features_df.index)).fillna(0)
    )

    # нормировка по бойцу — у каждого свой baseline
    parts = []
    for fighter, idx in features_df.groupby("fighter").groups.items():
        vals = s.loc[idx]
        q95 = float(np.quantile(vals, 0.95)) if len(vals) > 0 else 1.0
        denom = max(q95, 1e-6)
        prob = np.clip(vals / denom, 0.0, 1.0)
        parts.append(prob)

    df["event_prob"] = pd.concat(parts).sort_index().values
    return df


# ----------------------------------------------------------- public API


def predict_event_probabilities(features_df: pd.DataFrame) -> pd.DataFrame:
    """Вернёт DataFrame [frame, fighter, event_prob] на каждый кадр и бойца."""
    if features_df.empty:
        return pd.DataFrame(columns=["frame", "fighter", "event_prob"])

    if has_event_model():
        return _predict_with_model(features_df)

    print("[event_model] no bundle — using heuristic fallback")
    return _heuristic_event_prob(features_df)


def probabilities_to_events(
    probs_df: pd.DataFrame,
    threshold: float = None,
    peak_distance_frames: int = None,
    smooth_window: int = None,
) -> pd.DataFrame:
    """Превращает per-frame вероятности в дискретные события (удары).

    Сглаживаем сигнал и берём пики через scipy.signal.find_peaks.
    """
    if probs_df.empty:
        return pd.DataFrame(columns=["frame", "fighter", "event_score"])

    threshold = threshold if threshold is not None else ml_settings.peak_threshold
    peak_distance_frames = peak_distance_frames if peak_distance_frames is not None else ml_settings.peak_distance_frames
    smooth_window = smooth_window if smooth_window is not None else ml_settings.smooth_window

    events: list = []
    for fighter, g in probs_df.groupby("fighter"):
        g = g.sort_values("frame").reset_index(drop=True)
        sig = g["event_prob"].rolling(window=smooth_window, min_periods=1, center=True).mean().values

        peaks, props = find_peaks(sig, height=threshold, distance=peak_distance_frames)
        for p in peaks:
            events.append({
                "frame": int(g.iloc[p]["frame"]),
                "fighter": fighter,
                "event_score": float(sig[p]),
            })

    return pd.DataFrame(events).sort_values("frame").reset_index(drop=True)
