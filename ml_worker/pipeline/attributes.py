"""
Attribute models — для каждого удара предсказываем punch_type / hand / target / effectiveness.

В ноутбуке это были отдельные classifier'ы внутри ``AttributeModels`` (см. cell 17).

Тут — два режима:
1. Если в бандле есть ``attr_models`` — используем его реальные модели.
2. Если нет — эвристика:
   - hand: какая рука быстрее в момент удара
   - punch_type: по соотношению vertical/horizontal velocity wrist'а
   - target: по y-координате wrist в момент удара относительно противника
   - effectiveness: дефолт ``landed`` (отличать landed/blocked/miss без модели не получится)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .event_model import _load_bundle, has_event_model


DEFAULT_ATTRS = {
    "punch_type": "jab",
    "hand": "right",
    "target": "head",
    "effectiveness": "landed",
}


def has_attribute_models() -> bool:
    bundle = _load_bundle()
    return bundle is not None and "attr_models" in bundle


# ---------------------------------------------------- helpers


def _nearest_frame_features(events: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Подтянуть к каждому event'у строку features ближайшего кадра по fighter+frame."""
    out_rows = []
    feats_by_fighter = {f: g.sort_values("frame") for f, g in features_df.groupby("fighter")}

    for _, ev in events.iterrows():
        g = feats_by_fighter.get(ev["fighter"])
        if g is None or g.empty:
            continue
        diffs = (g["frame"] - ev["frame"]).abs()
        idx = diffs.idxmin()
        row = g.loc[idx].to_dict()
        row["event_frame"] = int(ev["frame"])
        row["event_score"] = ev.get("event_score", 0.0)
        out_rows.append(row)

    return pd.DataFrame(out_rows)


# ---------------------------------------------------- real predictor


def _predict_with_models(events: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    bundle = _load_bundle()
    attr_models = bundle["attr_models"]
    feature_cols = bundle["feature_cols"]

    ev_feat = _nearest_frame_features(events, features_df)
    if ev_feat.empty:
        return events.copy()

    pred = attr_models.predict(ev_feat, feature_cols)
    out = events.merge(
        ev_feat[["event_frame", "fighter"]].rename(columns={"event_frame": "frame"}).assign(
            **{c: pred[c].values for c in pred.columns}
        ),
        on=["frame", "fighter"],
        how="left",
    )
    return out


# ---------------------------------------------------- heuristic predictor


def _heuristic_hand(row: dict) -> str:
    lws = float(row.get("left_wrist_speed", 0.0) or 0.0)
    rws = float(row.get("right_wrist_speed", 0.0) or 0.0)
    return "left" if lws > rws else "right"


def _heuristic_punch_type(row: dict, hand: str) -> str:
    vx = float(row.get(f"{hand}_wrist_vx", 0.0) or 0.0)
    vy = float(row.get(f"{hand}_wrist_vy", 0.0) or 0.0)
    elbow_ext = float(row.get(f"{hand}_arm_extension", 0.0) or 0.0)

    # Сильно вытянутая рука вперёд = jab/cross
    if elbow_ext > 0.55:
        # left hand → jab (передняя), right hand → cross (задняя) — упрощённо
        return "jab" if hand == "left" else "cross"

    # Большая вертикальная составляющая снизу-вверх → uppercut
    if abs(vy) > abs(vx) * 1.2 and vy < 0:
        return "uppercut"

    # Большая горизонтальная составляющая с боку → hook
    if abs(vx) > abs(vy):
        return "hook"

    return "jab"


def _heuristic_target(row: dict, hand: str) -> str:
    """Если wrist в момент удара выше плеч — head, иначе body."""
    wy = float(row.get(f"{hand}_wrist_y", 0.0) or 0.0)
    sy = float(row.get(f"{hand}_shoulder_y", 0.0) or 0.0)
    if wy == 0.0 or sy == 0.0:
        return "head"
    return "head" if wy <= sy + 20.0 else "body"


def _predict_heuristic(events: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    ev_feat = _nearest_frame_features(events, features_df)
    if ev_feat.empty:
        out = events.copy()
        for k, v in DEFAULT_ATTRS.items():
            out[k] = v
        return out

    rows = []
    for _, r in ev_feat.iterrows():
        row = r.to_dict()
        hand = _heuristic_hand(row)
        rows.append({
            "frame": int(row["event_frame"]),
            "fighter": row["fighter"],
            "hand": hand,
            "punch_type": _heuristic_punch_type(row, hand),
            "target": _heuristic_target(row, hand),
            "effectiveness": "landed",  # без модели — дефолт
        })

    attrs_df = pd.DataFrame(rows)
    return events.merge(attrs_df, on=["frame", "fighter"], how="left")


# ---------------------------------------------------- public API


def predict_attributes(events: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет колонки hand/punch_type/target/effectiveness для каждого events-ряда."""
    if events.empty:
        return events.copy()

    if has_attribute_models():
        result = _predict_with_models(events, features_df)
    else:
        result = _predict_heuristic(events, features_df)

    # гарантируем что все колонки есть и нет NaN
    for col, default in DEFAULT_ATTRS.items():
        if col not in result.columns:
            result[col] = default
        result[col] = result[col].fillna(default)

    return result
