"""Compatibility helpers for weights exported from the RASCAR notebook.

The notebook saved some objects from __main__ and some artefacts as separate
files. This module makes the production worker tolerant to both formats:
- full_pipeline.joblib with __main__.AttributeModels
- visual_reranker.joblib as either a dict or a raw CatBoostClassifier
- visual_pca.joblib / visual_reranker.cbm sidecars
- mlp_reranker.pt + mlp_scaler.joblib sidecar
- inference_config.json sidecar
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import ml_settings

DEFAULT_ATTRS = {
    "punch_type": "jab",
    "hand": "right",
    "target": "head",
    "effectiveness": "landed",
}


class AttributeModels:
    """Minimal runtime version of the notebook class used by joblib pickle.

    The object loaded from pickle carries `models`, `encoders` and `majority`.
    We only need predict() at inference time.
    """

    def predict(self, frame_feature_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        X = (
            frame_feature_df.reindex(columns=feature_cols, fill_value=0.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        pred = pd.DataFrame(index=frame_feature_df.index)
        models = getattr(self, "models", {}) or {}
        encoders = getattr(self, "encoders", {}) or {}
        majority = getattr(self, "majority", {}) or {}

        for attr in ["punch_type", "hand", "target", "effectiveness"]:
            if attr in models and attr in encoders:
                y = models[attr].predict(X)
                y = np.asarray(y).reshape(-1).astype(int)
                pred[attr] = encoders[attr].inverse_transform(y)
            else:
                pred[attr] = majority.get(attr, DEFAULT_ATTRS[attr])
        return pred


# Register the class where the notebook pickle expects it.
sys.modules["__main__"].AttributeModels = AttributeModels


def model_path(*names: str) -> Path | None:
    """Return first existing model file from models_dir by candidate names."""
    for name in names:
        p = ml_settings.models_path / name
        if p.exists():
            return p
    return None


def joblib_load_compat(path: Path) -> Any:
    from joblib import load

    sys.modules["__main__"].AttributeModels = AttributeModels
    return load(path)


@lru_cache(maxsize=1)
def load_inference_config() -> dict[str, Any]:
    path = model_path("inference_config.json")
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[compat] failed to read inference_config.json: {exc}")
        return {}
