"""
MLP reranker над pose + visual эмбеддингами (cell 26 ноутбука).

Ожидает models/mlp_reranker.pt — словарь с state_dict + список feature_cols + scaler.
Если файла нет — заглушка возвращает event_score как mlp_prob.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import ml_settings


_BUNDLE: Optional[dict] = None


def has_mlp_model() -> bool:
    return _load_mlp() is not None


def _load_mlp() -> Optional[dict]:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    path = ml_settings.model_file("mlp_reranker.pt")
    if path is None:
        return None

    try:
        import torch
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        _BUNDLE = bundle
        print(f"[mlp_reranker] loaded from {path}")
        return _BUNDLE
    except Exception as exc:
        print(f"[mlp_reranker] failed to load: {exc}")
        return None


def rerank_with_mlp(events: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет mlp_prob к events. Если модели нет — заглушка."""
    out = events.copy()

    bundle = _load_mlp()
    if bundle is None or events.empty:
        out["mlp_prob"] = out.get("vis_prob", out.get("event_score", 0.0))
        return out

    try:
        import torch
        feature_cols = bundle.get("feature_cols", [])
        scaler = bundle.get("scaler")
        state_dict = bundle["state_dict"]
        input_dim = bundle.get("input_dim", len(feature_cols))
        hidden_dim = bundle.get("hidden_dim", 128)

        # реконструируем сеть. Тонкости: если архитектура другая —
        # тренировка должна сохранить конфиг в bundle.
        class MLP(torch.nn.Module):
            def __init__(self, in_dim, hidden):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(hidden, hidden),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(hidden, 1),
                )

            def forward(self, x):
                return self.net(x)

        model = MLP(input_dim, hidden_dim)
        model.load_state_dict(state_dict)
        model.eval()

        # подтянем фичи: для каждого event берём ближайший кадр
        ev_feat = []
        feats_by_fighter = {f: g.sort_values("frame") for f, g in features_df.groupby("fighter")}
        for _, ev in events.iterrows():
            g = feats_by_fighter.get(ev["fighter"])
            if g is None or g.empty:
                ev_feat.append({c: 0.0 for c in feature_cols})
                continue
            idx = (g["frame"] - ev["frame"]).abs().idxmin()
            ev_feat.append(g.loc[idx, feature_cols].to_dict() if all(c in g.columns for c in feature_cols) else {c: 0.0 for c in feature_cols})

        X = pd.DataFrame(ev_feat).reindex(columns=feature_cols, fill_value=0.0).values.astype(np.float32)
        if scaler is not None:
            X = scaler.transform(X)

        with torch.no_grad():
            logits = model(torch.from_numpy(X.astype(np.float32))).squeeze(-1).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        out["mlp_prob"] = probs
    except Exception as exc:
        print(f"[mlp_reranker] inference failed: {exc}")
        out["mlp_prob"] = out.get("vis_prob", out.get("event_score", 0.0))

    return out
