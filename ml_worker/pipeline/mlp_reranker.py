"""
MLP reranker над pose + visual эмбеддингами (cell 26 ноутбука).

Ожидает models/mlp_reranker.pt — словарь с state_dict + список feature_cols + scaler.
Если файла нет — заглушка возвращает event_score как mlp_prob.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import ml_settings
from .compat import model_path, joblib_load_compat


_BUNDLE: Optional[dict] = None


def has_mlp_model() -> bool:
    return _load_mlp() is not None


def _load_mlp() -> Optional[dict]:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    path = model_path("mlp_reranker.pt", "mlp_reranker(1).pt")
    if path is None:
        return None

    try:
        import torch
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(bundle, dict) and bundle.get("scaler") is None:
            scaler_path = model_path("mlp_scaler.joblib", "mlp_scaler(1).joblib")
            if scaler_path is not None:
                bundle["scaler"] = joblib_load_compat(scaler_path)
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
        input_dim = int(bundle.get("input_dim", bundle.get("in_dim", len(feature_cols))))

        # Two supported architectures:
        # - notebook export: BatchNorm -> 512 -> 256 -> 128 -> 1, GELU
        # - older wrapper export: simple hidden_dim MLP
        class NotebookMLP(torch.nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(in_dim),
                    torch.nn.Linear(in_dim, 512),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.25),
                    torch.nn.Linear(512, 256),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.20),
                    torch.nn.Linear(256, 128),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.15),
                    torch.nn.Linear(128, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(1)

        class SimpleMLP(torch.nn.Module):
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
                return self.net(x).squeeze(-1)

        if "net.0.running_mean" in state_dict:
            model = NotebookMLP(input_dim)
        else:
            model = SimpleMLP(input_dim, int(bundle.get("hidden_dim", 128)))
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # подтянем фичи: для каждого event берём ближайший pose-frame и
        # объединяем его с колонками event'а (vis_pca_*, vis_prob, event_score).
        ev_feat = []
        feats_by_fighter = {f: g.sort_values("frame") for f, g in features_df.groupby("fighter")}
        for _, ev in events.iterrows():
            row = ev.to_dict()
            g = feats_by_fighter.get(ev["fighter"])
            if g is not None and not g.empty:
                idx = (g["frame"] - ev["frame"]).abs().idxmin()
                base = g.loc[idx].to_dict()
                base.update(row)
                row = base
            ev_feat.append(row)

        X = (
            pd.DataFrame(ev_feat)
            .reindex(columns=feature_cols, fill_value=0.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .values.astype(np.float32)
        )
        if scaler is not None:
            X = scaler.transform(X)

        with torch.no_grad():
            logits = model(torch.from_numpy(X.astype(np.float32))).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        out["mlp_prob"] = probs
    except Exception as exc:
        print(f"[mlp_reranker] inference failed: {exc}")
        out["mlp_prob"] = out.get("vis_prob", out.get("event_score", 0.0))

    return out
