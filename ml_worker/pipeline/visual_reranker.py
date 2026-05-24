"""
Visual reranker — ResNet18 + PCA + CatBoost, как в ноутбуке (cell 24).

Реализация делает upgrade event-вероятностей с использованием визуальных
эмбеддингов из 7 кадров вокруг каждого кандидата. В inference-pipeline
ожидаем что:

  models/visual_reranker.joblib хранит:
    - pca: sklearn PCA
    - clf: CatBoostClassifier
    - frame_offsets: список int
    - feature_cols: какие pose-фичи добавляются к visual

Если файла нет — visual reranker пропускается, используется только event_score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import ml_settings
from .compat import joblib_load_compat, model_path


FRAME_OFFSETS = [-12, -8, -4, 0, 4, 8, 12]
CROP_SIZE = 224


_MODEL: Optional[dict] = None


def has_visual_model() -> bool:
    return _load_visual() is not None


def _load_visual() -> Optional[dict]:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    bundle_path = model_path("visual_reranker.joblib", "visual_reranker(1).joblib")
    pca_path = model_path("visual_pca.joblib", "visual_pca(1).joblib")
    cbm_path = model_path("visual_reranker.cbm", "visual_reranker(1).cbm")

    if bundle_path is None and cbm_path is None:
        return None

    try:
        bundle = joblib_load_compat(bundle_path) if bundle_path is not None else None

        # Supported formats:
        # 1) {"pca": PCA, "clf": CatBoostClassifier}
        # 2) raw CatBoostClassifier + sidecar visual_pca.joblib
        if isinstance(bundle, dict):
            _MODEL = bundle
        else:
            clf = bundle
            if clf is None and cbm_path is not None:
                from catboost import CatBoostClassifier
                clf = CatBoostClassifier()
                clf.load_model(str(cbm_path))
            pca = joblib_load_compat(pca_path) if pca_path is not None else None
            _MODEL = {"pca": pca, "clf": clf}

        print(f"[visual_reranker] loaded model bundle from {bundle_path or cbm_path}")
        return _MODEL
    except Exception as exc:
        print(f"[visual_reranker] failed to load: {exc}")
        return None


def _extract_visual_embeddings(video_path: str, events: pd.DataFrame, pose_df: pd.DataFrame) -> Optional[np.ndarray]:
    """Считает ResNet18-эмбеддинги для 7 кадров вокруг каждого event'а.

    Возвращает массив (N_events, embedding_dim). Если CV или torch недоступны — None.
    """
    try:
        import cv2
        import torch
        from torchvision.models import resnet18, ResNet18_Weights
        from torchvision import transforms
    except ImportError:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device).eval()
    # снимем последний классификационный слой
    model.fc = torch.nn.Identity()  # type: ignore

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    embeddings: list = []

    pose_by_key = {(int(r["frame"]), r["fighter"]): r for _, r in pose_df.iterrows()}

    for _, ev in events.iterrows():
        clip_embs = []
        for off in FRAME_OFFSETS:
            target_frame = int(ev["frame"]) + off
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame))
            ret, frame = cap.read()
            if not ret:
                clip_embs.append(np.zeros(512, dtype=np.float32))
                continue

            # crop по bbox того же бойца (если есть)
            pose_row = pose_by_key.get((target_frame, ev["fighter"]))
            if pose_row is not None:
                x1, y1, x2, y2 = int(pose_row["bbox_x1"]), int(pose_row["bbox_y1"]), int(pose_row["bbox_x2"]), int(pose_row["bbox_y2"])
            else:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = w // 4, h // 4, 3 * w // 4, 3 * h // 4

            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                clip_embs.append(np.zeros(512, dtype=np.float32))
                continue

            crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            tensor = preprocess(crop_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model(tensor).cpu().numpy()[0]
            clip_embs.append(emb.astype(np.float32))

        embeddings.append(np.concatenate(clip_embs))

    cap.release()

    if not embeddings:
        return None
    return np.stack(embeddings)


def rerank_with_visual(
    events: pd.DataFrame,
    pose_df: pd.DataFrame,
    video_path: str,
) -> pd.DataFrame:
    """Добавляет колонку ``vis_prob`` к events. Если модели нет — копирует event_score."""
    out = events.copy()

    bundle = _load_visual()
    if bundle is None or events.empty:
        out["vis_prob"] = out.get("event_score", 0.0)
        return out

    embeddings = _extract_visual_embeddings(video_path, events, pose_df)
    if embeddings is None:
        out["vis_prob"] = out["event_score"]
        return out

    try:
        pca = bundle.get("pca")
        clf = bundle["clf"]

        # PCA sidecar is required for the current exported CatBoost model.
        if pca is None:
            raise RuntimeError("visual PCA is missing; put visual_pca.joblib into ml_worker/models")

        emb_pca = pca.transform(embeddings)
        vis_cols = [f"vis_pca_{i}" for i in range(emb_pca.shape[1])]
        for i, col in enumerate(vis_cols):
            out[col] = emb_pca[:, i]

        # The exported CatBoost was trained on pose features + vis_pca_* features.
        feature_names = list(getattr(clf, "feature_names_", None) or bundle.get("feature_cols", []))
        if feature_names:
            rows = []
            pose_by_fighter = {f: g.sort_values("frame") for f, g in pose_df.groupby("fighter")}
            for _, ev in out.iterrows():
                row = ev.to_dict()
                g = pose_by_fighter.get(ev["fighter"])
                if g is not None and not g.empty:
                    idx = (g["frame"] - ev["frame"]).abs().idxmin()
                    row.update(g.loc[idx].to_dict())
                rows.append(row)
            X = pd.DataFrame(rows).reindex(columns=feature_names, fill_value=0.0)
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            X = pd.DataFrame(emb_pca)

        vis_prob = clf.predict_proba(X)[:, 1]
        out["vis_prob"] = vis_prob
    except Exception as exc:
        print(f"[visual_reranker] inference failed: {exc}")
        out["vis_prob"] = out.get("event_score", 0.0)

    return out
