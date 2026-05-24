"""
Inference entry: одна функция ``run_inference(video_path)``, которая
дёргается из RQ-таски и возвращает финальный dict для frontend'а.

Каждый этап pipeline получает прогресс-callback. Worker записывает
этот прогресс в Redis, gateway отдаёт его клиенту через GET /api/jobs/{id}.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from config import ml_settings

from pipeline.pose import extract_poses
from pipeline.features import build_frame_features, get_feature_columns
from pipeline.event_model import predict_event_probabilities, probabilities_to_events, has_event_model
from pipeline.attributes import predict_attributes, has_attribute_models
from pipeline.visual_reranker import rerank_with_visual, has_visual_model
from pipeline.mlp_reranker import rerank_with_mlp, has_mlp_model
from pipeline.postprocess import postprocess_events
from pipeline.stats import compute_summary, compute_dna

from visualization.highlights import build_highlights_reel


# Все этапы и их вес в общем прогрессе. Сумма должна быть = 1.0.
STAGES = [
    ("pose_extraction",   0.55),
    ("features",          0.05),
    ("event_detection",   0.10),
    ("visual_rerank",     0.10),
    ("mlp_rerank",        0.05),
    ("attributes",        0.02),
    ("postprocess",       0.01),
    ("highlights_render", 0.12),
]


def _make_progress_fn(progress_callback: Optional[Callable[[str, float], None]]):
    """Возвращает (cumulative, stage_idx) -> wrapped callback который умножает
    прогресс этапа на его вес и сдвигает на cumulative."""
    cumulative_ref = {"value": 0.0, "stage_idx": 0}

    def start_stage(stage_idx: int) -> Callable[[float], None]:
        cumulative_ref["stage_idx"] = stage_idx
        stage_name, stage_weight = STAGES[stage_idx]
        base = cumulative_ref["value"]

        def cb(stage_frac: float) -> None:
            if progress_callback is None:
                return
            total = base + stage_weight * max(0.0, min(1.0, stage_frac))
            progress_callback(stage_name, total)

        return cb

    def finish_stage() -> None:
        _, stage_weight = STAGES[cumulative_ref["stage_idx"]]
        cumulative_ref["value"] = min(1.0, cumulative_ref["value"] + stage_weight)

    def overall(stage_name: str, fraction: float = 0.0) -> None:
        if progress_callback is None:
            return
        progress_callback(stage_name, cumulative_ref["value"] + fraction)

    return start_stage, finish_stage, overall, cumulative_ref


def run_inference(
    video_path: str,
    job_id: str,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """Полный pipeline на одно видео. Возвращает dict в формате результата API.

    Структура:
        {
          "video_url": ".../highlights.mp4",
          "highlights_url": ".../highlights.mp4",
          "events": [...],
          "summary": {...},
          "dna": {...},
          "model_info": {...}
        }
    """
    video_path = str(Path(video_path).resolve())
    media_root = ml_settings.media_path
    results_dir = media_root / "results" / job_id
    results_dir.mkdir(parents=True, exist_ok=True)

    start_stage, finish_stage, overall, _ = _make_progress_fn(progress_callback)

    # 1. Pose extraction
    overall("pose_extraction", 0.001)
    poses_df = extract_poses(
        video_path,
        yolo_model_name=ml_settings.yolo_pose_model,
        yolo_conf=ml_settings.yolo_conf,
        yolo_iou=ml_settings.yolo_iou,
        frame_stride=ml_settings.frame_stride,
        progress_callback=start_stage(0),
    )
    finish_stage()

    if poses_df.empty:
        return _empty_result(job_id, results_dir, reason="no poses detected")

    # 2. Frame features
    overall("features", 0.001)
    features_df = build_frame_features(poses_df)
    start_stage(1)(1.0)  # фичи строятся быстро, сразу 100%
    finish_stage()

    # 3. Event detection
    overall("event_detection", 0.001)
    probs_df = predict_event_probabilities(features_df)
    events = probabilities_to_events(probs_df)
    start_stage(2)(1.0)
    finish_stage()

    # 4. Visual reranker
    overall("visual_rerank", 0.001)
    if has_visual_model() and not events.empty:
        events = rerank_with_visual(events, poses_df, video_path)
    else:
        events["vis_prob"] = events.get("event_score", 0.0)
    start_stage(3)(1.0)
    finish_stage()

    # 5. MLP reranker
    overall("mlp_rerank", 0.001)
    if has_mlp_model() and not events.empty:
        events = rerank_with_mlp(events, features_df)
    else:
        events["mlp_prob"] = events.get("vis_prob", events.get("event_score", 0.0))
    start_stage(4)(1.0)
    finish_stage()

    # 6. Attributes
    overall("attributes", 0.001)
    events = predict_attributes(events, features_df)
    start_stage(5)(1.0)
    finish_stage()

    # 7. Postprocess (rank blend + NMS)
    overall("postprocess", 0.001)
    events = postprocess_events(events)
    start_stage(6)(1.0)
    finish_stage()

    # 8. Highlights render
    overall("highlights_render", 0.001)
    highlights_path = results_dir / "highlights.mp4"
    built = build_highlights_reel(
        video_path,
        str(highlights_path),
        events,
        pad_before=ml_settings.highlight_pad_before,
        pad_after=ml_settings.highlight_pad_after,
        progress_callback=start_stage(7),
        overlay_pose_df=poses_df,
        draw_boxes=False,
        draw_event_labels=True,
    )
    finish_stage()

    # Финальная агрегация
    summary = compute_summary(events)
    dna = compute_dna(events, features_df)

    fps = float(features_df.attrs.get("fps", 30.0))

    events_out = []
    for _, ev in events.iterrows():
        events_out.append({
            "time": round(float(ev["frame"]) / fps, 3),
            "frame": int(ev["frame"]),
            "fighter": str(ev["fighter"]),
            "hand": str(ev.get("hand", "right")),
            "punch_type": str(ev.get("punch_type", "jab")),
            "target": str(ev.get("target", "head")),
            "effectiveness": str(ev.get("effectiveness", "landed")),
            "confidence": round(float(ev.get("final_score", ev.get("event_score", 0.0))), 3),
        })

    highlights_rel = f"/media/results/{job_id}/highlights.mp4" if built else ""

    result = {
        "video_url": highlights_rel,
        "highlights_url": highlights_rel,
        "events": events_out,
        "summary": summary,
        "dna": dna,
        "model_info": {
            "event_model": "real" if has_event_model() else "heuristic",
            "attribute_model": "real" if has_attribute_models() else "heuristic",
            "visual_reranker": "real" if has_visual_model() else "off",
            "mlp_reranker": "real" if has_mlp_model() else "off",
            "yolo_model": ml_settings.yolo_pose_model,
        },
    }

    (results_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _empty_result(job_id: str, results_dir: Path, reason: str) -> dict:
    """Если пайплайн не нашёл ни поз ни ударов — возвращаем пустую но валидную структуру."""
    return {
        "video_url": "",
        "highlights_url": "",
        "events": [],
        "summary": {
            "red":  {"punches": 0, "accuracy": 0.0, "landed": 0, "blocked": 0, "missed": 0},
            "blue": {"punches": 0, "accuracy": 0.0, "landed": 0, "blocked": 0, "missed": 0},
        },
        "dna": {
            "red":  {"aggression": 0, "defense": 0, "mobility": 0, "pressure": 0},
            "blue": {"aggression": 0, "defense": 0, "mobility": 0, "pressure": 0},
        },
        "model_info": {"reason": reason},
    }
