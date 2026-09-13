#!/usr/bin/env python3
"""Shared helpers for sharded boundary-alignment runs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import reference_tracks as core


TARGET_TOLERANCES = {
    "phoneme": [20.0, 40.0, 80.0],
    "syllable": [40.0, 80.0, 160.0],
    "bpe_subword": [40.0, 80.0, 160.0],
    "word": [40.0, 80.0, 160.0],
    "acoustic_event": [40.0, 80.0, 160.0],
    "vuv": [40.0, 80.0, 160.0],
}

CONFUSION_TOLERANCES = {
    "phoneme": 0.04,
    "syllable": 0.08,
    "bpe_subword": 0.08,
    "word": 0.08,
    "acoustic_event": 0.08,
    "vuv": 0.08,
}

# Exp1-3 v4 uses one common collar so encoder and boundary-type comparisons are
# not changed by category-specific tolerance choices. Legacy runs keep the
# curves above for reproducibility.
EXP123_PRIMARY_TOLERANCE_MS = 40.0
EXP123_TARGET_TOLERANCES = {
    target: [EXP123_PRIMARY_TOLERANCE_MS]
    for target in TARGET_TOLERANCES
}
EXP123_CONFUSION_TOLERANCES = {
    target: EXP123_PRIMARY_TOLERANCE_MS / 1000.0
    for target in TARGET_TOLERANCES
}


def boundary_set_to_dict(boundaries: core.BoundarySet) -> dict[str, list[float]]:
    return {
        "phoneme": boundaries.phoneme,
        "syllable": boundaries.syllable,
        "bpe_subword": boundaries.bpe_subword,
        "word": boundaries.word,
        "acoustic_event": boundaries.acoustic_event,
        "vuv": boundaries.vuv,
        "silence_phone": boundaries.silence_phone,
    }


def boundary_set_from_dict(payload: dict) -> core.BoundarySet:
    return core.BoundarySet(
        phoneme=list(payload.get("phoneme", [])),
        syllable=list(payload.get("syllable", [])),
        bpe_subword=list(payload.get("bpe_subword", [])),
        word=list(payload.get("word", [])),
        acoustic_event=list(payload.get("acoustic_event", [])),
        vuv=list(payload.get("vuv", [])),
        silence_phone=list(payload.get("silence_phone", [])),
    )


def read_ground_truth(path: Path, max_utterances: int | None = None) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
            if max_utterances is not None and len(items) >= max_utterances:
                break
    return items


def gt_targets(boundaries: core.BoundarySet) -> dict[str, list[float]]:
    return {
        "phoneme": boundaries.phoneme,
        "syllable": boundaries.syllable,
        "bpe_subword": boundaries.bpe_subword,
        "word": boundaries.word,
        "acoustic_event": boundaries.acoustic_event,
        "vuv": boundaries.vuv,
    }


def evaluate_extractors(
    gt_items: list[dict],
    extractors: list[object],
    flexicodec_rate_hz: float,
    run_label: str,
) -> dict[str, object]:
    metric_rows: list[dict] = []
    ap_store: dict[tuple[str, str, float], dict] = defaultdict(lambda: {"labels": [], "scores": []})
    confusion_counts: dict[str, Counter] = defaultdict(Counter)
    hybrid_counts: dict[str, Counter] = defaultdict(Counter)
    runtime_counts: Counter[str] = Counter()

    for row_idx, item in enumerate(gt_items, 1):
        row = item.get("manifest_row") or item
        y, sr = core.load_audio(item["audio"])
        boundaries = boundary_set_from_dict(item["boundaries"])
        targets = gt_targets(boundaries)
        duration = float(item.get("duration_sec") or len(y) / sr)

        for extractor in extractors:
            try:
                feature_map = extractor(row, y, sr)
            except Exception as exc:  # noqa: BLE001
                runtime_counts[f"{extractor.__class__.__name__}:error:{type(exc).__name__}:{exc}"] += 1
                continue

            for enc_name, payload in feature_map.items():
                pred_override = None
                frame_rate_override = None
                score_times_override = None
                scores_override = None
                requested_segments = None
                actual_segments = None
                realized_rate_hz = None
                canonical_grid = False
                if isinstance(payload, dict):
                    feats = payload["features"]
                    feat_times = payload["times"]
                    pred_override = payload.get("pred_times")
                    frame_rate_override = payload.get("frame_rate_hz")
                    score_times_override = payload.get("score_times")
                    scores_override = payload.get("scores")
                    requested_segments = payload.get("requested_segments")
                    actual_segments = payload.get("actual_segments")
                    realized_rate_hz = payload.get("realized_rate_hz")
                    canonical_grid = bool(payload.get("canonical_grid", False))
                else:
                    feats, feat_times = payload

                feats = np.asarray(feats, dtype=np.float32)
                feat_times = np.asarray(feat_times, dtype=np.float32)
                if pred_override is None and not canonical_grid:
                    feats, feat_times = core.canonicalize_feature_grid(feats, duration)
                if score_times_override is not None and scores_override is not None:
                    score_times = np.asarray(score_times_override, dtype=np.float32)
                    scores = np.asarray(scores_override, dtype=np.float32)
                else:
                    score_times, scores = core.delta_similarity(
                        feats,
                        feat_times,
                        normalize_features=False,
                        normalize_scores=False,
                    )
                if len(scores) == 0 and pred_override is None:
                    runtime_counts[f"{enc_name}:empty_scores"] += 1
                    continue

                if pred_override is not None:
                    pred_times = np.asarray(pred_override, dtype=np.float32)
                else:
                    pred_times, _, selection = core.select_flexicodec_boundaries(
                        feat_times,
                        scores,
                        duration=duration,
                        target_rate_hz=flexicodec_rate_hz,
                        max_span=8,
                    )
                    requested_segments = selection["requested_segments"]
                    actual_segments = selection["actual_segments"]
                    realized_rate_hz = selection["realized_rate_hz"]
                if actual_segments is None:
                    actual_segments = len(pred_times) + 1
                if realized_rate_hz is None:
                    realized_rate_hz = actual_segments / max(duration, 1e-8)

                for target, gt_times in targets.items():
                    for tol_ms in TARGET_TOLERANCES[target]:
                        tol = tol_ms / 1000.0
                        tp, fp, fn, offsets = core.match_boundaries(pred_times, gt_times, tol)
                        labels = core.frame_labels(score_times, gt_times, tol)
                        key = (enc_name, target, tol_ms)
                        ap_store[key]["labels"].extend(labels.tolist())
                        ap_store[key]["scores"].extend(scores.tolist())
                        metric_rows.append(
                            {
                                "utt_id": item["utt_id"],
                                "encoder": enc_name,
                                "target": target,
                                "tolerance_ms": tol_ms,
                                "gt": len(gt_times),
                                "pred": len(pred_times),
                                "tp": tp,
                                "fp": fp,
                                "fn": fn,
                                "offsets_ms": [x * 1000.0 for x in offsets],
                                "frame_rate_hz": frame_rate_override
                                if frame_rate_override is not None
                                else len(feat_times) / max(duration, 1e-6),
                                "flexicodec_rate_hz": flexicodec_rate_hz,
                                "requested_segments": requested_segments,
                                "actual_segments": actual_segments,
                                "realized_rate_hz": realized_rate_hz,
                                "time_grid": "frame_centers",
                                "overlap_definition": "jaccard_iou",
                                "run_label": run_label,
                            }
                        )

                for pred_t in pred_times:
                    types = core.overlapping_types(float(pred_t), boundaries, CONFUSION_TOLERANCES)
                    if not types:
                        types = ["unmatched"]
                    confusion_counts[enc_name][types[0]] += 1
                    hybrid_counts[enc_name]["+".join(types)] += 1

        if row_idx % 10 == 0:
            print(f"{run_label}: processed {row_idx}/{len(gt_items)}", flush=True)

    agg = core.aggregate_metrics(metric_rows, ap_store)
    confusion_rows = []
    for encoder, counts in confusion_counts.items():
        total = sum(counts.values()) or 1
        for category, count in counts.items():
            confusion_rows.append({"encoder": encoder, "category": category, "count": count, "ratio": count / total})

    hybrid_rows = []
    for encoder, counts in hybrid_counts.items():
        total = sum(counts.values()) or 1
        for explanation, count in counts.items():
            hybrid_rows.append({"encoder": encoder, "explanation": explanation, "count": count, "ratio": count / total})

    per_utt = pd.DataFrame(metric_rows)
    if not per_utt.empty:
        per_utt["offsets_ms"] = per_utt["offsets_ms"].map(json.dumps)
    return {
        "per_utt": per_utt,
        "agg": agg,
        "confusion": pd.DataFrame(confusion_rows),
        "hybrid": pd.DataFrame(hybrid_rows),
        "runtime_counts": runtime_counts,
    }


def write_status(path: Path, statuses: list[dict], runtime_counts: Counter[str]) -> None:
    rows = list(statuses)
    for key, value in sorted(runtime_counts.items()):
        rows.append({"encoder": key, "family": "runtime", "status": "runtime_count", "count": value})
    pd.DataFrame(rows).to_csv(path, index=False)


def main_tolerance_rows(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return agg
    return agg[agg.apply(lambda row: float(row["tolerance_ms"]) == core.main_tolerance(str(row["target"])), axis=1)]
