#!/usr/bin/env python3
"""Calibrate native-grid similarity and PLE controls on LibriTTS validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common_selectors import cosine_change, ple_fixed_tau_scores, threshold_starts
from protocol import TARGET_RATE_HZ, physical_max_span


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_feature(row: dict) -> np.ndarray:
    with np.load(row["feature_path"]) as payload:
        values = payload[str(row.get("feature_key", "features"))].astype(np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid feature: {row['utt_id']}")
    return values


def corpus_rate(
    trajectories: list[tuple[np.ndarray, float, int]],
    parameter: float,
    system: str,
) -> float:
    segments = 0
    duration = 0.0
    for scores, seconds, max_span in trajectories:
        if system == "similarity":
            starts = threshold_starts(scores, parameter, max_span)
        elif system == "ple":
            starts = ple_fixed_tau_scores(scores, parameter, max_span)
        else:
            raise KeyError(system)
        segments += len(starts)
        duration += seconds
    return segments / max(duration, 1e-12)


def calibrate_similarity(
    trajectories: list[tuple[np.ndarray, float, int]],
    target_rate: float,
) -> tuple[float, float]:
    finite = [scores[np.isfinite(scores)] for scores, _duration, _span in trajectories]
    unique = np.unique(np.concatenate(finite)) if finite else np.asarray([0.0])
    scale = max(1.0, float(np.max(np.abs(unique))))
    epsilon = np.finfo(np.float64).eps * scale * 8.0
    candidates = np.concatenate(([unique[0] - epsilon], unique, [unique[-1] + epsilon]))
    cache: dict[int, float] = {}

    def rate(index: int) -> float:
        if index not in cache:
            cache[index] = corpus_rate(
                trajectories, float(candidates[index]), "similarity"
            )
        return cache[index]

    low, high = 0, len(candidates) - 1
    while low < high:
        middle = (low + high) // 2
        if rate(middle) <= target_rate:
            high = middle
        else:
            low = middle + 1
    nearby = range(max(0, low - 3), min(len(candidates), low + 4))
    best = min(
        nearby,
        key=lambda index: (
            abs(rate(index) - target_rate),
            rate(index),
            -float(candidates[index]),
        ),
    )
    return float(candidates[best]), float(rate(best))


def calibrate_ple(
    trajectories: list[tuple[np.ndarray, float, int]],
    target_rate: float,
) -> tuple[float, float]:
    maximum = max(
        float(np.maximum(np.nan_to_num(scores, nan=0.0), 0.0).sum())
        for scores, _duration, _span in trajectories
    )
    low = max(np.finfo(np.float64).eps, maximum * 1e-12)
    high = max(1.0, maximum + 1.0)
    for _ in range(60):
        middle = (low + high) / 2.0
        if corpus_rate(trajectories, middle, "ple") > target_rate:
            low = middle
        else:
            high = middle
    candidates = (low, (low + high) / 2.0, high)
    scored = [
        (value, corpus_rate(trajectories, value, "ple")) for value in candidates
    ]
    return min(
        scored,
        key=lambda item: (abs(item[1] - target_rate), item[1], -item[0]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-rate", type=float, default=TARGET_RATE_HZ)
    parser.add_argument("--max-utterances", type=int)
    args = parser.parse_args()

    rows = read_jsonl(args.feature_index)
    validation = [
        row
        for row in rows
        if str(row.get("split")) in {"validation", "val"}
        or str(row.get("source_split", "")).startswith("dev-")
    ]
    if args.max_utterances is not None:
        validation = validation[: args.max_utterances]
    if not validation:
        raise RuntimeError("feature index has no validation records")

    trajectories = []
    for completed, row in enumerate(validation, 1):
        features = load_feature(row)
        duration = float(row["duration_sec"])
        trajectories.append(
            (
                cosine_change(features),
                duration,
                physical_max_span(len(features), duration),
            )
        )
        if completed % 500 == 0 or completed == len(validation):
            print(f"calibration scores: {completed}/{len(validation)}", flush=True)

    similarity, similarity_rate = calibrate_similarity(
        trajectories, args.target_rate
    )
    ple, ple_rate = calibrate_ple(trajectories, args.target_rate)
    payload = {
        "status": "passed",
        "protocol": "libritts_validation_native_grid_physical_max_span",
        "target_rate_hz": args.target_rate,
        "validation_records": len(validation),
        "validation_duration_sec": sum(item[1] for item in trajectories),
        "parameters": {
            "similarity": {
                "name": "cosine_change_threshold",
                "value": similarity,
                "realized_rate_hz": similarity_rate,
                "absolute_error_hz": abs(similarity_rate - args.target_rate),
            },
            "ple": {
                "name": "cumulative_path_tau",
                "value": ple,
                "realized_rate_hz": ple_rate,
                "absolute_error_hz": abs(ple_rate - args.target_rate),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
