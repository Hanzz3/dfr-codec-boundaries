#!/usr/bin/env python3
"""Calibrate A-ToMe merge ratios and TADPC thresholds on held-out speech."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from supplement_selectors import atome_style_starts, tadpc_style_starts


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    marker = path.with_suffix(path.suffix + ".exit")
    temporary = marker.with_name(marker.name + f".tmp.{os.getpid()}")
    temporary.write_text("0\n", encoding="ascii")
    os.replace(temporary, marker)


def output_count(frames: int, ratios: tuple[float, ...]) -> int:
    count = frames
    for ratio in ratios:
        count -= min(int(math.floor(count * ratio)), count // 2)
    return count


def calibrate_atome(
    lengths: list[tuple[int, float]], target: float
) -> tuple[tuple[float, ...], float]:
    total_duration = sum(duration for _, duration in lengths)

    def rate(ratios: tuple[float, ...]) -> float:
        return sum(output_count(frames, ratios) for frames, _ in lengths) / total_duration

    prefix: tuple[float, ...] = ()
    while rate(prefix + (0.5,)) > target:
        prefix += (0.5,)
        if len(prefix) >= 3:
            break
    candidates = np.linspace(0.0, 0.5, 5001)
    best_ratio = min(
        (float(value) for value in candidates),
        key=lambda value: (abs(rate(prefix + (value,)) - target), value),
    )
    ratios = prefix + (best_ratio,)
    if best_ratio == 0.0:
        ratios = prefix
    return ratios, rate(ratios)


def calibrate_tadpc(
    features: list[tuple[np.ndarray, float]], target: float, max_span: int
) -> tuple[float, float, list[dict]]:
    total_duration = sum(duration for _, duration in features)
    history: list[dict] = []

    def evaluate(threshold: float) -> float:
        segments = sum(
            len(
                tadpc_style_starts(
                    matrix,
                    threshold=threshold,
                    tadpc_max_span=max_span,
                ).starts
            )
            for matrix, _ in features
        )
        observed = segments / total_duration
        history.append(
            {
                "threshold": float(threshold),
                "segments": int(segments),
                "realized_rate_hz": float(observed),
                "absolute_error_hz": abs(float(observed) - target),
            }
        )
        return float(observed)

    low, high = -1.0, 1.0
    low_rate, high_rate = evaluate(low), evaluate(high)
    if not low_rate <= target <= high_rate:
        raise RuntimeError(
            f"TADPC cannot bracket {target:g} Hz: {low_rate:g}--{high_rate:g} Hz"
        )
    for _ in range(24):
        middle = (low + high) / 2.0
        if evaluate(middle) < target:
            low = middle
        else:
            high = middle
    best = min(history, key=lambda row: (row["absolute_error_hz"], row["threshold"]))
    return float(best["threshold"]), float(best["realized_rate_hz"]), history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--target-rate-hz", type=float, required=True)
    parser.add_argument("--tadpc-max-span", type=int, default=8)
    parser.add_argument("--atome-output", type=Path, required=True)
    parser.add_argument("--tadpc-output", type=Path, required=True)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.feature_index) if row.get("split") == args.split]
    if not rows:
        raise RuntimeError(f"no feature records for split={args.split!r}")
    items: list[tuple[np.ndarray, float]] = []
    lengths: list[tuple[int, float]] = []
    for row in rows:
        with np.load(row["feature_path"]) as payload:
            matrix = payload["features"].astype(np.float32)
        if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
            raise RuntimeError(f"invalid feature cache: {row['utt_id']}")
        duration = float(row["duration_sec"])
        items.append((matrix, duration))
        lengths.append((len(matrix), duration))

    ratios, atome_rate = calibrate_atome(lengths, args.target_rate_hz)
    threshold, tadpc_rate, history = calibrate_tadpc(
        items, args.target_rate_hz, args.tadpc_max_span
    )
    common = {
        "status": "passed",
        "feature_index": str(args.feature_index),
        "feature_index_sha256": sha256(args.feature_index),
        "calibration_split": args.split,
        "utterances": len(rows),
        "duration_sec": sum(duration for _, duration in lengths),
        "target_rate_hz": args.target_rate_hz,
        "selection_uses_test": False,
        "selection_uses_gt": False,
        "selection_uses_transcript": False,
        "selection_uses_utility": False,
    }
    atomic_json(
        args.atome_output,
        {
            **common,
            "protocol": "atome_official_fixed_merge_ratio_heldout_rate_v2",
            "merge_ratios": list(ratios),
            "realized_rate_hz": atome_rate,
            "absolute_error_hz": abs(atome_rate - args.target_rate_hz),
            "maximum_native_span_frames": 2 ** len(ratios),
        },
    )
    atomic_json(
        args.tadpc_output,
        {
            **common,
            "protocol": "varstok_tadpc_fixed_global_threshold_heldout_rate_v2",
            "threshold": threshold,
            "realized_rate_hz": tadpc_rate,
            "absolute_error_hz": abs(tadpc_rate - args.target_rate_hz),
            "official_parameters": {"k": 10, "beta": 0.2},
            "maximum_native_span_frames": args.tadpc_max_span,
            "history": history,
        },
    )
    print(
        json.dumps(
            {
                "target_rate_hz": args.target_rate_hz,
                "atome_ratios": ratios,
                "atome_rate_hz": atome_rate,
                "tadpc_threshold": threshold,
                "tadpc_rate_hz": tadpc_rate,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
