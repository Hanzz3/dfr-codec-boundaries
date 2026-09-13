#!/usr/bin/env python3
"""Calibrate native-grid base and trained mergers on label-free validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_rates import calibrate_ple, calibrate_similarity
from common_selectors import cosine_change
from followup_selectors import physical_max_span
from kmeans_selector import code_change_scores
from trained_selectors import load_trained_artifacts, normalize_features


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_feature(row: dict) -> np.ndarray:
    with np.load(row["feature_path"]) as payload:
        values = payload[str(row.get("feature_key", "features"))].astype(
            np.float32
        )
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid feature: {row['utt_id']}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-rate", type=float, required=True)
    parser.add_argument("--predictor-dir", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--kmeans-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument(
        "--protocol",
        default="libritts_dev_feature_only_nine_merger_native_grid",
    )
    parser.add_argument("--max-rate-error", type=float, default=0.05)
    args = parser.parse_args()

    rows = list(read_jsonl(args.feature_index))
    if args.max_utterances is not None:
        rows = rows[: args.max_utterances]
    if not rows:
        raise RuntimeError("empty calibration feature index")
    feature_dim = int(rows[0]["feature_dim"])
    artifacts = load_trained_artifacts(
        args.predictor_dir,
        args.normalization,
        args.kmeans_dir,
        args.device,
        feature_dim,
    )

    cosine_trajectories = []
    dcdit_trajectories = []
    kmeans_trajectories = []
    duration_total = 0.0
    for completed, row in enumerate(rows, 1):
        features = load_feature(row)
        duration = float(row["duration_sec"])
        max_span = physical_max_span(len(features), duration)
        normalized = normalize_features(features, artifacts)
        cosine_trajectories.append(
            (cosine_change(features), duration, max_span)
        )
        dcdit_trajectories.append(
            (artifacts.dcdit.difficulty(normalized, artifacts.device), duration, max_span)
        )
        _codes, kmeans_scores = code_change_scores(normalized, artifacts.kmeans)
        kmeans_trajectories.append((kmeans_scores, duration, max_span))
        duration_total += duration
        if completed % 250 == 0 or completed == len(rows):
            print(f"calibration scores: {completed}/{len(rows)}", flush=True)

    similarity, similarity_rate = calibrate_similarity(
        cosine_trajectories, args.target_rate
    )
    ple, ple_rate = calibrate_ple(cosine_trajectories, args.target_rate)
    dcdit, dcdit_rate = calibrate_similarity(
        dcdit_trajectories, args.target_rate
    )
    kmeans, kmeans_rate = calibrate_similarity(
        kmeans_trajectories, args.target_rate
    )
    realized = {
        "similarity": similarity_rate,
        "ple": ple_rate,
        "dcdit_1d": dcdit_rate,
        "hubert_kmeans": kmeans_rate,
    }
    errors = {
        system: abs(rate - args.target_rate)
        for system, rate in realized.items()
    }
    if any(error > args.max_rate_error for error in errors.values()):
        raise RuntimeError(
            f"calibration missed target rate: errors={errors}, "
            f"limit={args.max_rate_error}"
        )
    payload = {
        "status": "passed",
        "protocol": args.protocol,
        "label_free": True,
        "labels_or_transcripts_read": False,
        "row_fields_used": ["feature_path", "feature_key", "duration_sec"],
        "target_rate_hz": args.target_rate,
        "records": len(rows),
        "duration_sec": duration_total,
        "max_rate_error_hz": args.max_rate_error,
        "parameters": {
            "similarity": {
                "name": "cosine_change_threshold",
                "value": similarity,
                "realized_rate_hz": similarity_rate,
            },
            "ple": {
                "name": "cumulative_path_tau",
                "value": ple,
                "realized_rate_hz": ple_rate,
            },
            "dcdit_1d": {
                "name": "difficulty_threshold",
                "value": dcdit,
                "realized_rate_hz": dcdit_rate,
            },
            "hubert_kmeans": {
                "name": "code_change_confidence_threshold",
                "value": kmeans,
                "realized_rate_hz": kmeans_rate,
            },
        },
        "absolute_rate_error_hz": errors,
        "exact_budget_systems": [
            "uniform",
            "codecslime_dp",
            "peak_detection",
            "elastic_time_greedy",
            "elastic_time_dp",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".exit").write_text(
        "0\n", encoding="ascii"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
