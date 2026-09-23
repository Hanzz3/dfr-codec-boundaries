#!/usr/bin/env python3
"""Materialize the archived nine-method bundle used before final filtering."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from metrics import boundary_times, feature_nmse, match_boundaries
from supplement_selectors import atome_style_starts, tadpc_style_starts


OLD_SYSTEMS = (
    "uniform",
    "flexicodec_threshold",
    "codecslime_dp",
    "ple",
    "elastic_time_greedy",
    "elastic_time_dp",
    "dcdit_1d",
)
NEW_SYSTEMS = ("atome_style", "tadpc_style")
SYSTEMS = OLD_SYSTEMS + NEW_SYSTEMS
TARGETS = ("phoneme", "syllable", "bpe_subword", "word", "acoustic_event", "vuv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-predictions", type=Path, required=True)
    parser.add_argument("--old-point-metrics", type=Path, required=True)
    parser.add_argument("--old-segment-metrics", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--reference-tracks", "--ground-truth", dest="reference_tracks", type=Path, required=True)
    parser.add_argument("--atome-calibration", type=Path, required=True)
    parser.add_argument("--tadpc-calibration", type=Path, required=True)
    parser.add_argument("--target-rate-hz", type=float, required=True)
    parser.add_argument("--expected-utterances", type=int, default=1680)
    parser.add_argument("--collar-ms", type=float, default=40.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


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


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def calibration(path: Path, rate: float) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("calibration_split") != "val":
        raise RuntimeError(f"calibration is not accepted held-out validation: {path}")
    if abs(float(payload["target_rate_hz"]) - rate) > 1e-9:
        raise RuntimeError(f"calibration target mismatch: {path}")
    return payload


def main() -> int:
    args = parse_args()
    atome = calibration(args.atome_calibration, args.target_rate_hz)
    tadpc = calibration(args.tadpc_calibration, args.target_rate_hz)
    ratios = tuple(float(value) for value in atome["merge_ratios"])
    threshold = float(tadpc["threshold"])
    tadpc_max_span = int(tadpc["maximum_native_span_frames"])

    features = {
        str(row["utt_id"]): row
        for row in read_jsonl(args.feature_index)
        if row.get("split") == "test"
    }
    gt = {str(row["utt_id"]): row for row in read_jsonl(args.reference_tracks)}
    if len(features) != args.expected_utterances or set(features) - set(gt):
        raise RuntimeError("TIMIT TEST feature/GT coverage mismatch")

    old: dict[tuple[str, str], dict] = {}
    for row in read_jsonl(args.old_predictions):
        if str(row["system"]) not in OLD_SYSTEMS:
            continue
        if abs(float(row["rate_hz"]) - args.target_rate_hz) > 1e-9:
            continue
        key = (str(row["utt_id"]), str(row["system"]))
        if key in old:
            raise RuntimeError(f"duplicate accepted selector row: {key}")
        old[key] = row
    expected_old = args.expected_utterances * len(OLD_SYSTEMS)
    if len(old) != expected_old:
        raise RuntimeError(f"accepted selector coverage mismatch: {len(old)} != {expected_old}")

    old_points: list[dict] = []
    with args.old_point_metrics.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row["system"]) not in OLD_SYSTEMS:
                continue
            if abs(float(row["rate_hz"]) - args.target_rate_hz) > 1e-9:
                continue
            if abs(float(row["tolerance_ms"]) - args.collar_ms) > 1e-9:
                continue
            old_points.append(
                {
                    "utt_id": row["utt_id"],
                    "system": row["system"],
                    "target": row["target"],
                    "rate_hz": args.target_rate_hz,
                    "collar_ms": args.collar_ms,
                    "tp": int(row["tp"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                }
            )
    expected_points = args.expected_utterances * len(OLD_SYSTEMS) * len(TARGETS)
    if len(old_points) != expected_points:
        raise RuntimeError(f"accepted point coverage mismatch: {len(old_points)}")

    old_segments: dict[tuple[str, str], dict] = {}
    with args.old_segment_metrics.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row["system"]) not in OLD_SYSTEMS:
                continue
            if abs(float(row["rate_hz"]) - args.target_rate_hz) > 1e-9:
                continue
            key = (row["utt_id"], row["system"])
            old_segments[key] = {
                "utt_id": row["utt_id"],
                "system": row["system"],
                "rate_hz": args.target_rate_hz,
                "segments": int(float(row["segments"])),
                "realized_rate_hz": float(row["realized_rate_hz"]),
                "feature_nmse": float(row["feature_nmse"]),
                "max_segment_frames": int(float(row["max_segment_frames"])),
            }
    if len(old_segments) != expected_old:
        raise RuntimeError(f"accepted segment coverage mismatch: {len(old_segments)}")

    predictions: list[dict] = []
    point_rows = list(old_points)
    segment_rows = list(old_segments.values())
    for utt_id in sorted(features):
        record = features[utt_id]
        for system in OLD_SYSTEMS:
            row = dict(old[(utt_id, system)])
            row.update(
                {
                    "frames": int(record["frames"]),
                    "duration_sec": float(record["duration_sec"]),
                    "feature_path": record["feature_path"],
                    "audio": record["audio"],
                }
            )
            predictions.append(row)

        with np.load(record["feature_path"]) as payload:
            matrix = payload["features"].astype(np.float32)
            times = payload["times"].astype(np.float32)
        selections = {
            "atome_style": atome_style_starts(matrix, ratios),
            "tadpc_style": tadpc_style_starts(
                matrix,
                threshold=threshold,
                tadpc_max_span=tadpc_max_span,
            ),
        }
        for system, selection in selections.items():
            starts = list(selection.starts)
            ends = [*starts[1:], len(matrix)]
            lengths = [right - left for left, right in zip(starts, ends)]
            values = boundary_times(starts, times)
            row = {
                "utt_id": utt_id,
                "system": system,
                "rate_hz": args.target_rate_hz,
                "starts": starts,
                "boundary_times": values.tolist(),
                "segment_lengths": lengths,
                "frames": len(matrix),
                "duration_sec": float(record["duration_sec"]),
                "feature_path": record["feature_path"],
                "audio": record["audio"],
                "realized_rate_hz": len(starts) / float(record["duration_sec"]),
                "implementation": (
                    "atome_stacked_fixed_ratio_codec_grid_v2"
                    if system == "atome_style"
                    else "varstok_tadpc_fixed_threshold_codec_grid_v2"
                ),
                "control_mode": (
                    "heldout_fixed_merge_ratio"
                    if system == "atome_style"
                    else "heldout_global_threshold"
                ),
                "parameter_value": list(ratios) if system == "atome_style" else threshold,
            }
            predictions.append(row)
            segment_rows.append(
                {
                    "utt_id": utt_id,
                    "system": system,
                    "rate_hz": args.target_rate_hz,
                    "segments": len(starts),
                    "realized_rate_hz": row["realized_rate_hz"],
                    "feature_nmse": feature_nmse(matrix, starts),
                    "max_segment_frames": max(lengths),
                }
            )
            references = gt[utt_id]["boundaries"]
            for target in TARGETS:
                tp, fp, fn, _ = match_boundaries(
                    values,
                    references[target],
                    args.collar_ms / 1000.0,
                )
                point_rows.append(
                    {
                        "utt_id": utt_id,
                        "system": system,
                        "target": target,
                        "rate_hz": args.target_rate_hz,
                        "collar_ms": args.collar_ms,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }
                )

    counts = Counter(row["system"] for row in predictions)
    expected_counts = {system: args.expected_utterances for system in SYSTEMS}
    if dict(counts) != expected_counts:
        raise RuntimeError(f"manifest system coverage mismatch: {dict(counts)}")
    for row in predictions:
        lengths = row["segment_lengths"]
        if not row["starts"] or row["starts"][0] != 0 or max(lengths) > 8:
            raise RuntimeError(f"invalid boundary structure: {row['utt_id']}/{row['system']}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.output_dir / "boundary_predictions.jsonl"
    atomic_text(manifest, "".join(json.dumps(row, allow_nan=True) + "\n" for row in predictions))
    atomic_csv(args.output_dir / "point_metrics_40ms.csv", point_rows)
    atomic_csv(args.output_dir / "segment_metrics.csv", segment_rows)
    duration = sum(float(row["duration_sec"]) for row in features.values())
    rates = {
        system: sum(len(row["starts"]) for row in predictions if row["system"] == system)
        / duration
        for system in SYSTEMS
    }
    payload = {
        "status": "passed",
        "protocol": "exp2_codec_grid_core9_multirate_v2",
        "target_rate_hz": args.target_rate_hz,
        "collar_ms": args.collar_ms,
        "utterances": args.expected_utterances,
        "records": len(predictions),
        "point_rows": len(point_rows),
        "segment_rows": len(segment_rows),
        "systems": list(SYSTEMS),
        "corpus_realized_rate_hz": rates,
        "selection_uses_test_labels_or_utility": False,
        "manifest_sha256": sha256(manifest),
        "old_predictions_sha256": sha256(args.old_predictions),
        "old_point_metrics_sha256": sha256(args.old_point_metrics),
        "old_segment_metrics_sha256": sha256(args.old_segment_metrics),
    }
    atomic_text(args.output_dir / "acceptance.json", json.dumps(payload, indent=2) + "\n")
    atomic_text(args.output_dir / "acceptance.exit", "0\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
