#!/usr/bin/env python3
"""Materialize nine merger conditions from each native feature sequence."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from followup_selectors import physical_max_span, select_base, validate_starts
from trained_selectors import (
    BASE_SYSTEMS,
    MAIN_SYSTEMS,
    TRAINED_SYSTEMS,
    load_trained_artifacts,
    normalize_features,
    select_trained,
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_calibration(value: str) -> tuple[float, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("calibration must be RATE=PATH")
    rate, path = value.split("=", 1)
    return float(rate), Path(path)


def rate_label(rate: float) -> str:
    return f"{rate:g}".replace(".", "p")


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
    parser.add_argument(
        "--calibration", action="append", type=parse_calibration, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictor-dir", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--kmeans-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assignment", type=Path)
    parser.add_argument("--system", action="append")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-utterances", type=int)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be inside [0, num-shards)")

    calibrations = {}
    for requested_rate, path in args.calibration:
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual_rate = float(payload["target_rate_hz"])
        if abs(actual_rate - requested_rate) > 1e-9:
            raise RuntimeError(
                f"calibration rate mismatch: requested={requested_rate}, "
                f"file={actual_rate}"
            )
        if payload.get("status") != "passed":
            raise RuntimeError(f"calibration did not pass: {path}")
        calibrations[requested_rate] = payload["parameters"]
    rates = tuple(sorted(calibrations))

    requested = tuple(args.system or MAIN_SYSTEMS)
    unknown = sorted(set(requested) - set(MAIN_SYSTEMS))
    if unknown:
        raise RuntimeError(f"unsupported systems: {unknown}")
    assignments = None
    if args.assignment:
        assignments = {
            str(row["utt_id"]): tuple(row["systems"])
            for row in read_jsonl(args.assignment)
        }

    all_rows = list(read_jsonl(args.feature_index))
    if args.max_utterances is not None:
        all_rows = all_rows[: args.max_utterances]
    rows = [
        row
        for index, row in enumerate(all_rows)
        if index % args.num_shards == args.shard_index
    ]
    if not rows:
        raise RuntimeError("selected feature shard is empty")
    artifacts = load_trained_artifacts(
        args.predictor_dir,
        args.normalization,
        args.kmeans_dir,
        args.device,
        int(rows[0]["feature_dim"]),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    counts = Counter()
    segment_sums = Counter()
    duration_sums = Counter()
    started = time.perf_counter()
    with temporary.open("w", encoding="utf-8") as handle:
        for completed, row in enumerate(rows, 1):
            systems = (
                assignments.get(str(row["utt_id"]), ())
                if assignments is not None
                else requested
            )
            systems = tuple(system for system in systems if system in requested)
            if not systems:
                raise RuntimeError(f"{row['utt_id']} has no requested systems")
            features = load_feature(row)
            duration = float(row["duration_sec"])
            frames = len(features)
            frame_seconds = duration / frames
            max_span = physical_max_span(frames, duration)
            normalized = (
                normalize_features(features, artifacts)
                if set(systems) & set(TRAINED_SYSTEMS)
                else None
            )
            trained_cache: dict[str, np.ndarray] = {}
            for target_rate in rates:
                controls = calibrations[target_rate]
                selections = {}
                for system in systems:
                    if system in BASE_SYSTEMS:
                        selections[system] = select_base(
                            system,
                            features,
                            duration,
                            target_rate,
                            max_span,
                            controls,
                        )
                    else:
                        selections[system] = select_trained(
                            system,
                            normalized,
                            duration,
                            target_rate,
                            max_span,
                            controls,
                            artifacts,
                            trained_cache,
                        )
                for system in systems:
                    starts = selections[system].starts
                    validate_starts(starts, frames, max_span)
                    ends = [*starts[1:], frames]
                    lengths = [end - start for start, end in zip(starts, ends)]
                    key = f"{system}@{rate_label(target_rate)}hz"
                    output = {
                        **row,
                        "system": system,
                        "condition": f"native_{rate_label(target_rate)}hz",
                        "condition_key": key,
                        "target_rate_hz": target_rate,
                        "starts": starts,
                        "segment_lengths": lengths,
                        "segment_durations_sec": [
                            length * frame_seconds for length in lengths
                        ],
                        "realized_rate_hz": len(starts) / duration,
                        "max_span_frames": max_span,
                        "max_span_seconds": max_span * frame_seconds,
                        "duration_unit": "seconds",
                        "trained_merger_extension": system in TRAINED_SYSTEMS,
                    }
                    handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                    counts[key] += 1
                    segment_sums[key] += len(starts)
                    duration_sums[key] += duration
            if completed % 100 == 0 or completed == len(rows):
                elapsed = time.perf_counter() - started
                print(
                    f"shard {args.shard_index}: {completed}/{len(rows)} "
                    f"({completed / max(elapsed, 1e-6):.2f} utt/s)",
                    flush=True,
                )
    temporary.replace(args.output)

    report = {
        "status": "passed",
        "protocol": "native_grid_nine_main_mergers_v1",
        "records": sum(counts.values()),
        "utterances": len(rows),
        "rates_hz": list(rates),
        "systems": list(requested),
        "assignment": str(args.assignment) if args.assignment else None,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "condition_counts": dict(sorted(counts.items())),
        "corpus_realized_rate_hz": {
            key: segment_sums[key] / duration_sums[key]
            for key in sorted(counts)
        },
    }
    args.output.with_suffix(args.output.suffix + ".validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(args.output.suffix + ".exit").write_text(
        "0\n", encoding="ascii"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
