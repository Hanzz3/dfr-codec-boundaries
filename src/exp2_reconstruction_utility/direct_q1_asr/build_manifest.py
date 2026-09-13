#!/usr/bin/env python3
"""Build a compact, validated LM-ASR manifest from semantic-token predictions."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_prediction_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be SYSTEM=PATH")
    system, path = value.split("=", 1)
    if not system or not path:
        raise argparse.ArgumentTypeError("prediction must be SYSTEM=PATH")
    return system, Path(path)


def validate_starts(starts: list[int], frames: int, key: tuple[str, str]) -> list[int]:
    values = [int(value) for value in starts]
    if not values or values[0] != 0:
        raise RuntimeError(f"{key}: starts must begin at frame zero")
    if values != sorted(set(values)):
        raise RuntimeError(f"{key}: starts must be strictly increasing")
    if values[-1] >= frames:
        raise RuntimeError(f"{key}: final start {values[-1]} exceeds {frames} frames")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_prediction_spec,
        required=True,
        help="Repeat SYSTEM=PATH. A shared path may contain multiple systems.",
    )
    parser.add_argument("--condition", default="rate_6_25")
    parser.add_argument("--split", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-per-system", type=int)
    args = parser.parse_args()

    requested_splits = set(args.split)
    feature_records = {
        row["utt_id"]: row
        for row in read_jsonl(args.feature_index)
        if row.get("split") in requested_splits
    }
    if not feature_records:
        raise RuntimeError("feature index has no requested records")

    systems = [system for system, _ in args.prediction]
    if len(systems) != len(set(systems)):
        raise RuntimeError(f"duplicate requested systems: {systems}")
    systems_by_path: dict[Path, set[str]] = {}
    for system, path in args.prediction:
        systems_by_path.setdefault(path, set()).add(system)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    counts: Counter[str] = Counter()
    split_counts: Counter[tuple[str, str]] = Counter()
    observed: set[tuple[str, str]] = set()
    realized_rate_sum: Counter[str] = Counter()

    with temporary.open("w", encoding="utf-8") as destination:
        for path, allowed_systems in systems_by_path.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            for prediction in read_jsonl(path):
                system = prediction.get("system")
                if system not in allowed_systems:
                    continue
                if prediction.get("condition") != args.condition:
                    continue
                split = prediction.get("split")
                if split not in requested_splits:
                    continue
                utt_id = prediction["utt_id"]
                feature = feature_records.get(utt_id)
                if feature is None:
                    raise RuntimeError(f"{system}/{utt_id}: missing feature record")
                key = (system, utt_id)
                if key in observed:
                    raise RuntimeError(f"duplicate prediction: {key}")
                observed.add(key)
                frames = int(feature["frames"])
                starts = validate_starts(prediction["starts"], frames, key)
                lengths = [
                    end - start
                    for start, end in zip(starts, [*starts[1:], frames])
                ]
                if any(length <= 0 for length in lengths) or sum(lengths) != frames:
                    raise RuntimeError(f"{key}: invalid segment coverage")
                transcript = prediction.get("transcript") or feature.get("transcript")
                if not transcript:
                    raise RuntimeError(f"{key}: missing transcript")
                row = {
                    "utt_id": utt_id,
                    "system": system,
                    "condition": args.condition,
                    "split": split,
                    "transcript": transcript,
                    "feature_path": feature["feature_path"],
                    "frames": frames,
                    "feature_dim": int(feature["feature_dim"]),
                    "duration_sec": float(feature["duration_sec"]),
                    "realized_rate_hz": float(prediction["realized_rate_hz"]),
                    "starts": starts,
                    "segment_lengths": lengths,
                }
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[system] += 1
                split_counts[(system, split)] += 1
                realized_rate_sum[system] += row["realized_rate_hz"]

    expected_systems = set(systems)
    if set(counts) != expected_systems:
        raise RuntimeError(f"missing systems: counts={counts}, expected={expected_systems}")
    if args.expected_per_system is not None:
        bad = {
            system: counts[system]
            for system in systems
            if counts[system] != args.expected_per_system
        }
        if bad:
            raise RuntimeError(
                f"unexpected records per system: {bad}; expected {args.expected_per_system}"
            )

    temporary.replace(args.output)
    summary = {
        "status": "passed",
        "condition": args.condition,
        "systems": systems,
        "splits": sorted(requested_splits),
        "feature_records": len(feature_records),
        "rows": sum(counts.values()),
        "counts": dict(sorted(counts.items())),
        "split_counts": {
            f"{system}/{split}": count
            for (system, split), count in sorted(split_counts.items())
        },
        "mean_realized_rate_hz": {
            system: realized_rate_sum[system] / counts[system] for system in systems
        },
    }
    validation_path = args.output.with_suffix(args.output.suffix + ".validation.json")
    validation_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
