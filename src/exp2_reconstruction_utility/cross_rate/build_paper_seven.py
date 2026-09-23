#!/usr/bin/env python3
"""Combine the archived bundle with Entropy-Mass into the paper's seven methods."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from paper_protocol import PAPER_SYSTEMS


TARGETS = (
    "phoneme",
    "syllable",
    "bpe_subword",
    "word",
    "acoustic_event",
    "vuv",
)
ENTROPY_SYSTEM = "tfc_style_entropy"
BASE_SYSTEMS = tuple(system for system in PAPER_SYSTEMS if system != ENTROPY_SYSTEM)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def rows_for_systems(rows, systems: set[str], rate: float) -> list[dict]:
    output = []
    for source in rows:
        row = dict(source)
        if str(row.get("system")) not in systems:
            continue
        row_rate = float(row.get("rate_hz", row.get("target_rate_hz", rate)))
        if abs(row_rate - rate) > 1e-6:
            continue
        row["rate_hz"] = rate
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--entropy-dir", type=Path, required=True)
    parser.add_argument("--target-rate-hz", type=float, required=True)
    parser.add_argument("--expected-utterances", type=int, default=1680)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    prediction_name = "boundary_predictions.jsonl"
    point_name = "point_metrics_40ms.csv"
    segment_name = "segment_metrics.csv"
    base_predictions = rows_for_systems(
        read_jsonl(args.base_dir / prediction_name), set(BASE_SYSTEMS), args.target_rate_hz
    )
    entropy_predictions = rows_for_systems(
        read_jsonl(args.entropy_dir / prediction_name), {ENTROPY_SYSTEM}, args.target_rate_hz
    )
    predictions = base_predictions + entropy_predictions

    base_points = rows_for_systems(
        read_csv(args.base_dir / point_name), set(BASE_SYSTEMS), args.target_rate_hz
    )
    entropy_points = rows_for_systems(
        read_csv(args.entropy_dir / point_name), {ENTROPY_SYSTEM}, args.target_rate_hz
    )
    points = base_points + entropy_points
    base_segments = rows_for_systems(
        read_csv(args.base_dir / segment_name), set(BASE_SYSTEMS), args.target_rate_hz
    )
    entropy_segments = rows_for_systems(
        read_csv(args.entropy_dir / segment_name), {ENTROPY_SYSTEM}, args.target_rate_hz
    )
    segments = base_segments + entropy_segments

    expected_counts = Counter(
        {system: args.expected_utterances for system in PAPER_SYSTEMS}
    )
    prediction_counts = Counter(str(row["system"]) for row in predictions)
    segment_counts = Counter(str(row["system"]) for row in segments)
    if prediction_counts != expected_counts or segment_counts != expected_counts:
        raise RuntimeError(
            f"seven-method coverage mismatch: predictions={prediction_counts}, "
            f"segments={segment_counts}"
        )
    point_keys = {
        (str(row["utt_id"]), str(row["system"]), str(row["target"]))
        for row in points
    }
    expected_point_rows = args.expected_utterances * len(PAPER_SYSTEMS) * len(TARGETS)
    if len(points) != expected_point_rows or len(point_keys) != expected_point_rows:
        raise RuntimeError(f"point-metric coverage mismatch: {len(points)}")
    if {str(row["target"]) for row in points} != set(TARGETS):
        raise RuntimeError("point-metric target coverage mismatch")

    order = {system: index for index, system in enumerate(PAPER_SYSTEMS)}
    predictions.sort(key=lambda row: (str(row["utt_id"]), order[str(row["system"])]))
    points.sort(
        key=lambda row: (
            str(row["utt_id"]),
            order[str(row["system"])],
            TARGETS.index(str(row["target"])),
        )
    )
    segments.sort(key=lambda row: (str(row["utt_id"]), order[str(row["system"])]))
    args.output_dir.mkdir(parents=True)
    manifest = args.output_dir / prediction_name
    atomic_text(
        manifest,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
    )
    atomic_csv(args.output_dir / point_name, points)
    atomic_csv(args.output_dir / segment_name, segments)
    acceptance = {
        "status": "passed",
        "protocol": "paper_seven_boundary_bundle_v1",
        "target_rate_hz": args.target_rate_hz,
        "systems": list(PAPER_SYSTEMS),
        "utterances_per_system": args.expected_utterances,
        "records": len(predictions),
        "point_rows": len(points),
        "segment_rows": len(segments),
        "manifest_sha256": sha256(manifest),
    }
    atomic_text(
        args.output_dir / "acceptance.json",
        json.dumps(acceptance, indent=2) + "\n",
    )
    print(json.dumps(acceptance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
