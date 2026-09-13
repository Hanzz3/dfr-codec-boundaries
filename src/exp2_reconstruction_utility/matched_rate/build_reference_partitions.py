#!/usr/bin/env python3
"""Build a direct-boundary prediction manifest from one EXP1 reference track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from direct_boundaries import select_direct_boundaries
from io_utils import read_jsonl, write_jsonl


def boundaries_from_starts(starts: np.ndarray, times: np.ndarray) -> list[float]:
    return [
        (float(times[index - 1]) + float(times[index])) / 2.0
        for index in starts[1:]
    ]


def build_prediction(
    record: dict,
    ground_truth: dict,
    *,
    target: str | None = None,
    targets: tuple[str, ...] | None = None,
    system: str,
    max_span: int,
) -> dict:
    with np.load(record["feature_path"]) as payload:
        features = payload["features"].astype(np.float32)
        times = payload["times"].astype(np.float32)
    targets = targets or (target or "syllable",)
    sources = {}
    for target in targets:
        source = ground_truth.get("boundaries", {}).get(target)
        if source is None:
            raise KeyError(f"missing {target} boundaries for {record['utt_id']}")
        sources[target] = [float(value) for value in source]
    source = sorted({round(value, 9) for values in sources.values() for value in values})
    result = select_direct_boundaries(system, features, times, source, max_span)
    duration = float(record["duration_sec"])
    return {
        "utt_id": record["utt_id"],
        "audio": record["audio"],
        "transcript": record.get("transcript", ""),
        "split": record["split"],
        "duration_sec": duration,
        "system": system,
        "condition": "natural",
        "target_rate_hz": None,
        "realized_rate_hz": len(result.starts) / duration,
        "starts": result.starts.tolist(),
        "boundary_times": boundaries_from_starts(result.starts, times),
        "segment_lengths": result.segment_lengths.tolist(),
        "boundary_scores": [-1.0] * len(features),
        "feature_source": "sensevoice",
        "boundary_source": "exp1_" + "+".join(targets),
        "implementation": result.metadata["implementation"],
        "source_boundaries": len(source),
        "source_boundaries_before_union_dedup": sum(len(values) for values in sources.values()),
        "source_boundary_counts": {target: len(values) for target, values in sources.items()},
        "source_boundaries_after_collision": result.metadata[
            "source_boundaries_after_collision"
        ],
        "technical_maxspan_boundaries": result.metadata[
            "technical_maxspan_boundaries"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--reference-tracks", "--ground-truth", dest="reference_tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="syllable")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--system", default="syllable_direct")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-span", type=int, default=8)
    parser.add_argument("--expected-utterances", type=int)
    args = parser.parse_args()

    targets = tuple(args.targets or [args.target])
    if len(targets) != len(set(targets)):
        raise ValueError("targets must be unique")

    if args.max_span < 1:
        raise ValueError("max-span must be positive")
    gt = {row["utt_id"]: row for row in read_jsonl(args.reference_tracks)}
    records = [
        row for row in read_jsonl(args.feature_index) if row["split"] == args.split
    ]
    ids = [row["utt_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate utterances in feature index")
    if args.expected_utterances is not None and len(records) != args.expected_utterances:
        raise RuntimeError(
            f"expected {args.expected_utterances} utterances, found {len(records)}"
        )
    missing = sorted(set(ids) - set(gt))
    if missing:
        raise RuntimeError(f"missing ground truth: {missing[:10]}")

    rows = [
        build_prediction(
            record,
            gt[record["utt_id"]],
            targets=targets,
            system=args.system,
            max_span=args.max_span,
        )
        for record in records
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    validation = {
        "status": "passed",
        "system": args.system,
        "target": "+".join(targets),
        "targets": list(targets),
        "split": args.split,
        "utterances": len(rows),
        "mean_realized_rate_hz": float(
            np.mean([row["realized_rate_hz"] for row in rows])
        ),
        "source_boundaries": int(sum(row["source_boundaries"] for row in rows)),
        "source_boundaries_before_union_dedup": int(
            sum(row["source_boundaries_before_union_dedup"] for row in rows)
        ),
        "source_boundaries_after_collision": int(
            sum(row["source_boundaries_after_collision"] for row in rows)
        ),
        "technical_maxspan_boundaries": int(
            sum(row["technical_maxspan_boundaries"] for row in rows)
        ),
    }
    args.output.with_suffix(".validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
