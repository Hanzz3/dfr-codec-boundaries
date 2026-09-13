#!/usr/bin/env python3
"""Merge and validate frozen q1/q8 reconstruction JSONL shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path


SYSTEMS = (
    "uniform",
    "flexicodec_threshold",
    "codecslime_dp",
    "ple",
    "elastic_time_greedy",
    "elastic_time_dp",
    "dcdit_1d",
    "atome_style",
    "tadpc_style",
)


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


def words(value: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", value.lower()).split()


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for row, lhs in enumerate(left, 1):
        current = [row]
        for column, rhs in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (lhs != rhs),
                )
            )
        previous = current
    return previous[-1]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("q1", "q8"), required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--segment-metrics", type=Path, required=True)
    parser.add_argument("--target-rate-hz", type=float, required=True)
    parser.add_argument("--expected-utterances", type=int, default=1680)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    segment_nmse: dict[tuple[str, str], float] = {}
    with args.segment_metrics.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            segment_nmse[(row["utt_id"], row["system"])] = float(row["feature_nmse"])

    rows = [dict(row) for path in args.shard for row in read_jsonl(path)]
    expected_rows = args.expected_utterances * len(SYSTEMS)
    keys = [(str(row["utt_id"]), str(row["system"])) for row in rows]
    if len(rows) != expected_rows or len(set(keys)) != expected_rows:
        raise RuntimeError(f"{args.arm} row/key coverage mismatch: {len(rows)}")
    expected_counts = {system: args.expected_utterances for system in SYSTEMS}
    if dict(Counter(system for _, system in keys)) != expected_counts:
        raise RuntimeError(f"{args.arm} system coverage mismatch")
    if set(keys) != set(segment_nmse):
        raise RuntimeError(f"{args.arm} reconstruction/segment keys differ")

    for row in rows:
        key = (str(row["utt_id"]), str(row["system"]))
        reference = words(str(row.get("reference_text", "")))
        hypothesis = words(str(row.get("hypothesis", "")))
        row["word_errors"] = int(float(row.get("word_errors", edit_distance(reference, hypothesis))))
        row["reference_words"] = int(float(row.get("reference_words", len(reference))))
        row["wer"] = float(row["wer"])
        row["feature_nmse"] = segment_nmse[key]
        row["target_rate_hz"] = args.target_rate_hz
        required = [row["wer"], row["feature_nmse"]]
        if args.arm == "q8":
            required.extend(float(row[column]) for column in ("pesq_wb", "speaker_similarity"))
        if not all(math.isfinite(float(value)) for value in required):
            raise RuntimeError(f"non-finite {args.arm} metric: {key}")
        if row["reference_words"] <= 0 or row.get("decoded_waveform_finite") is not True:
            raise RuntimeError(f"invalid {args.arm} waveform/word count: {key}")

    rows.sort(key=lambda row: (str(row["utt_id"]), str(row["system"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"reconstruction_{args.arm}.csv"
    fields = sorted({key for row in rows for key in row})
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    summary = []
    for system in SYSTEMS:
        subset = [row for row in rows if row["system"] == system]
        summary.append(
            {
                "system": system,
                "mean_utterance_wer": sum(row["wer"] for row in subset) / len(subset),
                "corpus_wer": sum(row["word_errors"] for row in subset)
                / sum(row["reference_words"] for row in subset),
                "feature_nmse": sum(row["feature_nmse"] for row in subset) / len(subset),
                "pesq_wb": (
                    sum(float(row["pesq_wb"]) for row in subset) / len(subset)
                    if args.arm == "q8"
                    else None
                ),
                "speaker_similarity": (
                    sum(float(row["speaker_similarity"]) for row in subset) / len(subset)
                    if args.arm == "q8"
                    else None
                ),
            }
        )
    atomic_text(args.output_dir / f"summary_{args.arm}.json", json.dumps(summary, indent=2) + "\n")
    acceptance = {
        "status": "passed",
        "protocol": f"timit_codec_grid_crossrate_frozen_flexicodec_{args.arm}_v2",
        "arm": args.arm,
        "num_quantizers": 1 if args.arm == "q1" else 8,
        "target_rate_hz": args.target_rate_hz,
        "systems": list(SYSTEMS),
        "utterances_per_system": args.expected_utterances,
        "records": len(rows),
        "shards": [{"path": str(path), "sha256": sha256(path)} for path in args.shard],
        "output_sha256": sha256(output),
    }
    atomic_text(args.output_dir / f"acceptance_{args.arm}.json", json.dumps(acceptance, indent=2) + "\n")
    atomic_text(args.output_dir / f"acceptance_{args.arm}.exit", "0\n")
    print(json.dumps(acceptance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
