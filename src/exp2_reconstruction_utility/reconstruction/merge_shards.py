#!/usr/bin/env python3
"""Merge, validate, and summarize sharded tuned-decoder evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from metrics import SYSTEMS, read_jsonl


METRICS = ("wer", "feature_nmse", "pesq_wb", "stoi", "speaker_similarity")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-utterances", type=int, default=1680)
    parser.add_argument("--expected-step", type=int, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.shards:
        marker = path.parent / "eval.exit"
        if not marker.is_file() or marker.read_text(encoding="ascii").strip() != "0":
            raise RuntimeError(f"shard is not accepted: {path}")
        rows.extend(read_jsonl(path))
    keys = [(str(row["utt_id"]), str(row["system"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate utterance/system rows")
    counts = Counter(system for _, system in keys)
    expected_counts = {system: args.expected_utterances for system in SYSTEMS}
    if dict(counts) != expected_counts:
        raise RuntimeError(f"unexpected per-system counts: {counts}")
    utterances = sorted({utt_id for utt_id, _ in keys})
    expected_keys = {(utt_id, system) for utt_id in utterances for system in SYSTEMS}
    if len(utterances) != args.expected_utterances or set(keys) != expected_keys:
        raise RuntimeError("incomplete utterance/system Cartesian product")
    if {int(row["decoder_step"]) for row in rows} != {args.expected_step}:
        raise RuntimeError("decoder checkpoint step mismatch")
    for row in rows:
        for metric in ("wer", "feature_nmse", "stoi", "speaker_similarity"):
            if not math.isfinite(float(row[metric])):
                raise RuntimeError(f"non-finite {metric}: {row['utt_id']}/{row['system']}")
        if int(row["reference_words"]) <= 0:
            raise RuntimeError("reference_words must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = args.output_dir / "reconstruction.jsonl"
    temporary = merged.with_name(merged.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: (item["utt_id"], SYSTEMS.index(item["system"]))):
            handle.write(json.dumps(row, allow_nan=True, sort_keys=True) + "\n")
    temporary.replace(merged)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["system"]].append(row)
    summaries: list[dict] = []
    for system in SYSTEMS:
        subset = grouped[system]
        pesq = [float(row["pesq_wb"]) for row in subset if math.isfinite(float(row["pesq_wb"]))]
        summaries.append(
            {
                "system": system,
                "utterances": len(subset),
                "mean_utterance_wer": sum(float(row["wer"]) for row in subset) / len(subset),
                "micro_wer": sum(int(row["word_errors"]) for row in subset)
                / sum(int(row["reference_words"]) for row in subset),
                "feature_nmse": sum(float(row["feature_nmse"]) for row in subset) / len(subset),
                "pesq_wb": sum(pesq) / len(pesq) if pesq else math.nan,
                "pesq_valid_fraction": len(pesq) / len(subset),
                "stoi": sum(float(row["stoi"]) for row in subset) / len(subset),
                "speaker_similarity": sum(float(row["speaker_similarity"]) for row in subset) / len(subset),
                "word_errors": sum(int(row["word_errors"]) for row in subset),
                "reference_words": sum(int(row["reference_words"]) for row in subset),
            }
        )
    summary_path = args.output_dir / "summary.csv"
    summary_tmp = summary_path.with_name(summary_path.name + f".tmp.{os.getpid()}")
    with summary_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    summary_tmp.replace(summary_path)

    minimum_pesq = min(row["pesq_valid_fraction"] for row in summaries)
    acceptance = {
        "status": "passed" if minimum_pesq >= 0.95 else "failed",
        "protocol": "current_nine_tuned_decoder_five_metrics_timit_v1",
        "decoder_step": args.expected_step,
        "systems": list(SYSTEMS),
        "utterances": len(utterances),
        "records": len(rows),
        "rows_per_system": expected_counts,
        "metrics": list(METRICS),
        "minimum_pesq_valid_fraction": minimum_pesq,
    }
    acceptance_path = args.output_dir / "acceptance.json"
    acceptance_tmp = acceptance_path.with_name(acceptance_path.name + f".tmp.{os.getpid()}")
    acceptance_tmp.write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")
    acceptance_tmp.replace(acceptance_path)
    exit_code = 0 if acceptance["status"] == "passed" else 1
    (args.output_dir / "acceptance.exit").write_text(f"{exit_code}\n", encoding="ascii")
    print(json.dumps(acceptance, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
