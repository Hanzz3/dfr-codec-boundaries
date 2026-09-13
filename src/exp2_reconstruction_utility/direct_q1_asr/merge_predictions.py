#!/usr/bin/env python3
"""Merge two TIMIT Direct q1-sem Qwen ASR shards."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
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


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-utterances", type=int, default=1680)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for root in args.shard_root:
        if (root / "eval.exit").read_text().strip() != "0":
            raise RuntimeError(f"incomplete shard: {root}")
        with (root / "predictions.jsonl").open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    keys = [(str(row["utt_id"]), str(row["system"])) for row in rows]
    expected_rows = args.expected_utterances * len(SYSTEMS)
    if len(rows) != expected_rows or len(set(keys)) != expected_rows:
        raise RuntimeError(f"prediction coverage mismatch: {len(rows)}")
    counts = Counter(system for _, system in keys)
    if counts != Counter({system: args.expected_utterances for system in SYSTEMS}):
        raise RuntimeError(f"system coverage mismatch: {counts}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["system"])].append(row)
    summary = []
    for system in SYSTEMS:
        subset = grouped[system]
        for row in subset:
            row["errors"] = sum(int(row[name]) for name in ("substitutions", "deletions", "insertions"))
            row["utterance_wer"] = row["errors"] / max(int(row["reference_words"]), 1)
        summary.append(
            {
                "system": "similarity" if system == "flexicodec_threshold" else system,
                "utterances": len(subset),
                "mean_utterance_wer": sum(row["utterance_wer"] for row in subset) / len(subset),
                "corpus_wer": sum(row["errors"] for row in subset)
                / sum(int(row["reference_words"]) for row in subset),
            }
        )
    rows.sort(key=lambda row: (str(row["utt_id"]), str(row["system"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(
        args.output_dir / "predictions.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    report = {
        "status": "passed",
        "protocol": "timit_current_nine_direct_q1_sem_full960h_qwen_v1",
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_selection": "LibriTTS_dev_only",
        "test_selection": "forbidden",
        "utterances_per_system": args.expected_utterances,
        "records": len(rows),
        "summary": summary,
    }
    atomic_text(args.output_dir / "summary.json", json.dumps(report, indent=2) + "\n")
    atomic_text(args.output_dir / "acceptance.exit", "0\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
