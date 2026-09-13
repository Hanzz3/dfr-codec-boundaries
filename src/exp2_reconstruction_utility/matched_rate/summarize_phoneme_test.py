#!/usr/bin/env python3
"""Validate and compare strict-budget reconstruction summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import read_jsonl


SYSTEMS = ("syllable_direct", "sv_uniform", "sv_similarity", "sv_ple")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--syllable-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--syllable-predictions", type=Path, required=True)
    args = parser.parse_args()

    summaries = [pd.read_csv(args.syllable_root / "reconstruction_summary.csv")]
    for system in SYSTEMS[1:]:
        summaries.append(
            pd.read_csv(args.run_root / "summaries" / system / "reconstruction_summary.csv")
        )
    frame = pd.concat(summaries, ignore_index=True)
    if set(frame["system"]) != set(SYSTEMS) or len(frame) != len(SYSTEMS):
        raise RuntimeError("comparison summaries must contain exactly four systems")
    if not (frame["utterances"] == 1680).all():
        raise RuntimeError("incomplete reconstruction summaries")

    strict_rows = read_jsonl(args.predictions)
    syllable_rows = read_jsonl(args.syllable_predictions)
    segment_counts = {
        system: sum(len(row["starts"]) for row in strict_rows if row["system"] == system)
        for system in SYSTEMS[1:]
    }
    segment_counts["syllable_direct"] = sum(len(row["starts"]) for row in syllable_rows)
    target = segment_counts["syllable_direct"]
    if set(segment_counts.values()) != {target}:
        raise RuntimeError(f"strict token budgets differ: {segment_counts}")

    frame["total_segments"] = frame["system"].map(segment_counts)
    reference = frame[frame["system"] == "syllable_direct"].iloc[0]
    directions = {
        "wer": -1.0,
        "pesq_wb": 1.0,
        "stoi": 1.0,
        "si_sdr": 1.0,
        "log_mel_distance": -1.0,
        "wavlm_speaker_similarity": 1.0,
    }
    for metric, direction in directions.items():
        frame[f"improvement_vs_syllable_{metric}"] = direction * (
            frame[metric] - float(reference[metric])
        )
    frame = frame.sort_values("wer").reset_index(drop=True)
    args.run_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.run_root / "strict_rate_comparison.csv", index=False)

    metric_columns = [
        "system",
        "total_segments",
        "realized_rate_hz",
        "wer",
        "pesq_wb",
        "stoi",
        "si_sdr",
        "log_mel_distance",
        "wavlm_speaker_similarity",
        "total_bps",
    ]
    report = [
        "# Strict Syllable-Rate Reconstruction Comparison",
        "",
        "All systems use exactly 22,875 segments on the same 1,680 TIMIT TEST utterances. "
        "Similarity and PLE parameters are calibrated from TEST features without GT labels solely "
        "to enforce the shared token budget; reconstruction metrics are not used for selection.",
        "",
        frame[metric_columns].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (args.run_root / "strict_rate_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    validation = {
        "status": "passed",
        "systems": list(SYSTEMS),
        "utterances_per_system": 1680,
        "total_segments_per_system": target,
        "finite_metrics": bool(
            np.isfinite(
                frame[
                    [
                        "wer",
                        "pesq_wb",
                        "stoi",
                        "si_sdr",
                        "log_mel_distance",
                        "wavlm_speaker_similarity",
                    ]
                ].to_numpy()
            ).all()
        ),
    }
    if not validation["finite_metrics"]:
        raise RuntimeError("comparison contains non-finite metrics")
    (args.run_root / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    (args.run_root / "acceptance.exit").write_text("0\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
