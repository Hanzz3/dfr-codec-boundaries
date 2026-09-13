#!/usr/bin/env python3
"""Validate and rank validation-controlled 4.47 Hz reconstruction results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEMS = [
    "sv_uniform",
    "sv_similarity",
    "sv_codecslime",
    "sv_ple",
    "sv_peak",
    "sv_elastic_greedy",
    "sv_elastic_dp",
    "sv_dcdit",
    "sv_kmeans",
]
METRICS = {
    "wer": True,
    "pesq_wb": False,
    "stoi": False,
    "si_sdr": False,
    "log_mel_distance": True,
    "wavlm_speaker_similarity": False,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--syllable-summary", type=Path, required=True)
    parser.add_argument("--expected-utterances", type=int, default=1680)
    parser.add_argument("--target-rate", type=float, default=4.4729)
    args = parser.parse_args()

    frames = [pd.read_csv(args.syllable_summary)]
    for system in SYSTEMS:
        path = args.run_root / "summaries" / system / "reconstruction_summary.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if frame["system"].tolist() != [system]:
            raise RuntimeError(f"unexpected summary system in {path}")
        frames.append(frame)
    comparison = pd.concat(frames, ignore_index=True)
    expected = {"syllable_direct", *SYSTEMS}
    if set(comparison["system"]) != expected or len(comparison) != len(expected):
        raise RuntimeError("system coverage mismatch")
    if not (comparison["utterances"] == args.expected_utterances).all():
        raise RuntimeError("utterance coverage mismatch")
    if not (comparison["rows"] == args.expected_utterances).all():
        raise RuntimeError("row coverage mismatch")
    finite_columns = [*METRICS, "realized_rate_hz", "total_bps", "total_rtf"]
    if not np.isfinite(comparison[finite_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("non-finite reconstruction metric")

    comparison["rate_error_vs_syllable_hz"] = (
        comparison["realized_rate_hz"]
        - float(comparison.loc[comparison["system"] == "syllable_direct", "realized_rate_hz"].iloc[0])
    )
    for metric, lower_is_better in METRICS.items():
        comparison[f"{metric}_rank"] = comparison[metric].rank(
            method="min", ascending=lower_is_better
        ).astype(int)
    comparison = comparison.sort_values(["wer", "system"]).reset_index(drop=True)
    comparison.to_csv(args.run_root / "controlled_rate_comparison.csv", index=False)

    syllable = comparison.loc[comparison["system"] == "syllable_direct"].iloc[0]
    validation = {
        "status": "passed",
        "protocol": "timit_train_speaker_heldout_validation_controlled_rate",
        "test_labels_used_for_rate_control": False,
        "target_rate_hz": args.target_rate,
        "systems": sorted(expected),
        "utterances_per_system": args.expected_utterances,
        "finite_metrics": True,
        "syllable_ranks": {
            metric: int(syllable[f"{metric}_rank"]) for metric in METRICS
        },
    }
    (args.run_root / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    display = [
        "system", "realized_rate_hz", "wer", "pesq_wb", "stoi", "si_sdr",
        "log_mel_distance", "wavlm_speaker_similarity", "total_bps",
    ]
    report = [
        "# Validation-Controlled Syllable-Rate Comparison",
        "",
        "All selector controls are fitted on speaker-held-out TIMIT TRAIN validation and frozen on TEST. No TEST labels or reconstruction metrics are used for rate control.",
        "",
        comparison[display].to_markdown(index=False),
        "",
        "## Syllable ranks",
        "",
    ]
    for metric in METRICS:
        report.append(f"- `{metric}`: {int(syllable[f'{metric}_rank'])}/{len(expected)}")
    (args.run_root / "controlled_rate_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (args.run_root / "acceptance.exit").write_text("0\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
