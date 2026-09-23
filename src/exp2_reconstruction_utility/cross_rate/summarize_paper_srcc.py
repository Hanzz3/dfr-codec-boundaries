#!/usr/bin/env python3
"""Compute Table 3 absolute Spearman correlations from accepted outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

from paper_protocol import PAPER_CORRELATION_RATES_HZ, PAPER_SYSTEMS


TRACKS = ("syllable", "bpe_subword", "word")


def rate_path(value: str) -> tuple[float, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected RATE=PATH")
    rate, path = value.split("=", 1)
    return float(rate), Path(path)


def keyed_specs(values: list[tuple[float, Path]], name: str) -> dict[float, Path]:
    output = dict(values)
    if len(output) != len(values):
        raise RuntimeError(f"duplicate {name} rate")
    expected = set(PAPER_CORRELATION_RATES_HZ)
    if set(output) != expected:
        raise RuntimeError(f"{name} rates {set(output)} != {expected}")
    return output


def alignment_f1(path: Path, rate: float) -> dict[tuple[str, str], float]:
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            system = str(row["system"])
            target = str(row["target"])
            row_rate = float(row.get("rate_hz", rate))
            if system not in PAPER_SYSTEMS or target not in TRACKS:
                continue
            if abs(row_rate - rate) > 1e-6:
                continue
            values = counts[(system, target)]
            values[0] += int(float(row["tp"]))
            values[1] += int(float(row["fp"]))
            values[2] += int(float(row["fn"]))
    expected = {(system, target) for system in PAPER_SYSTEMS for target in TRACKS}
    if set(counts) != expected:
        raise RuntimeError(f"incomplete alignment rows at {rate} Hz")
    return {
        key: 2 * tp / max(2 * tp + fp + fn, 1)
        for key, (tp, fp, fn) in counts.items()
    }


def summaries(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "summary" in payload:
        payload = payload["summary"]
    rows = {str(row["system"]): row for row in payload}
    if set(rows) != set(PAPER_SYSTEMS):
        raise RuntimeError(f"summary systems differ in {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-metrics", type=rate_path, action="append", required=True)
    parser.add_argument("--q1-summary", type=rate_path, action="append", required=True)
    parser.add_argument("--q8-summary", type=rate_path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = keyed_specs(args.point_metrics, "point-metric")
    q1_paths = keyed_specs(args.q1_summary, "q1 summary")
    q8_paths = keyed_specs(args.q8_summary, "q1:8 summary")

    rows = []
    for rate in PAPER_CORRELATION_RATES_HZ:
        f1 = alignment_f1(points[rate], rate)
        q1 = summaries(q1_paths[rate])
        q8 = summaries(q8_paths[rate])
        utility = {
            "q1_recon_wer": [-float(q1[system]["corpus_wer"]) for system in PAPER_SYSTEMS],
            "nmse": [-float(q8[system]["feature_nmse"]) for system in PAPER_SYSTEMS],
            "spksim": [float(q8[system]["speaker_similarity"]) for system in PAPER_SYSTEMS],
        }
        for metric, values in utility.items():
            row = {"rate_hz": rate, "metric": metric}
            for track in TRACKS:
                alignment = [f1[(system, track)] for system in PAPER_SYSTEMS]
                rho = float(spearmanr(alignment, values).statistic)
                row[track] = abs(rho)
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
