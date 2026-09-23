#!/usr/bin/env python3
"""Materialize Entropy-Mass boundaries and alignment metrics on TIMIT TEST."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from boundary_metrics import match_boundaries
from entropy_mass import IMPLEMENTATION, SYSTEM, entropy_mass_starts


TARGETS = (
    "phoneme",
    "syllable",
    "bpe_subword",
    "word",
    "acoustic_event",
    "vuv",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def boundary_times(starts: list[int], times: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (float(times[start - 1]) + float(times[start])) / 2.0
            for start in starts[1:]
        ],
        dtype=np.float32,
    )


def feature_nmse(features: np.ndarray, starts: list[int]) -> float:
    reconstructed = np.empty_like(features)
    for start, end in zip(starts, [*starts[1:], len(features)]):
        reconstructed[start:end] = features[start:end].mean(axis=0)
    denominator = float(np.mean(features * features))
    return float(
        np.mean((reconstructed - features) ** 2) / max(denominator, 1e-8)
    )


def rate_tag(rate: float) -> str:
    return f"{rate:g}".replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument(
        "--reference-tracks", "--ground-truth", dest="reference_tracks",
        type=Path, required=True,
    )
    parser.add_argument(
        "--target-rate-hz", type=float, action="append", required=True,
        dest="target_rates_hz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-utterances", type=int, default=1680)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--collar-ms", type=float, default=40.0)
    parser.add_argument("--entropy-bins", type=int, default=64)
    parser.add_argument("--entropy-sigma", type=float)
    args = parser.parse_args()

    rates = tuple(sorted(set(float(rate) for rate in args.target_rates_hz)))
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("target rates must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    features = {
        str(row["utt_id"]): row
        for row in read_jsonl(args.feature_index)
        if row.get("split") == "test"
    }
    references = {
        str(row["utt_id"]): row for row in read_jsonl(args.reference_tracks)
    }
    if set(features) - set(references):
        raise RuntimeError("feature/reference coverage mismatch")
    utterance_ids = sorted(features)
    if args.max_utterances is not None:
        utterance_ids = utterance_ids[: args.max_utterances]
    expected = args.max_utterances or args.expected_utterances
    if len(utterance_ids) != expected:
        raise RuntimeError(f"utterance count {len(utterance_ids)} != {expected}")

    predictions: dict[float, list[dict]] = defaultdict(list)
    points: dict[float, list[dict]] = defaultdict(list)
    segments: dict[float, list[dict]] = defaultdict(list)
    tolerance = args.collar_ms / 1000.0

    for completed, utt_id in enumerate(utterance_ids, 1):
        record = features[utt_id]
        with np.load(record["feature_path"]) as payload:
            matrix = payload["features"].astype(np.float32)
            times = payload["times"].astype(np.float32)
        waveform, sample_rate = sf.read(
            record["audio"], dtype="float32", always_2d=False
        )
        duration = float(record["duration_sec"])
        native_rate = len(matrix) / duration
        max_span = max(1, int(math.floor(native_rate * 0.640 + 1e-12)))

        for rate in rates:
            starts, entropy = entropy_mass_starts(
                waveform,
                sample_rate,
                times,
                duration,
                rate,
                bins=args.entropy_bins,
                sigma=args.entropy_sigma,
            )
            ends = [*starts[1:], len(matrix)]
            lengths = [right - left for left, right in zip(starts, ends)]
            values = boundary_times(starts, times)
            realized_rate = len(starts) / duration
            predictions[rate].append(
                {
                    "utt_id": utt_id,
                    "system": SYSTEM,
                    "rate_hz": rate,
                    "starts": starts,
                    "boundary_times": values.tolist(),
                    "segment_lengths": lengths,
                    "frames": len(matrix),
                    "duration_sec": duration,
                    "feature_path": record["feature_path"],
                    "audio": record["audio"],
                    "realized_rate_hz": realized_rate,
                    "implementation": IMPLEMENTATION,
                    "control_mode": "exact_utterance_budget",
                    "requested_segments": max(1, int(round(duration * rate))),
                    "maximum_native_span_frames": max_span,
                    "entropy_bins": args.entropy_bins,
                    "entropy_sigma": (
                        args.entropy_sigma
                        if args.entropy_sigma is not None
                        else 2.0 / (args.entropy_bins - 1)
                    ),
                    "entropy_mean": float(entropy.mean()),
                }
            )
            segments[rate].append(
                {
                    "utt_id": utt_id,
                    "system": SYSTEM,
                    "rate_hz": rate,
                    "segments": len(starts),
                    "realized_rate_hz": realized_rate,
                    "feature_nmse": feature_nmse(matrix, starts),
                    "max_segment_frames": max(lengths),
                }
            )
            for target in TARGETS:
                tp, fp, fn, _ = match_boundaries(
                    values,
                    references[utt_id]["boundaries"][target],
                    tolerance,
                )
                points[rate].append(
                    {
                        "utt_id": utt_id,
                        "system": SYSTEM,
                        "target": target,
                        "rate_hz": rate,
                        "collar_ms": args.collar_ms,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }
                )
        if completed % 100 == 0 or completed == expected:
            print(f"processed {completed}/{expected}", flush=True)

    for rate in rates:
        output = args.output_dir / f"rate_{rate_tag(rate)}"
        atomic_text(
            output / "boundary_predictions.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions[rate]),
        )
        atomic_csv(output / "point_metrics_40ms.csv", points[rate])
        atomic_csv(output / "segment_metrics.csv", segments[rate])
        atomic_text(
            output / "acceptance.json",
            json.dumps(
                {
                    "status": "passed",
                    "protocol": "entropy_mass_crossrate_v1",
                    "system": SYSTEM,
                    "target_rate_hz": rate,
                    "utterances": expected,
                    "collar_ms": args.collar_ms,
                },
                indent=2,
            )
            + "\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
