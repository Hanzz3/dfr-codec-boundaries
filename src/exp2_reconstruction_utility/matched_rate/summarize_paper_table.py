#!/usr/bin/env python3
"""Summarize the matched-rate phoneme/syllable corpus WER table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHODS = ("Uniform", "Reference-derived", "CodecSlime", "Similarity")


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected METHOD=PATH")
    method, path = value.split("=", 1)
    if method not in METHODS:
        raise argparse.ArgumentTypeError(f"method must be one of {METHODS}")
    return method, Path(path)


def paths_by_method(values: list[tuple[str, Path]], name: str) -> dict[str, Path]:
    output = dict(values)
    if len(output) != len(values) or set(output) != set(METHODS):
        raise RuntimeError(f"{name} inputs must contain each method exactly once")
    return output


def corpus_wer(path: Path, decoder_mode: str, expected: int) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if not row.get("decoder_mode") or row["decoder_mode"] == decoder_mode
        ]
    utterances = {str(row["utt_id"]) for row in rows}
    if len(rows) != expected or len(utterances) != expected:
        raise RuntimeError(f"unexpected coverage in {path}: {len(rows)} rows")
    errors = sum(int(float(row["wer_errors"])) for row in rows)
    words = sum(int(float(row["wer_reference_words"])) for row in rows)
    if words <= 0:
        raise RuntimeError(f"empty references in {path}")
    return errors / words


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phoneme", type=named_path, action="append", required=True)
    parser.add_argument("--syllable", type=named_path, action="append", required=True)
    parser.add_argument("--decoder-mode", default="tuned")
    parser.add_argument("--expected-utterances", type=int, default=1680)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phoneme = paths_by_method(args.phoneme, "phoneme")
    syllable = paths_by_method(args.syllable, "syllable")
    rows = [
        {
            "method": method,
            "phoneme_rate_corpus_wer_percent": 100 * corpus_wer(
                phoneme[method], args.decoder_mode, args.expected_utterances
            ),
            "syllable_rate_corpus_wer_percent": 100 * corpus_wer(
                syllable[method], args.decoder_mode, args.expected_utterances
            ),
        }
        for method in METHODS
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
