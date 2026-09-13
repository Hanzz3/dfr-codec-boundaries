#!/usr/bin/env python3
"""EXP1: build the six derived reference tracks used for boundary evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import split_utils as split
import reference_tracks as core


def write_report(path: Path, summary: dict, args: argparse.Namespace) -> None:
    lines = [
        "# EXP1 Reference Tracks",
        "",
        f"- Manifest: `{args.manifest}`",
        f"- MFA dir: `{args.mfa_dir}`",
        f"- Output root: `{args.out_root}`",
        f"- Analyzed utterances: `{summary['analysis_rows']}`",
        f"- Total duration sec: `{summary['total_duration_sec']}`",
        f"- Phone sources: `{summary['phone_sources']}`",
        f"- Word sources: `{summary['word_sources']}`",
        "",
        "## Boundary Counts",
        "",
    ]
    rows = []
    for level in ("phoneme", "syllable", "bpe_subword", "word", "acoustic_event", "vuv"):
        rows.append(
            {
                "level": level,
                "boundaries": summary[f"total_{level}_boundaries"],
                "rate_hz": summary[f"{level}_boundary_rate_hz"],
            }
        )
    lines.append(core.markdown_table(pd.DataFrame(rows), index=False))
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{args.out_root / 'reference_tracks_by_utterance.jsonl'}`",
            f"- `{args.out_root / 'reference_boundaries_long.jsonl'}`",
            f"- `{args.out_root / 'utterance_stats.csv'}`",
            f"- `{args.out_root / 'summary.json'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mfa-dir", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--max-utterances", type=int, default=200)
    parser.add_argument("--bpe-tokenizer", default="gpt2")
    args = parser.parse_args()

    rows_all = core.read_jsonl(args.manifest)
    rows = [row for row in rows_all if row.get("audio") and (row.get("phone") or row.get("timit_phn"))]
    rows = rows[: args.max_utterances]
    if not rows:
        raise SystemExit(f"No usable rows found in {args.manifest}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    bpe_tokenizer, bpe_source = core.load_bpe_tokenizer(args.bpe_tokenizer)

    reference_items: list[dict] = []
    reference_records: list[dict] = []
    utt_stats: list[dict] = []

    for idx, row in enumerate(rows, 1):
        y, sr = core.load_audio(row["audio"])
        reference = core.build_ground_truth(row, y, sr, args.mfa_dir, bpe_tokenizer, bpe_source)
        duration = float(reference.stats["duration_sec"])
        reference_items.append(
            {
                "utt_id": row["utt_id"],
                "audio": row["audio"],
                "duration_sec": duration,
                "boundaries": split.boundary_set_to_dict(reference.boundaries),
                "stats": reference.stats,
                "manifest_row": row,
            }
        )
        reference_records.extend(reference.records)
        utt_stats.append(reference.stats)
        if idx % 50 == 0:
            print(f"exp1: processed {idx}/{len(rows)}", flush=True)

    with (args.out_root / "reference_tracks_by_utterance.jsonl").open("w", encoding="utf-8") as handle:
        for item in reference_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (args.out_root / "reference_boundaries_long.jsonl").open("w", encoding="utf-8") as handle:
        for record in reference_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    pd.DataFrame(utt_stats).to_csv(args.out_root / "utterance_stats.csv", index=False)

    summary = core.summarize_ground_truth(utt_stats, len(rows_all), bpe_source)
    core.write_json(args.out_root / "summary.json", summary)
    core.write_json(
        args.out_root / "config.json",
        {
            "manifest": str(args.manifest),
            "mfa_dir": str(args.mfa_dir) if args.mfa_dir else None,
            "out_root": str(args.out_root),
            "max_utterances": args.max_utterances,
            "bpe_tokenizer": args.bpe_tokenizer,
            "bpe_source": bpe_source,
            "bpe_boundary_definition": "all_bpe_unit_starts_after_time_zero_including_word_starts",
            "bpe_time_mapping": "tokenizer_character_offsets_linearly_interpolated_within_mfa_word_intervals",
            "bpe_phone_snapping": False,
        },
    )
    write_report(args.out_root / "exp1_report.md", summary, args)
    print(json.dumps({"exp": 1, "rows": len(rows), "out_root": str(args.out_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
