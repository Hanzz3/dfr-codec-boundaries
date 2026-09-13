#!/usr/bin/env python3
"""Compare Exp1 MFA boundaries with the original TIMIT time labels."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import reference_tracks as core


DEFAULT_TOLERANCES_MS = (20.0, 40.0, 50.0)


def timit_boundaries(row: dict, target: str) -> list[float]:
    if target == "phoneme":
        intervals = core.read_timit_intervals(row.get("phone") or row.get("timit_phn"), "timit")
        return core.interval_boundaries(intervals, include_silence=True)[0]
    intervals = core.read_timit_intervals(row.get("word") or row.get("timit_wrd"), "timit")
    return core.interval_boundaries(intervals, include_silence=False)[0]


def aggregate(rows: list[dict]) -> pd.DataFrame:
    grouped: dict[tuple[str, float], dict] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "mfa": 0, "timit": 0, "offsets": [], "utterances": 0}
    )
    for row in rows:
        accumulator = grouped[(row["target"], row["tolerance_ms"])]
        for name in ("tp", "fp", "fn", "mfa", "timit"):
            accumulator[name] += row[name]
        accumulator["offsets"].extend(row["offsets_ms"])
        accumulator["utterances"] += 1

    output = []
    for (target, tolerance_ms), values in sorted(grouped.items()):
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        offsets = values["offsets"]
        output.append(
            {
                "target": target,
                "tolerance_ms": tolerance_ms,
                "utterances": values["utterances"],
                "mfa_boundaries": values["mfa"],
                "timit_boundaries": values["timit"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_offset_ms": statistics.mean(offsets) if offsets else math.nan,
                "mean_abs_offset_ms": statistics.mean(abs(value) for value in offsets) if offsets else math.nan,
                "matching_protocol": core.BOUNDARY_MATCHING_PROTOCOL,
            }
        )
    return pd.DataFrame(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-tracks", "--ground-truth", dest="reference_tracks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tolerances-ms",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOLERANCES_MS),
        help="Boundary matching tolerances in milliseconds.",
    )
    args = parser.parse_args()

    items = core.read_jsonl(args.reference_tracks)
    rows = []
    for item in items:
        manifest = item.get("manifest_row") or {}
        for target in ("phoneme", "word"):
            mfa = item["boundaries"][target]
            timit = timit_boundaries(manifest, target)
            for tolerance_ms in args.tolerances_ms:
                tp, fp, fn, offsets = core.match_boundaries(
                    np.asarray(mfa, dtype=np.float32),
                    timit,
                    tolerance_ms / 1000.0,
                )
                rows.append(
                    {
                        "utt_id": item["utt_id"],
                        "target": target,
                        "tolerance_ms": tolerance_ms,
                        "mfa": len(mfa),
                        "timit": len(timit),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "offsets_ms": [offset * 1000.0 for offset in offsets],
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_utterance = pd.DataFrame(rows)
    serializable = per_utterance.copy()
    serializable["offsets_ms"] = serializable["offsets_ms"].map(json.dumps)
    serializable.to_csv(args.output_dir / "mfa_timit_per_utterance.csv", index=False)
    summary = aggregate(rows)
    summary.to_csv(args.output_dir / "mfa_timit_agreement.csv", index=False)
    report = [
        "# MFA 3.0 vs. TIMIT Boundary Validation",
        "",
        f"- Utterances: `{len(items)}`",
        f"- Matching: `{core.BOUNDARY_MATCHING_PROTOCOL}`",
        f"- Collars: `{', '.join(f'{value:g}' for value in args.tolerances_ms)} ms`",
        "- MFA boundaries are predictions; original TIMIT `.PHN/.WRD` boundaries are references.",
        "",
        core.markdown_table(summary, index=False),
    ]
    (args.output_dir / "mfa_timit_validation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
