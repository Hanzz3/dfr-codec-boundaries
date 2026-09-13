#!/usr/bin/env python3
"""Materialize nine selector boundaries for four rates on LibriTTS features."""
from __future__ import annotations

import argparse, json, math, os
from collections import Counter
from pathlib import Path

import numpy as np

from followup_selectors import physical_max_span, requested_segments, select_base, uniform_starts
from trained_selectors import load_trained_artifacts, normalize_features, select_trained
from common_selectors import cosine_change, threshold_starts
from supplement_selectors import atome_style_starts, tadpc_style_starts

SYSTEMS = ("uniform", "flexicodec_threshold", "codecslime_dp", "ple",
           "elastic_time_greedy", "elastic_time_dp", "dcdit_1d",
           "atome_style", "tadpc_style")
RATES = (3.0, 6.25, 8.33, 10.0)

def rows(path: Path):
    with path.open(encoding="utf-8") as h:
        for line in h:
            if line.strip():
                yield json.loads(line)

def load_feature(row):
    with np.load(row["feature_path"]) as p:
        key = str(row.get("feature_key", "features"))
        x = p[key].astype(np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise RuntimeError(f"invalid feature {row['utt_id']}")
    return x

def atome_ratio(items, target):
    total = sum(d for _, d in items)
    def rate(r):
        return sum(n - min(int(math.floor(n*r)), n//2) for n, _ in items) / total
    grid = np.linspace(0.0, 0.5, 5001)
    best = min((float(v) for v in grid), key=lambda v: (abs(rate(v)-target), v))
    return best, rate(best)

def tadpc_threshold(items, target, max_span=8):
    total = sum(d for _, d in items)
    history = []
    def evaluate(t):
        seg = 0
        for x, _ in items:
            seg += len(tadpc_style_starts(x, threshold=t, tadpc_max_span=max_span).starts)
        r = seg / total
        history.append((float(t), float(r)))
        return r
    lo, hi = -1.0, 1.0
    if not evaluate(lo) <= target <= evaluate(hi):
        raise RuntimeError(f"TADPC cannot bracket target {target}")
    for _ in range(24):
        mid = (lo + hi) / 2.0
        if evaluate(mid) < target: lo = mid
        else: hi = mid
    best = min(history, key=lambda z: (abs(z[1]-target), z[0]))
    return best[0], best[1], history

def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-index", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=("train", "val"))
    ap.add_argument("--source-split", required=True)
    ap.add_argument("--calibration-index", type=Path, required=True)
    ap.add_argument("--predictor-dir", type=Path, required=True)
    ap.add_argument("--normalization", type=Path, required=True)
    ap.add_argument("--kmeans-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-utterances", type=int)
    args = ap.parse_args()
    source = [r for r in rows(args.feature_index)
              if r.get("split") == args.split and r.get("source_split") == args.source_split]
    if not source: raise RuntimeError("empty source split")
    if args.max_utterances is not None:
        source = source[:args.max_utterances]
    calibration_rows = [r for r in rows(args.calibration_index) if r.get("split") == "val"]
    if not calibration_rows: raise RuntimeError("empty calibration split")
    artifacts = load_trained_artifacts(args.predictor_dir, args.normalization,
                                        args.kmeans_dir, "cpu", int(source[0]["feature_dim"]))
    controls = {}
    cal_items = []
    cal_lengths = []
    for r in calibration_rows:
        x = load_feature(r); d = float(r["duration_sec"])
        cal_items.append((x, d)); cal_lengths.append((len(x), d))
    for rate in RATES:
        # Reuse the accepted label-free calibration for base/trained mergers.
        # It is produced by calibrate_nine_system_rates.py and loaded below.
        controls[rate] = {}
    cal_path = args.output.parent / "_nine_rate_calibration.json"
    # This file is written by the launcher before materialization.
    if not cal_path.is_file():
        raise RuntimeError(f"missing calibration bundle {cal_path}")
    bundle = json.loads(cal_path.read_text())
    for rate in RATES:
        controls[rate] = bundle[str(rate)]["parameters"]
    dynamic = {}
    for rate in RATES:
        ratio, ar = atome_ratio(cal_lengths, rate)
        threshold, tr, history = tadpc_threshold(cal_items, rate, 8)
        dynamic[rate] = {"atome_ratio": ratio, "atome_rate_hz": ar,
                         "tadpc_threshold": threshold, "tadpc_rate_hz": tr,
                         "tadpc_history": history}
    cache_artifacts = artifacts
    out_rows = []
    counts = Counter(); rates = Counter(); durations = Counter()
    for index, row in enumerate(source, 1):
        x = load_feature(row); duration = float(row["duration_sec"])
        frames = len(x); max_span = physical_max_span(frames, duration)
        normalized = normalize_features(x, cache_artifacts)
        for rate in RATES:
            c = controls[rate]; trained_cache = {}
            nseg = requested_segments(duration, rate)
            selections = {
                "uniform": uniform_starts(frames, nseg, max_span),
                "flexicodec_threshold": threshold_starts(cosine_change(x), float(c["similarity"]["value"]), max_span),
                "codecslime_dp": select_base("codecslime_dp", x, duration, rate, max_span, c).starts,
                "ple": select_base("ple", x, duration, rate, max_span, c).starts,
                "elastic_time_greedy": select_trained("elastic_time_greedy", normalized, duration, rate, max_span, c, cache_artifacts, trained_cache).starts,
                "elastic_time_dp": select_trained("elastic_time_dp", normalized, duration, rate, max_span, c, cache_artifacts, trained_cache).starts,
                "dcdit_1d": select_trained("dcdit_1d", normalized, duration, rate, max_span, c, cache_artifacts, trained_cache).starts,
                "atome_style": atome_style_starts(x, (dynamic[rate]["atome_ratio"],)).starts,
                "tadpc_style": tadpc_style_starts(x, threshold=dynamic[rate]["tadpc_threshold"], tadpc_max_span=8).starts,
            }
            for system in SYSTEMS:
                starts = [int(v) for v in selections[system]]
                lengths = np.diff(np.asarray([*starts, frames], dtype=np.int64)).tolist()
                if starts[0] != 0 or min(lengths) <= 0 or max(lengths) > max_span:
                    raise RuntimeError(f"invalid starts {row['utt_id']} {system} {rate}")
                key = f"{system}@{str(rate).replace('.', 'p')}hz"
                record = dict(row)
                record.update({"system": system, "condition": f"native_{str(rate).replace('.', 'p')}hz",
                    "condition_key": key, "target_rate_hz": rate, "starts": starts,
                    "segment_lengths": lengths,
                    "segment_durations_sec": [float(v)*duration/frames for v in lengths],
                    "realized_rate_hz": len(starts)/duration, "max_span_frames": max_span,
                    "max_span_seconds": max_span*duration/frames,
                    "duration_unit": "seconds", "force_manifest_alignment": True,
                    "dynamic_selector_source": "native_sensevoice_l49" if system in ("atome_style","tadpc_style") else None,
                    "dynamic_parameter": dynamic[rate] if system in ("atome_style","tadpc_style") else None})
                out_rows.append(record); counts[key] += 1; rates[key] += len(starts); durations[key] += duration
        if index % 500 == 0 or index == len(source): print(f"materialized {index}/{len(source)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as h:
        for r in out_rows: h.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.output)
    report = {"status":"passed", "protocol":"decoder_multirate_native_grid_nine_v1",
        "split": args.split, "source_split": args.source_split, "utterances": len(source),
        "records": len(out_rows), "rates_hz": list(RATES), "systems": list(SYSTEMS),
        "condition_counts": dict(sorted(counts.items())),
        "corpus_realized_rate_hz": {k: rates[k]/durations[k] for k in sorted(counts)},
        "dynamic_calibration": dynamic, "selection_uses_test": False,
        "selection_uses_gt": False, "selection_uses_transcript": False,
        "selection_uses_utility": False}
    atomic_json(args.output.with_suffix(args.output.suffix+".validation.json"), report)
    args.output.with_suffix(args.output.suffix+".exit").write_text("0\n", encoding="ascii")
    atomic_json(args.output.with_suffix(args.output.suffix+".acceptance.json"), report)
    args.output.with_suffix(args.output.suffix+".acceptance.exit").write_text("0\n", encoding="ascii")
    print(json.dumps(report, sort_keys=True))

if __name__ == "__main__": main()
