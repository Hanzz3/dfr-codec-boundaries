#!/usr/bin/env python3
"""Validate the two-utterance Direct q1-sem cache and Qwen decode smoke."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np


SYSTEMS = {
    "uniform",
    "flexicodec_threshold",
    "codecslime_dp",
    "ple",
    "elastic_time_greedy",
    "elastic_time_dp",
    "dcdit_1d",
    "atome_style",
    "tadpc_style",
}


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--utterances", type=int, default=2)
    args = parser.parse_args()

    manifest_rows = [
        json.loads(line) for line in args.manifest.open(encoding="utf-8") if line.strip()
    ]
    prediction_rows = [
        json.loads(line)
        for line in args.predictions.open(encoding="utf-8")
        if line.strip()
    ]
    expected = args.utterances * len(SYSTEMS)
    for name, rows in (("manifest", manifest_rows), ("predictions", prediction_rows)):
        keys = {(str(row["utt_id"]), str(row["system"])) for row in rows}
        counts = Counter(str(row["system"]) for row in rows)
        if len(rows) != expected or len(keys) != expected:
            raise RuntimeError(f"{name} smoke key coverage mismatch: {len(rows)}")
        if set(counts) != SYSTEMS or set(counts.values()) != {args.utterances}:
            raise RuntimeError(f"{name} smoke system coverage mismatch: {counts}")

    if {
        (str(row["utt_id"]), str(row["system"])) for row in manifest_rows
    } != {
        (str(row["utt_id"]), str(row["system"])) for row in prediction_rows
    }:
        raise RuntimeError("cache and decode smoke keys differ")

    loaded_paths: dict[Path, dict[str, np.ndarray]] = {}
    for row in manifest_rows:
        path = Path(row["feature_path"])
        if path not in loaded_paths:
            with np.load(path) as payload:
                loaded_paths[path] = {
                    key: payload[key] for key in payload.files if key.startswith("q1__")
                }
        matrix = loaded_paths[path][str(row["feature_key"])]
        if matrix.ndim != 2 or matrix.shape[1] != 512 or matrix.shape[0] <= 0:
            raise RuntimeError(f"invalid smoke latent shape: {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise RuntimeError("non-finite smoke latent")
        if row.get("latent_definition") != "semantic_component_from_shared_q8_encode":
            raise RuntimeError("unexpected q1-sem latent definition")

    for row in prediction_rows:
        for field in ("reference_words", "substitutions", "deletions", "insertions"):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"non-finite decode field: {field}")
        if int(row["reference_words"]) <= 0:
            raise RuntimeError("empty Direct-ASR reference")
        if "hypothesis" not in row:
            raise RuntimeError("missing Direct-ASR hypothesis")

    report = {
        "status": "passed",
        "protocol": "timit_current_nine_direct_q1_sem_smoke_v1",
        "utterances": args.utterances,
        "records": expected,
        "feature_dim": 512,
        "latent_definition": "semantic_component_from_shared_q8_encode",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(args.output_dir / "acceptance.json", json.dumps(report, indent=2) + "\n")
    atomic_text(args.output_dir / "acceptance.exit", "0\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
