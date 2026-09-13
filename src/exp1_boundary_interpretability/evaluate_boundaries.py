#!/usr/bin/env python3
"""Evaluate DFR algorithms against the six EXP1 reference tracks."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from deferred_models import load_codebooks, load_normalization, load_predictors
from direct_boundaries import select_direct_boundaries
from io_utils import read_jsonl, write_jsonl
from kmeans_selector import PRIMARY_KMEANS_CLUSTERS, code_change_scores, select_kmeans
from merging_selectors import (
    DIRECT_RATE_SYSTEMS,
    cosine_change,
    dense_peak_prominence,
    normalized_mse,
    select,
)
from rate_control import calibrate_ple_tau, calibrate_score_threshold


def load_record_features(record: dict) -> tuple[np.ndarray, np.ndarray]:
    duration = float(record["duration_sec"])
    with np.load(record["feature_path"]) as payload:
        features = payload[
            str(record.get("feature_key", "features"))
        ].astype(np.float32)
        times = (
            payload["times"].astype(np.float32)
            if "times" in payload
            else (
                (np.arange(len(features), dtype=np.float32) + 0.5)
                * duration
                / len(features)
            )
        )
    return features, times


TARGETS = ("phoneme", "syllable", "bpe_subword", "word", "acoustic_event", "vuv")
COLLARS_MS = (20.0, 40.0, 50.0)
PRIMARY_COLLAR_MS = 40.0
CONTROLLED_RATES = (8.0, 6.25, 4.0, 3.0)
DIRECT_SYSTEMS = frozenset({"qwen_timestamp_direct", "mfa_phone_direct"})
MATCHED_CONDITION_NAMES = {
    "matched_qwen_timestamp": "qwen_timestamp_direct",
    "matched_mfa_phone": "mfa_phone_direct",
}
DEFERRED_BASES = frozenset({"elastic_time_greedy", "elastic_time_dp", "dcdit_1d", "hubert_kmeans"})
RATE_CALIBRATION_COLUMNS = (
    "system",
    "parameter_name",
    "parameter_value",
    "target_rate_hz",
    "realized_rate_hz",
    "absolute_error_hz",
    "calibration_split",
    "calibration_records",
    "calibration_duration_sec",
    "conditions",
)


def add_shared_path() -> None:
    shared = Path(__file__).resolve().parents[2] / "shared"
    if str(shared) not in sys.path:
        sys.path.insert(0, str(shared))


add_shared_path()
from boundary_metrics import match_boundaries  # noqa: E402


def selector_name(system: str) -> str:
    if system == "no_merge":
        return system
    for prefix in ("sv_", "qwen_"):
        if system.startswith(prefix):
            value = system[len(prefix) :]
            aliases = {
                "peak": "peak_detection",
                "codecslime": "codecslime_dp",
                "elastic_greedy": "elastic_time_greedy",
                "elastic_dp": "elastic_time_dp",
                "dcdit": "dcdit_1d",
                "kmeans": "hubert_kmeans",
            }
            return aliases.get(value, value)
    return system


def feature_source(system: str) -> str:
    if system.startswith("qwen_"):
        return "qwen"
    return "sensevoice"


def boundary_times(starts: np.ndarray, frame_times: np.ndarray) -> np.ndarray:
    return np.asarray(
        [(float(frame_times[index - 1]) + float(frame_times[index])) / 2.0 for index in starts[1:]],
        dtype=np.float32,
    )


def score_times(frame_times: np.ndarray) -> np.ndarray:
    return ((frame_times[:-1] + frame_times[1:]) / 2.0).astype(np.float32)


def candidate_labels(times: np.ndarray, reference: list[float], collar_sec: float) -> np.ndarray:
    refs = np.asarray(reference, dtype=np.float64)
    if not len(refs):
        return np.zeros(len(times), dtype=np.int8)
    return np.asarray([np.any(np.abs(refs - value) <= collar_sec) for value in times], dtype=np.int8)


def union_boundaries(boundaries: dict[str, list[float]], targets: tuple[str, ...]) -> list[float]:
    return sorted({round(float(value), 6) for target in targets for value in boundaries.get(target, [])})


def all_unions() -> list[tuple[str, ...]]:
    return [combo for size in range(1, len(TARGETS) + 1) for combo in itertools.combinations(TARGETS, size)]


def direct_reference_rates(
    indexes: dict[str, list[dict]],
    ground_truth: dict[str, dict],
    max_span: int,
    matched_rate_controls: list[str],
) -> dict[str, float]:
    """Measure validation-only direct rates after grid snapping and max-span repair."""
    definitions = {
        "matched_qwen_timestamp": ("qwen", "qwen_timestamp_direct"),
        "matched_mfa_phone": ("sensevoice", "mfa_phone_direct"),
    }
    requested = set(matched_rate_controls)
    definitions = {
        condition: definition
        for condition, definition in definitions.items()
        if definition[1] in requested
    }
    output = {}
    for condition, (source, system) in definitions.items():
        if source not in indexes:
            raise RuntimeError(f"cannot measure {condition}: missing {source} feature index")
        segments = 0
        duration = 0.0
        records = [row for row in indexes[source] if row["split"] == "val"]
        for record in records:
            features, times = load_record_features(record)
            if system == "qwen_timestamp_direct":
                boundaries = record.get("qwen_timestamp_boundaries", [])
            else:
                boundaries = ground_truth[record["utt_id"]]["boundaries"]["phoneme"]
            result = select_direct_boundaries(system, features, times, boundaries, max_span)
            segments += len(result.starts)
            duration += float(record["duration_sec"])
        if not records or duration <= 0:
            raise RuntimeError(f"cannot measure {condition}: no validation records")
        output[condition] = float(segments / duration)
    return output


def calibrate(
    records: list[dict],
    systems: list[str],
    rates: list[float],
    max_span: int,
    *,
    local_model=None,  # noqa: ANN001
    normalization: tuple[np.ndarray, np.ndarray] | None = None,
    codebooks: dict | None = None,
    device: str = "cpu",
    calibration_split: str = "timit_train_speaker_heldout_val",
) -> tuple[dict[tuple[str, float], float], list[dict]]:
    by_source: dict[str, dict[str, list[tuple[np.ndarray, float]]]] = defaultdict(lambda: defaultdict(list))
    needed_sources = {feature_source(system) for system in systems if system not in DIRECT_SYSTEMS and system != "no_merge"}
    for record in records:
        if record["feature_source"] not in needed_sources:
            continue
        features, _ = load_record_features(record)
        source = record["feature_source"]
        duration = float(record["duration_sec"])
        cosine = cosine_change(features)
        peak_signal, _ = dense_peak_prominence(features)
        by_source[source]["cosine"].append((cosine, duration))
        by_source[source]["peak"].append((peak_signal, duration))
        if any(selector_name(system) in {"dcdit_1d", "hubert_kmeans"} for system in systems):
            if normalization is None:
                raise RuntimeError("deferred systems require LibriTTS TRAIN normalization")
            mean, std = normalization
            normalized = (features - mean) / std
            if local_model is not None:
                by_source[source]["dcdit"].append((local_model.difficulty(normalized, device), duration))
            if codebooks:
                if PRIMARY_KMEANS_CLUSTERS not in codebooks:
                    raise RuntimeError(
                        f"missing primary K-means codebook K={PRIMARY_KMEANS_CLUSTERS}"
                    )
                _, scores = code_change_scores(normalized, codebooks[PRIMARY_KMEANS_CLUSTERS])
                by_source[source]["kmeans_change"].append((scores, duration))

    parameters: dict[tuple[str, float], float] = {}
    rows = []
    for system in systems:
        base = selector_name(system)
        if base not in {"similarity", "ple", "dcdit_1d", "hubert_kmeans"}:
            continue
        source = feature_source(system)
        for rate in rates:
            if base == "hubert_kmeans":
                if not codebooks:
                    raise RuntimeError("sv_kmeans requires trained codebooks")
                result = calibrate_score_threshold(
                    by_source[source]["kmeans_change"],
                    rate,
                    max_span,
                    system=system,
                    parameter_name="code_change_confidence_threshold",
                    calibration_split=calibration_split,
                )
                value = result.parameter_value
                row = result.to_dict()
            elif base == "similarity":
                result = calibrate_score_threshold(
                    by_source[source]["cosine"],
                    rate,
                    max_span,
                    system=system,
                    parameter_name="cosine_change_threshold",
                    calibration_split=calibration_split,
                )
                value = 1.0 - result.parameter_value
                row = result.to_dict()
                row["boundary_change_threshold"] = row["parameter_value"]
                row["parameter_name"] = "similarity_threshold"
                row["parameter_value"] = value
            elif base == "ple":
                result = calibrate_ple_tau(
                    by_source[source]["cosine"], rate, max_span, calibration_split=calibration_split
                )
                value = result.parameter_value
                row = result.to_dict()
                row["system"] = system
            else:
                result = calibrate_score_threshold(
                    by_source[source]["dcdit"],
                    rate,
                    max_span,
                    system=system,
                    parameter_name="difficulty_threshold",
                    calibration_split=calibration_split,
                )
                value = result.parameter_value
                row = result.to_dict()
            parameters[(system, float(rate))] = float(value)
            rows.append(row)
    return parameters, rows


def summarize(rows: list[dict], ap_store: dict[tuple, tuple[list[int], list[float]]]) -> pd.DataFrame:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["system"], row["condition"], row["target"], row["collar_ms"])].append(row)
    output = []
    for key, items in grouped.items():
        system, condition, target, collar_ms = key
        tp = sum(item["tp"] for item in items)
        fp = sum(item["fp"] for item in items)
        fn = sum(item["fn"] for item in items)
        offsets = [offset for item in items for offset in item["offsets_ms"]]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        labels, scores = ap_store.get(key, ([], []))
        ap = math.nan
        if labels and len(set(labels)) > 1 and np.isfinite(scores).all():
            ap = float(average_precision_score(labels, scores))
        output.append(
            {
                "system": system,
                "condition": condition,
                "target": target,
                "collar_ms": collar_ms,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
                "ap": ap,
                "mean_offset_ms": float(np.mean(offsets)) if offsets else math.nan,
                "mean_abs_offset_ms": float(np.mean(np.abs(offsets))) if offsets else math.nan,
                "median_abs_offset_ms": float(np.median(np.abs(offsets))) if offsets else math.nan,
                "utterances": len(items),
            }
        )
    return pd.DataFrame(output).sort_values(["condition", "target", "f1"], ascending=[True, True, False])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-tracks", "--ground-truth", dest="reference_tracks", type=Path, required=True)
    parser.add_argument("--sensevoice-index", type=Path, required=True)
    parser.add_argument("--qwen-index", type=Path)
    parser.add_argument("--systems", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=list(CONTROLLED_RATES))
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--predictor-dir", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--kmeans-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    max_span = int(config["max_span"])
    gt_rows = read_jsonl(args.reference_tracks)
    gt = {row["utt_id"]: row for row in gt_rows}
    indexes = {"sensevoice": read_jsonl(args.sensevoice_index)}
    if args.qwen_index is not None:
        indexes["qwen"] = read_jsonl(args.qwen_index)
    required_sources = {feature_source(system) for system in args.systems}
    missing_sources = required_sources - set(indexes)
    if missing_sources:
        raise ValueError(f"missing feature indexes for requested systems: {sorted(missing_sources)}")
    deferred_requested = any(selector_name(system) in DEFERRED_BASES for system in args.systems)
    elastic = local = None
    normalization = None
    codebooks = None
    if deferred_requested:
        if args.predictor_dir is None or args.normalization is None:
            raise ValueError("deferred systems require --predictor-dir and --normalization")
        elastic, local = load_predictors(args.predictor_dir, args.device)
        normalization = load_normalization(args.normalization, int(indexes["sensevoice"][0]["feature_dim"]))
        if any(selector_name(system) == "hubert_kmeans" for system in args.systems):
            if args.kmeans_dir is None:
                raise ValueError("sv_kmeans requires --kmeans-dir")
            codebooks = load_codebooks(args.kmeans_dir)
    all_records = [record for source_records in indexes.values() for record in source_records]
    validation = [row for row in all_records if row["split"] == "val"]
    if not validation:
        raise RuntimeError("no validation records available for rate calibration")
    matched_rates = direct_reference_rates(
        indexes,
        gt,
        max_span,
        list(config.get("matched_rate_controls", [])),
    )
    controlled_conditions = [(f"rate_{str(rate).replace('.', '_')}", float(rate)) for rate in args.rates]
    controlled_conditions.extend((condition, rate) for condition, rate in matched_rates.items())
    calibration_rates = list(dict.fromkeys(rate for _, rate in controlled_conditions))
    parameters, calibration_rows = calibrate(
        validation,
        args.systems,
        calibration_rates,
        max_span,
        local_model=local,
        normalization=normalization,
        codebooks=codebooks,
        device=args.device,
        calibration_split="timit_train_speaker_heldout_val",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rate_aliases: dict[float, list[str]] = defaultdict(list)
    for condition, rate in controlled_conditions:
        rate_aliases[float(rate)].append(condition)
    for row in calibration_rows:
        row["conditions"] = json.dumps(rate_aliases.get(float(row["target_rate_hz"]), []))
    calibration_frame = pd.DataFrame(calibration_rows)
    if calibration_frame.empty:
        calibration_frame = pd.DataFrame(columns=RATE_CALIBRATION_COLUMNS)
    calibration_frame.to_csv(args.output_dir / "rate_calibration.csv", index=False)
    (args.output_dir / "direct_reference_rates.json").write_text(
        json.dumps(
            {
                condition: {
                    "reference_system": MATCHED_CONDITION_NAMES[condition],
                    "target_rate_hz": rate,
                    "calibration_split": "timit_train_speaker_heldout_val",
                }
                for condition, rate in matched_rates.items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    metric_rows = []
    union_rows = []
    segment_rows = []
    prediction_rows = []
    confusion: dict[tuple[str, str], Counter] = defaultdict(Counter)
    ap_store: dict[tuple, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    union_store: dict[tuple, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    unions = all_unions()

    for system in args.systems:
        source = feature_source(system)
        records = [row for row in indexes[source] if row["split"] == "test"]
        if args.max_utterances:
            records = records[: args.max_utterances]
        base = selector_name(system)
        conditions: list[tuple[str, float | None]] = (
            [("natural", None)] if base == "no_merge" or system in DIRECT_SYSTEMS else controlled_conditions
        )
        for completed, record in enumerate(records, 1):
            item = gt[record["utt_id"]]
            features, times = load_record_features(record)
            normalized = None
            if base in DEFERRED_BASES:
                if normalization is None:
                    raise RuntimeError("missing deferred normalization")
                normalized = (features - normalization[0]) / normalization[1]
            duration = float(record["duration_sec"])
            cosine = cosine_change(features)
            peak_signal, peak_scores = dense_peak_prominence(features)
            for condition, rate in conditions:
                if system == "qwen_timestamp_direct":
                    result = select_direct_boundaries(
                        system, features, times, record.get("qwen_timestamp_boundaries", []), max_span
                    )
                elif system == "mfa_phone_direct":
                    result = select_direct_boundaries(system, features, times, item["boundaries"]["phoneme"], max_span)
                elif base == "hubert_kmeans":
                    result = select_kmeans(
                        normalized,
                        times,
                        codebooks[PRIMARY_KMEANS_CLUSTERS],
                        max_span,
                        control_parameter=parameters[(system, float(rate))],
                    )
                else:
                    requested = None if rate is None else max(1, int(round(duration * rate)))
                    parameter = None if rate is None else parameters.get((system, float(rate)))
                    selection_features = normalized if base in DEFERRED_BASES else features
                    result = select(
                        base,
                        selection_features,
                        times,
                        requested if base in DIRECT_RATE_SYSTEMS or base in {"elastic_time_greedy", "elastic_time_dp"} else None,
                        max_span,
                        elastic_model=elastic,
                        local_model=local,
                        device=args.device,
                        precomputed=(
                            {"cosine_scores": cosine, "peak_signal": peak_signal, "peak_scores": peak_scores}
                            if base not in DEFERRED_BASES
                            else None
                        ),
                        control_parameter=parameter,
                    )
                predicted = boundary_times(result.starts, times)
                candidate_time = score_times(times)
                candidate_score = result.boundary_scores[1:]
                prediction_rows.append(
                    {
                        "utt_id": record["utt_id"],
                        "audio": record["audio"],
                        "transcript": record.get("transcript", ""),
                        "split": record["split"],
                        "duration_sec": duration,
                        "system": system,
                        "condition": condition,
                        "target_rate_hz": rate,
                        "realized_rate_hz": len(result.starts) / duration,
                        "starts": result.starts.tolist(),
                        "boundary_times": predicted.tolist(),
                        "segment_lengths": result.segment_lengths.tolist(),
                        "boundary_scores": np.nan_to_num(
                            result.boundary_scores, nan=-1.0, neginf=-1.0, posinf=np.finfo(np.float32).max
                        ).tolist(),
                        "feature_source": source,
                        "implementation": result.metadata.get("implementation"),
                    }
                )
                lengths = result.segment_lengths
                segment_rows.append(
                    {
                        "utt_id": record["utt_id"],
                        "system": system,
                        "condition": condition,
                        "target_rate_hz": rate,
                        "realized_rate_hz": len(result.starts) / duration,
                        "segments": len(result.starts),
                        "mean_segment_frames": float(lengths.mean()),
                        "p95_segment_frames": float(np.quantile(lengths, 0.95)),
                        "max_segment_frames": int(lengths.max()),
                        "feature_nmse": normalized_mse(features, result),
                        "objective": result.objective,
                        "runtime_ms": result.runtime_ms,
                        "memory_bytes": result.memory_bytes,
                    }
                )
                for target in TARGETS:
                    reference = item["boundaries"].get(target, [])
                    for collar_ms in COLLARS_MS:
                        tp, fp, fn, offsets = match_boundaries(predicted, reference, collar_ms / 1000.0)
                        row = {
                            "utt_id": record["utt_id"],
                            "system": system,
                            "condition": condition,
                            "target": target,
                            "collar_ms": collar_ms,
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "offsets_ms": [value * 1000.0 for value in offsets],
                        }
                        metric_rows.append(row)
                        if system not in DIRECT_SYSTEMS and base not in {"uniform", "no_merge"}:
                            key = (system, condition, target, collar_ms)
                            ap_store[key][0].extend(candidate_labels(candidate_time, reference, collar_ms / 1000.0).tolist())
                            ap_store[key][1].extend(
                                np.nan_to_num(
                                    candidate_score,
                                    nan=-1.0,
                                    neginf=-1.0,
                                    posinf=np.finfo(np.float32).max,
                                ).tolist()
                            )
                for combo in unions:
                    reference = union_boundaries(item["boundaries"], combo)
                    tp, fp, fn, offsets = match_boundaries(predicted, reference, PRIMARY_COLLAR_MS / 1000.0)
                    union_rows.append(
                        {
                            "utt_id": record["utt_id"],
                            "system": system,
                            "condition": condition,
                            "combination": "+".join(combo),
                            "collar_ms": PRIMARY_COLLAR_MS,
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "offsets_ms": [float(value) * 1000.0 for value in offsets],
                        }
                    )
                for value in predicted:
                    hits = []
                    for target in TARGETS:
                        refs = item["boundaries"].get(target, [])
                        distance = min((abs(float(value) - ref) for ref in refs), default=math.inf)
                        if distance <= PRIMARY_COLLAR_MS / 1000.0:
                            hits.append((distance, target))
                    labels = [target for _, target in sorted(hits)]
                    confusion[(system, condition)][labels[0] if labels else "unmatched"] += 1
                    confusion[(system, condition)]["hybrid:" + ("+".join(labels) if labels else "unmatched")] += 1
            if completed % 50 == 0 or completed == len(records):
                print(f"{system}: {completed}/{len(records)}", flush=True)

    write_jsonl(args.output_dir / "boundary_predictions.jsonl", prediction_rows)
    serializable = pd.DataFrame(metric_rows)
    serializable["offsets_ms"] = serializable["offsets_ms"].map(json.dumps)
    serializable.to_csv(args.output_dir / "per_utterance_boundary_metrics.csv", index=False)
    summarize(metric_rows, ap_store).to_csv(args.output_dir / "boundary_metrics.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(args.output_dir / "segment_statistics.csv", index=False)

    union_aggregate = summarize(
        [
            {**row, "target": row["combination"], "collar_ms": row["collar_ms"]}
            for row in union_rows
        ],
        union_store,
    ).rename(columns={"target": "combination"})
    union_aggregate.to_csv(args.output_dir / "boundary_combinations.csv", index=False)
    union_per_utterance = pd.DataFrame(union_rows)
    union_per_utterance["offsets_ms"] = union_per_utterance["offsets_ms"].map(
        lambda values: json.dumps(values)
    )
    union_per_utterance.to_csv(args.output_dir / "boundary_combinations_per_utterance.csv", index=False)
    confusion_rows = []
    for (system, condition), counts in confusion.items():
        normal = {key: value for key, value in counts.items() if not key.startswith("hybrid:")}
        hybrid = {key[7:]: value for key, value in counts.items() if key.startswith("hybrid:")}
        for kind, values in (("nearest", normal), ("hybrid", hybrid)):
            total = sum(values.values()) or 1
            for label, count in values.items():
                confusion_rows.append(
                    {"system": system, "condition": condition, "kind": kind, "category": label, "count": count, "ratio": count / total}
                )
    pd.DataFrame(confusion_rows).to_csv(args.output_dir / "boundary_type_confusion.csv", index=False)
    (args.output_dir / "boundary.exit").write_text("0\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
