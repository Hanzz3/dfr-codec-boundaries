#!/usr/bin/env python3
"""Generate unlabeled TEST segmentations with one exact corpus token budget."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from io_utils import read_jsonl, write_jsonl
from merging_selectors import cosine_change, select
from rate_control import calibrate_ple_tau, calibrate_score_threshold


SYSTEM_TO_BASE = {
    "sv_uniform": "uniform",
    "sv_similarity": "similarity",
    "sv_ple": "ple",
}


def allocate_uniform_budgets(
    durations: list[float],
    lengths: list[int],
    total_segments: int,
    max_span: int,
) -> list[int]:
    minimum = [max(1, math.ceil(length / max_span)) for length in lengths]
    maximum = list(lengths)
    if not sum(minimum) <= total_segments <= sum(maximum):
        raise ValueError("requested corpus budget is infeasible")
    rate = total_segments / sum(durations)
    ideals = [duration * rate for duration in durations]
    budgets = [
        min(high, max(low, int(math.floor(ideal))))
        for ideal, low, high in zip(ideals, minimum, maximum)
    ]
    remaining = total_segments - sum(budgets)
    while remaining > 0:
        candidates = [
            (ideals[index] - budgets[index], -index, index)
            for index in range(len(budgets))
            if budgets[index] < maximum[index]
        ]
        if not candidates:
            raise RuntimeError("cannot allocate remaining uniform segments")
        index = max(candidates)[2]
        budgets[index] += 1
        remaining -= 1
    while remaining < 0:
        candidates = [
            (budgets[index] - ideals[index], -index, index)
            for index in range(len(budgets))
            if budgets[index] > minimum[index]
        ]
        if not candidates:
            raise RuntimeError("cannot remove excess uniform segments")
        index = max(candidates)[2]
        budgets[index] -= 1
        remaining += 1
    return budgets


def exact_global_budget(
    starts_by_utterance: list[list[int]],
    scores_by_utterance: list[np.ndarray],
    lengths: list[int],
    target_segments: int,
    max_span: int,
) -> tuple[list[list[int]], int]:
    starts = [sorted(set(values)) for values in starts_by_utterance]
    correction = target_segments - sum(len(values) for values in starts)
    applied = 0
    while correction < 0:
        candidates: list[tuple[float, int, int]] = []
        for utterance, values in enumerate(starts):
            for position in range(1, len(values)):
                left = values[position - 1]
                right = values[position + 1] if position + 1 < len(values) else lengths[utterance]
                if right - left <= max_span:
                    boundary = values[position]
                    candidates.append((float(scores_by_utterance[utterance][boundary]), utterance, boundary))
        if not candidates:
            raise RuntimeError("cannot remove enough boundaries while respecting max-span")
        _, utterance, boundary = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        starts[utterance].remove(boundary)
        correction += 1
        applied -= 1
    while correction > 0:
        candidates = []
        for utterance, values in enumerate(starts):
            observed = set(values)
            for boundary in range(1, lengths[utterance]):
                if boundary not in observed:
                    candidates.append((float(scores_by_utterance[utterance][boundary]), utterance, boundary))
        if not candidates:
            raise RuntimeError("cannot add enough boundaries")
        _, utterance, boundary = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
        starts[utterance].append(boundary)
        starts[utterance].sort()
        correction -= 1
        applied += 1
    return starts, applied


def boundary_times(starts: list[int], times: np.ndarray) -> list[float]:
    return [
        (float(times[index - 1]) + float(times[index])) / 2.0
        for index in starts[1:]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", default=list(SYSTEM_TO_BASE))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-span", type=int, default=8)
    parser.add_argument("--expected-utterances", type=int, default=1680)
    args = parser.parse_args()

    unknown = sorted(set(args.systems) - set(SYSTEM_TO_BASE))
    if unknown:
        raise ValueError(f"unsupported systems: {unknown}")
    records = [row for row in read_jsonl(args.feature_index) if row["split"] == args.split]
    references = [
        row for row in read_jsonl(args.reference_predictions)
        if row["split"] == args.split
    ]
    if len(records) != args.expected_utterances or len(references) != args.expected_utterances:
        raise RuntimeError(
            f"expected {args.expected_utterances} records, found features={len(records)} references={len(references)}"
        )
    reference_by_id = {row["utt_id"]: row for row in references}
    if len(reference_by_id) != len(references):
        raise RuntimeError("duplicate reference utterances")
    if set(reference_by_id) != {row["utt_id"] for row in records}:
        raise RuntimeError("feature/reference utterance sets differ")

    features: list[np.ndarray] = []
    times: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    durations: list[float] = []
    lengths: list[int] = []
    for record in records:
        with np.load(record["feature_path"]) as payload:
            feature = payload["features"].astype(np.float32)
            frame_times = payload["times"].astype(np.float32)
        features.append(feature)
        times.append(frame_times)
        scores.append(cosine_change(feature))
        durations.append(float(record["duration_sec"]))
        lengths.append(len(feature))

    target_total = sum(len(reference_by_id[row["utt_id"]]["starts"]) for row in records)
    total_duration = sum(durations)
    target_corpus_rate = target_total / total_duration
    trajectories = list(zip(scores, durations))
    similarity_calibration = calibrate_score_threshold(
        trajectories,
        target_corpus_rate,
        args.max_span,
        system="sv_similarity",
        parameter_name="cosine_change_threshold",
        calibration_split="timit_test_unlabeled_exact_budget",
    )
    ple_calibration = calibrate_ple_tau(
        trajectories,
        target_corpus_rate,
        args.max_span,
        calibration_split="timit_test_unlabeled_exact_budget",
    )
    parameters = {
        "sv_similarity": 1.0 - similarity_calibration.parameter_value,
        "sv_ple": ple_calibration.parameter_value,
    }
    uniform_budgets = allocate_uniform_budgets(
        durations, lengths, target_total, args.max_span
    )

    rows: list[dict] = []
    validation_systems: dict[str, dict] = {}
    for system in args.systems:
        base = SYSTEM_TO_BASE[system]
        raw_starts: list[list[int]] = []
        if system == "sv_uniform":
            for feature, frame_times, budget in zip(features, times, uniform_budgets):
                result = select(base, feature, frame_times, budget, args.max_span)
                raw_starts.append(result.starts.tolist())
            corrected = raw_starts
            correction = 0
            control_parameter = None
        else:
            control_parameter = parameters[system]
            for feature, frame_times, score in zip(features, times, scores):
                result = select(
                    base,
                    feature,
                    frame_times,
                    None,
                    args.max_span,
                    precomputed={"cosine_scores": score},
                    control_parameter=control_parameter,
                )
                raw_starts.append(result.starts.tolist())
            corrected, correction = exact_global_budget(
                raw_starts, scores, lengths, target_total, args.max_span
            )

        total = sum(len(values) for values in corrected)
        if total != target_total:
            raise RuntimeError(f"{system} token budget mismatch: {total}/{target_total}")
        for record, frame_times, score, values, length in zip(
            records, times, scores, corrected, lengths
        ):
            segment_lengths = np.diff([*values, length]).astype(np.int64)
            if values[0] != 0 or np.any(segment_lengths < 1) or np.any(segment_lengths > args.max_span):
                raise RuntimeError(f"invalid segmentation: {system} {record['utt_id']}")
            duration = float(record["duration_sec"])
            rows.append(
                {
                    "utt_id": record["utt_id"],
                    "audio": record["audio"],
                    "transcript": record.get("transcript", ""),
                    "split": record["split"],
                    "duration_sec": duration,
                    "system": system,
                    "condition": "matched_syllable_budget",
                    "target_rate_hz": target_corpus_rate,
                    "realized_rate_hz": len(values) / duration,
                    "starts": values,
                    "boundary_times": boundary_times(values, frame_times),
                    "segment_lengths": segment_lengths.tolist(),
                    "boundary_scores": np.nan_to_num(score, nan=-1.0).tolist(),
                    "feature_source": "sensevoice",
                    "implementation": f"{base}_test_unlabeled_exact_corpus_budget_v1",
                    "control_parameter": control_parameter,
                    "target_total_segments": target_total,
                }
            )
        validation_systems[system] = {
            "segments": total,
            "corpus_rate_hz": total / total_duration,
            "mean_utterance_rate_hz": float(
                np.mean([len(values) / duration for values, duration in zip(corrected, durations)])
            ),
            "raw_segments": sum(len(values) for values in raw_starts),
            "budget_correction_segments": correction,
            "control_parameter": control_parameter,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "boundary_predictions.jsonl", rows)
    validation = {
        "status": "passed",
        "protocol": "test_features_unlabeled_exact_corpus_token_budget",
        "ground_truth_used_for_rate_control": False,
        "reference_system": "syllable_direct",
        "utterances": len(records),
        "target_total_segments": target_total,
        "total_duration_sec": total_duration,
        "target_corpus_rate_hz": target_corpus_rate,
        "reference_mean_utterance_rate_hz": float(
            np.mean([reference_by_id[row["utt_id"]]["realized_rate_hz"] for row in records])
        ),
        "systems": validation_systems,
        "calibration": {
            "sv_similarity": similarity_calibration.to_dict(),
            "sv_ple": ple_calibration.to_dict(),
        },
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "predictions.exit").write_text("0\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
