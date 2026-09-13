"""Variable-rate selectors and equal-weight boundary-voting consensus."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from common_selectors import (
    cosine_change,
    dense_peak_prominence,
    minimum_cost_starts,
    peak_budget_starts_from_signal,
    ple_fixed_tau_scores,
    segment_l2_costs,
    threshold_starts,
    uniform_starts,
)


SYSTEMS = (
    "uniform",
    "similarity",
    "codecslime_dp",
    "ple",
    "peak_detection",
    "consensus",
)
CONSENSUS_MEMBERS = (
    "similarity",
    "codecslime_dp",
    "ple",
    "peak_detection",
)
MAX_SEGMENT_SECONDS = 0.640


@dataclass(frozen=True)
class Selection:
    starts: list[int]
    dense_score: np.ndarray


def physical_max_span(frames: int, duration_sec: float) -> int:
    """Largest native-frame span that never exceeds 640 ms."""
    if frames < 1 or duration_sec <= 0:
        raise ValueError("frames and duration must be positive")
    native_rate = frames / duration_sec
    return max(
        1,
        int(math.floor(native_rate * MAX_SEGMENT_SECONDS + 1e-12)),
    )


def requested_segments(duration_sec: float, target_rate_hz: float) -> int:
    if duration_sec <= 0 or target_rate_hz <= 0:
        raise ValueError("duration and target rate must be positive")
    return max(1, int(round(duration_sec * target_rate_hz)))


def resample_feature_grid(
    features: np.ndarray,
    duration_sec: float,
    target_rate_hz: float = 12.5,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    target_frames = max(1, int(round(duration_sec * target_rate_hz)))
    if target_frames == len(values):
        return values.copy()
    source = (np.arange(len(values), dtype=np.float64) + 0.5) / len(values)
    target = (np.arange(target_frames, dtype=np.float64) + 0.5) / target_frames
    output = np.empty((target_frames, values.shape[1]), dtype=np.float32)
    for dimension in range(values.shape[1]):
        output[:, dimension] = np.interp(
            target, source, values[:, dimension]
        ).astype(np.float32)
    return output


def map_starts_to_native(
    starts: list[int],
    source_frames: int,
    native_frames: int,
    native_max_span: int,
) -> list[int]:
    mapped = sorted(
        {
            min(
                native_frames - 1,
                max(0, int(round(start * native_frames / source_frames))),
            )
            for start in starts
        }
    )
    if not mapped or mapped[0] != 0:
        mapped.insert(0, 0)
    output = [0]
    for boundary in mapped[1:]:
        while boundary - output[-1] > native_max_span:
            output.append(output[-1] + native_max_span)
        if boundary > output[-1]:
            output.append(boundary)
    while native_frames - output[-1] > native_max_span:
        output.append(output[-1] + native_max_span)
    return output


def map_exact_starts_to_native(
    starts: list[int],
    source_frames: int,
    native_frames: int,
    native_max_span: int,
) -> list[int] | None:
    mapped = sorted(
        {
            min(
                native_frames - 1,
                max(0, int(round(start * native_frames / source_frames))),
            )
            for start in starts
        }
    )
    if len(mapped) != len(starts) or not mapped or mapped[0] != 0:
        return None
    lengths = np.diff(
        np.asarray([*mapped, native_frames], dtype=np.int64)
    )
    if np.any(lengths <= 0) or int(lengths.max()) > native_max_span:
        return None
    return mapped


def select_base(
    system: str,
    features: np.ndarray,
    duration_sec: float,
    target_rate_hz: float,
    max_span: int,
    controls: dict,
) -> Selection:
    frames = len(features)
    segments = requested_segments(duration_sec, target_rate_hz)
    change = cosine_change(features)
    if system == "uniform":
        return Selection(uniform_starts(frames, segments, max_span), np.zeros(frames))
    if system == "similarity":
        return Selection(
            threshold_starts(
                change, float(controls["similarity"]["value"]), max_span
            ),
            change,
        )
    if system == "ple":
        return Selection(
            ple_fixed_tau_scores(
                change, float(controls["ple"]["value"]), max_span
            ),
            change,
        )
    if system == "peak_detection":
        signal, prominence = dense_peak_prominence(features)
        starts, _ = peak_budget_starts_from_signal(
            signal, prominence, segments, max_span
        )
        return Selection(starts, prominence)
    if system == "codecslime_dp":
        common = resample_feature_grid(features, duration_sec, 12.5)
        initial_span = physical_max_span(len(common), duration_sec)
        mapped = None
        for common_max_span in range(initial_span, 0, -1):
            minimum_segments = math.ceil(len(common) / common_max_span)
            if segments < minimum_segments or segments > len(common):
                continue
            starts, _ = minimum_cost_starts(
                segment_l2_costs(common, common_max_span),
                len(common),
                segments,
                common_max_span,
            )
            mapped = map_exact_starts_to_native(
                starts, len(common), frames, max_span
            )
            if mapped is not None:
                break
        if mapped is None:
            raise RuntimeError(
                "CodecSlime could not preserve exact-K on the native grid"
            )
        dense = np.zeros(frames, dtype=np.float32)
        dense[mapped[1:]] = 1.0
        return Selection(mapped, dense)
    raise KeyError(system)


def boundary_vote_score(
    member_starts: dict[str, list[int]],
    frames: int,
    duration_sec: float,
    tolerance_sec: float = 0.040,
) -> np.ndarray:
    if set(member_starts) != set(CONSENSUS_MEMBERS):
        raise ValueError("consensus requires every configured member")
    frame_seconds = duration_sec / frames
    radius = max(1, int(round(tolerance_sec / frame_seconds)))
    vote = np.zeros(frames, dtype=np.float32)
    for starts in member_starts.values():
        member_vote = np.zeros(frames, dtype=np.float32)
        for boundary in starts[1:]:
            low = max(1, boundary - radius)
            high = min(frames - 1, boundary + radius)
            for index in range(low, high + 1):
                weight = 1.0 - abs(index - boundary) / (radius + 1.0)
                member_vote[index] = max(member_vote[index], weight)
        vote += member_vote
    vote /= float(len(member_starts))
    vote[0] = 0.0
    return vote


def select_consensus(
    vote: np.ndarray,
    duration_sec: float,
    target_rate_hz: float,
    max_span: int,
    control: dict,
) -> Selection:
    mode = str(control.get("mode", "threshold"))
    if mode == "threshold":
        starts = threshold_starts(vote, float(control["value"]), max_span)
    elif mode == "exact_budget":
        starts, _ = peak_budget_starts_from_signal(
            vote,
            vote,
            requested_segments(duration_sec, target_rate_hz),
            max_span,
        )
    else:
        raise ValueError(f"unsupported consensus control mode: {mode}")
    return Selection(starts=starts, dense_score=vote)


def select_all(
    features: np.ndarray,
    duration_sec: float,
    target_rate_hz: float,
    max_span: int,
    controls: dict,
) -> dict[str, Selection]:
    selections = {
        system: select_base(
            system,
            features,
            duration_sec,
            target_rate_hz,
            max_span,
            controls,
        )
        for system in SYSTEMS
        if system != "consensus"
    }
    vote = boundary_vote_score(
        {
            system: selections[system].starts
            for system in CONSENSUS_MEMBERS
        },
        len(features),
        duration_sec,
    )
    selections["consensus"] = select_consensus(
        vote,
        duration_sec,
        target_rate_hz,
        max_span,
        controls["consensus"],
    )
    return selections


def validate_starts(starts: list[int], frames: int, max_span: int) -> None:
    if not starts or starts[0] != 0 or starts != sorted(set(starts)):
        raise RuntimeError("starts must be unique, sorted, and begin at zero")
    if starts[-1] >= frames:
        raise RuntimeError("last start exceeds feature length")
    lengths = np.diff(np.asarray([*starts, frames], dtype=np.int64))
    if np.any(lengths <= 0) or int(lengths.sum()) != frames:
        raise RuntimeError("segments do not cover every frame exactly once")
    if int(lengths.max()) > max_span:
        raise RuntimeError("segment exceeds the physical max span")
