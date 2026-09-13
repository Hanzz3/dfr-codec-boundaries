"""Validation-only calibration for selectors without native rate control."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RateCalibration:
    system: str
    parameter_name: str
    parameter_value: float
    target_rate_hz: float
    realized_rate_hz: float
    absolute_error_hz: float
    calibration_split: str
    calibration_records: int
    calibration_duration_sec: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def threshold_starts(scores: np.ndarray, threshold: float, max_span: int) -> list[int]:
    """Start a segment at a large score or when the span constraint requires it."""
    if max_span < 1:
        raise ValueError("max_span must be positive")
    values = np.nan_to_num(
        np.asarray(scores, dtype=np.float64),
        nan=-np.inf,
        posinf=np.inf,
        neginf=-np.inf,
    )
    if values.ndim != 1 or not len(values):
        raise ValueError("scores must be a non-empty vector")
    starts = [0]
    for index in range(1, len(values)):
        if values[index] >= threshold or index - starts[-1] >= max_span:
            starts.append(index)
    return starts


def _corpus_rate(segment_counts: list[int], durations: list[float]) -> float:
    return float(sum(segment_counts) / max(sum(durations), 1e-12))


def realized_threshold_rate(
    trajectories: list[tuple[np.ndarray, float]],
    threshold: float,
    max_span: int,
) -> float:
    return _corpus_rate(
        [len(threshold_starts(scores, threshold, max_span)) for scores, _ in trajectories],
        [duration for _, duration in trajectories],
    )


def calibrate_score_threshold(
    trajectories: list[tuple[np.ndarray, float]],
    target_rate_hz: float,
    max_span: int,
    *,
    system: str,
    parameter_name: str,
    calibration_split: str = "val",
) -> RateCalibration:
    """Choose one frozen corpus-level cutoff; never choose a TEST utterance budget."""
    if not trajectories:
        raise ValueError("calibration trajectories are empty")
    if target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be positive")
    finite = [
        np.asarray(scores, dtype=np.float64)[np.isfinite(scores)]
        for scores, _ in trajectories
        if np.isfinite(scores).any()
    ]
    unique = np.unique(np.concatenate(finite)) if finite else np.asarray([0.0])
    scale = max(1.0, float(np.max(np.abs(unique))))
    epsilon = np.finfo(np.float64).eps * scale * 8.0
    candidates = np.concatenate(([unique[0] - epsilon], unique, [unique[-1] + epsilon]))

    cache: dict[int, float] = {}

    def rate_at(index: int) -> float:
        if index not in cache:
            cache[index] = realized_threshold_rate(trajectories, float(candidates[index]), max_span)
        return cache[index]

    low, high = 0, len(candidates) - 1
    while low < high:
        middle = (low + high) // 2
        if rate_at(middle) <= target_rate_hz:
            high = middle
        else:
            low = middle + 1
    neighborhood = range(max(0, low - 2), min(len(candidates), low + 3))
    best_index = min(
        neighborhood,
        key=lambda index: (
            abs(rate_at(index) - target_rate_hz),
            rate_at(index),
            -float(candidates[index]),
        ),
    )
    realized = rate_at(best_index)
    return RateCalibration(
        system=system,
        parameter_name=parameter_name,
        parameter_value=float(candidates[best_index]),
        target_rate_hz=float(target_rate_hz),
        realized_rate_hz=realized,
        absolute_error_hz=abs(realized - target_rate_hz),
        calibration_split=calibration_split,
        calibration_records=len(trajectories),
        calibration_duration_sec=float(sum(duration for _, duration in trajectories)),
    )


def realized_ple_rate(
    trajectories: list[tuple[np.ndarray, float]],
    tau: float,
    max_span: int,
) -> float:
    # Local import avoids coupling the generic calibration helpers to selector models.
    from merging_selectors import ple_fixed_tau_scores

    return _corpus_rate(
        [len(ple_fixed_tau_scores(scores, tau, max_span)) for scores, _ in trajectories],
        [duration for _, duration in trajectories],
    )


def realized_peak_rate(
    trajectories: list[tuple[np.ndarray, float]],
    prominence: float,
    max_span: int,
) -> float:
    from merging_selectors import peak_starts_from_signal

    return _corpus_rate(
        [len(peak_starts_from_signal(signal, prominence, max_span)) for signal, _ in trajectories],
        [duration for _, duration in trajectories],
    )


def calibrate_peak_prominence(
    trajectories: list[tuple[np.ndarray, float]],
    target_rate_hz: float,
    max_span: int,
    *,
    calibration_split: str = "val",
) -> RateCalibration:
    """Calibrate one corpus-level scipy peak prominence on validation only."""
    if not trajectories:
        raise ValueError("calibration trajectories are empty")
    if target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be positive")
    finite = [signal[np.isfinite(signal)] for signal, _ in trajectories if np.isfinite(signal).any()]
    maximum = max((float(np.max(values) - np.min(values)) for values in finite), default=1.0)
    low = 0.0
    high = max(maximum, np.finfo(np.float64).eps)
    for _ in range(60):
        middle = (low + high) / 2.0
        if realized_peak_rate(trajectories, middle, max_span) > target_rate_hz:
            low = middle
        else:
            high = middle
    candidates = (low, (low + high) / 2.0, high)
    scored = [(value, realized_peak_rate(trajectories, value, max_span)) for value in candidates]
    prominence, realized = min(
        scored,
        key=lambda item: (abs(item[1] - target_rate_hz), item[1], -item[0]),
    )
    return RateCalibration(
        system="peak_detection",
        parameter_name="prominence",
        parameter_value=float(prominence),
        target_rate_hz=float(target_rate_hz),
        realized_rate_hz=float(realized),
        absolute_error_hz=abs(float(realized) - target_rate_hz),
        calibration_split=calibration_split,
        calibration_records=len(trajectories),
        calibration_duration_sec=float(sum(duration for _, duration in trajectories)),
    )


def calibrate_ple_tau(
    trajectories: list[tuple[np.ndarray, float]],
    target_rate_hz: float,
    max_span: int,
    *,
    calibration_split: str = "val",
) -> RateCalibration:
    if not trajectories:
        raise ValueError("calibration trajectories are empty")
    if target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be positive")
    maximum_path = max(
        float(np.maximum(np.nan_to_num(scores, nan=0.0), 0.0).sum())
        for scores, _ in trajectories
    )
    low = max(np.finfo(np.float64).eps, maximum_path * 1e-12)
    high = max(1.0, maximum_path + 1.0)
    for _ in range(60):
        middle = (low + high) / 2.0
        if realized_ple_rate(trajectories, middle, max_span) > target_rate_hz:
            low = middle
        else:
            high = middle
    candidates = (low, (low + high) / 2.0, high)
    scored = [(tau, realized_ple_rate(trajectories, tau, max_span)) for tau in candidates]
    tau, realized = min(scored, key=lambda item: (abs(item[1] - target_rate_hz), item[1], -item[0]))
    return RateCalibration(
        system="ple",
        parameter_name="tau",
        parameter_value=float(tau),
        target_rate_hz=float(target_rate_hz),
        realized_rate_hz=float(realized),
        absolute_error_hz=abs(float(realized) - target_rate_hz),
        calibration_split=calibration_split,
        calibration_records=len(trajectories),
        calibration_duration_sec=float(sum(duration for _, duration in trajectories)),
    )
