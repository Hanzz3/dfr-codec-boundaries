"""Validated non-learned selectors used by the LM-ASR matrix."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import peak_prominences

try:
    from numba import njit
except ImportError:  # pragma: no cover - the AIStation runtime includes numba.
    njit = None


def clean_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("features must have shape [T, D] with T > 0")
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def feasible_segments(length: int, requested: int, max_span: int) -> int:
    if length < 1 or max_span < 1:
        raise ValueError("length and max_span must be positive")
    minimum = int(np.ceil(length / max_span))
    return min(length, max(minimum, int(requested)))


def cosine_change(features: np.ndarray) -> np.ndarray:
    values = clean_features(features)
    scores = np.zeros(len(values), dtype=np.float32)
    if len(values) == 1:
        return scores
    norms = np.linalg.norm(values, axis=1)
    safe = np.where(norms < 1e-8, 1.0, norms)
    unit = values / safe[:, None]
    similarity = np.sum(unit[1:] * unit[:-1], axis=1)
    both_zero = (norms[1:] < 1e-8) & (norms[:-1] < 1e-8)
    similarity[both_zero] = 1.0
    scores[1:] = np.clip(1.0 - similarity, 0.0, 2.0)
    return scores


def threshold_starts(scores: np.ndarray, threshold: float, max_span: int) -> list[int]:
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


def uniform_starts(length: int, segments: int, max_span: int) -> list[int]:
    segments = feasible_segments(length, segments, max_span)
    base, remainder = divmod(length, segments)
    lengths = [base + (1 if index < remainder else 0) for index in range(segments)]
    starts = [0]
    for size in lengths[:-1]:
        starts.append(starts[-1] + size)
    return starts


def segment_l2_costs_python(features: np.ndarray, max_span: int) -> np.ndarray:
    values = clean_features(features).astype(np.float64)
    length = len(values)
    prefix = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    costs = np.full((length, max_span + 1), np.inf, dtype=np.float64)
    for start in range(length):
        for size in range(1, min(max_span, length - start) + 1):
            end = start + size
            mean = (prefix[end] - prefix[start]) / size
            costs[start, size] = float(
                np.linalg.norm(values[start:end] - mean, axis=1).sum()
            )
    return costs


def minimum_cost_starts_python(
    costs: np.ndarray,
    length: int,
    segments: int,
    max_span: int,
) -> tuple[list[int], float]:
    segments = feasible_segments(length, segments, max_span)
    dynamic = np.full((segments + 1, length + 1), np.inf, dtype=np.float64)
    parent = np.full((segments + 1, length + 1), -1, dtype=np.int64)
    dynamic[0, 0] = 0.0
    for group in range(1, segments + 1):
        for end in range(1, length + 1):
            for size in range(1, min(max_span, end) + 1):
                start = end - size
                previous = dynamic[group - 1, start]
                if not np.isfinite(previous):
                    continue
                candidate = previous + costs[start, size]
                if candidate < dynamic[group, end]:
                    dynamic[group, end] = candidate
                    parent[group, end] = start
    if not np.isfinite(dynamic[segments, length]):
        raise RuntimeError("no feasible minimum-cost segmentation")
    starts = []
    end = length
    for group in range(segments, 0, -1):
        start = int(parent[group, end])
        starts.append(start)
        end = start
    starts.reverse()
    return starts, float(dynamic[segments, length])


if njit is not None:

    @njit(cache=True)
    def _segment_l2_costs_numba(values: np.ndarray, max_span: int) -> np.ndarray:
        length, dimension = values.shape
        prefix = np.zeros((length + 1, dimension), dtype=np.float64)
        for index in range(length):
            for feature in range(dimension):
                prefix[index + 1, feature] = (
                    prefix[index, feature] + values[index, feature]
                )
        costs = np.full((length, max_span + 1), np.inf, dtype=np.float64)
        for start in range(length):
            maximum = min(max_span, length - start)
            for size in range(1, maximum + 1):
                end = start + size
                total = 0.0
                for frame in range(start, end):
                    squared = 0.0
                    for feature in range(dimension):
                        mean = (
                            prefix[end, feature] - prefix[start, feature]
                        ) / size
                        difference = values[frame, feature] - mean
                        squared += difference * difference
                    total += np.sqrt(squared)
                costs[start, size] = total
        return costs

    @njit(cache=True)
    def _minimum_cost_numba(
        costs: np.ndarray,
        length: int,
        segments: int,
        max_span: int,
    ) -> tuple[np.ndarray, float]:
        dynamic = np.full((segments + 1, length + 1), np.inf, dtype=np.float64)
        parent = np.full((segments + 1, length + 1), -1, dtype=np.int64)
        dynamic[0, 0] = 0.0
        for group in range(1, segments + 1):
            for end in range(1, length + 1):
                maximum = min(max_span, end)
                for size in range(1, maximum + 1):
                    start = end - size
                    previous = dynamic[group - 1, start]
                    if not np.isfinite(previous):
                        continue
                    candidate = previous + costs[start, size]
                    if candidate < dynamic[group, end]:
                        dynamic[group, end] = candidate
                        parent[group, end] = start
        starts = np.empty(segments, dtype=np.int64)
        end = length
        for group in range(segments, 0, -1):
            start = parent[group, end]
            starts[group - 1] = start
            end = start
        return starts, dynamic[segments, length]


def segment_l2_costs(features: np.ndarray, max_span: int) -> np.ndarray:
    values = clean_features(features).astype(np.float64)
    if njit is None:
        return segment_l2_costs_python(values, max_span)
    return _segment_l2_costs_numba(values, int(max_span))


def minimum_cost_starts(
    costs: np.ndarray,
    length: int,
    segments: int,
    max_span: int,
) -> tuple[list[int], float]:
    segments = feasible_segments(length, segments, max_span)
    if njit is None:
        return minimum_cost_starts_python(
            costs, length, segments, max_span
        )
    starts, objective = _minimum_cost_numba(
        np.asarray(costs, dtype=np.float64),
        int(length),
        int(segments),
        int(max_span),
    )
    if not np.isfinite(objective) or int(starts[0]) != 0:
        raise RuntimeError("no feasible minimum-cost segmentation")
    return starts.tolist(), float(objective)


def ple_fixed_tau_scores(scores: np.ndarray, tau: float, max_span: int) -> list[int]:
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    values = np.nan_to_num(
        np.asarray(scores, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    starts = [0]
    cumulative = 0.0
    next_threshold = float(tau)
    for index in range(1, len(values)):
        cumulative += max(0.0, float(values[index]))
        crossed = cumulative >= next_threshold
        if crossed or index - starts[-1] >= max_span:
            starts.append(index)
            if crossed:
                next_threshold += float(tau)
    return starts


def utterance_zscore(features: np.ndarray) -> np.ndarray:
    values = clean_features(features)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (values - mean) / std


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if width <= 1 or len(array) <= 2:
        return array.copy()
    width = min(int(width), len(array))
    kernel = np.full(width, 1.0 / width, dtype=np.float32)
    return np.convolve(array, kernel, mode="same").astype(np.float32)


def dense_peak_prominence(
    features: np.ndarray,
    smoothing_width: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    signal = moving_average(cosine_change(utterance_zscore(features)), smoothing_width)
    dense = np.zeros(len(signal), dtype=np.float32)
    candidates = np.arange(1, max(1, len(signal) - 1), dtype=np.int64)
    if len(candidates):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dense[candidates] = peak_prominences(signal, candidates)[0].astype(
                np.float32
            )
    return signal, dense


def peak_budget_starts_from_signal(
    signal: np.ndarray,
    dense_prominence: np.ndarray,
    segments: int,
    max_span: int,
) -> tuple[list[int], float]:
    values = np.nan_to_num(
        np.asarray(signal, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    prominence = np.nan_to_num(
        np.asarray(dense_prominence, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if values.ndim != 1 or prominence.shape != values.shape or not len(values):
        raise ValueError("signal and prominence must be aligned vectors")
    target = feasible_segments(len(values), segments, max_span)
    shifted = values - float(values.min())
    scale = float(shifted.max())
    secondary = shifted / scale if scale > 1e-12 else np.zeros_like(shifted)
    primary_scale = max(float(prominence.max()), 1.0)
    ranking = prominence + secondary * primary_scale * 1e-6
    ranking[0] = 0.0
    costs = np.full((len(values), max_span + 1), np.inf, dtype=np.float64)
    for start in range(len(values)):
        for size in range(1, min(max_span, len(values) - start) + 1):
            costs[start, size] = -float(ranking[start])
    starts, negative_score = minimum_cost_starts(
        costs, len(values), target, max_span
    )
    return starts, -negative_score
