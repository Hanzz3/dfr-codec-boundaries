from __future__ import annotations

import math
import time
import warnings
from itertools import combinations

import numpy as np
from scipy.signal import find_peaks, peak_prominences

from algorithm_types import SelectionResult, feasible_segments, result_from_starts, validate_result
from models import ElasticPredictor, LocalPredictabilityModel
from rate_control import threshold_starts


DIRECT_RATE_SYSTEMS = (
    "uniform",
    "codecslime_dp",
    "peak_detection",
)
CALIBRATED_SYSTEMS = (
    "similarity",
    "ple",
)
SYSTEMS = ("no_merge",) + DIRECT_RATE_SYSTEMS + CALIBRATED_SYSTEMS

# These systems consume LibriTTS TRAIN-normalized features and enter the
# trained-selector extension only after their checkpoints pass acceptance.
TRAINED_FEATURE_SYSTEMS = ("elastic_time_greedy", "elastic_time_dp", "dcdit_1d", "hubert_kmeans")

TRAIN_NORMALIZED_SYSTEMS = frozenset(TRAINED_FEATURE_SYSTEMS)


def selector_features(name: str, raw: np.ndarray, train_normalized: np.ndarray) -> np.ndarray:
    """Use TRAIN normalization only for predictors trained in that feature space."""
    return train_normalized if name in TRAIN_NORMALIZED_SYSTEMS else raw


def selector_feature_protocol(name: str) -> str:
    if name in TRAIN_NORMALIZED_SYSTEMS:
        return "libritts_train_global_standardized"
    if name in {"uniform", "no_merge"}:
        return "feature_independent"
    return "official_skip_normalize_raw"


def clean_features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or not len(x):
        raise ValueError("features must have shape [T, D] with T > 0")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def cosine_change(features: np.ndarray) -> np.ndarray:
    x = clean_features(features)
    scores = np.zeros(len(x), dtype=np.float32)
    if len(x) == 1:
        return scores
    norm = np.linalg.norm(x, axis=1)
    safe_norm = np.where(norm < 1e-8, 1.0, norm)
    unit = x / safe_norm[:, None]
    similarity = np.sum(unit[1:] * unit[:-1], axis=1)
    both_zero = (norm[1:] < 1e-8) & (norm[:-1] < 1e-8)
    similarity[both_zero] = 1.0
    scores[1:] = np.clip(1.0 - similarity, 0.0, 2.0)
    return scores


def utterance_zscore(features: np.ndarray) -> np.ndarray:
    """Paper-style per-dimension z-score used only by peak detection."""
    x = clean_features(features)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x - mean) / std


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if width <= 1 or len(x) <= 2:
        return x.copy()
    width = min(int(width), len(x))
    kernel = np.full(width, 1.0 / width, dtype=np.float32)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def dense_peak_prominence(features: np.ndarray, smoothing_width: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return the smoothed dissimilarity signal and a dense prominence score."""
    signal = moving_average(cosine_change(utterance_zscore(features)), smoothing_width)
    dense = np.zeros(len(signal), dtype=np.float32)
    candidates = np.arange(1, max(1, len(signal) - 1), dtype=np.int64)
    if len(candidates):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dense[candidates] = peak_prominences(signal, candidates)[0].astype(np.float32)
    return signal, dense


def peak_starts(
    features: np.ndarray,
    prominence: float,
    max_span: int,
    smoothing_width: int = 3,
) -> tuple[list[int], np.ndarray]:
    if prominence < 0 or not np.isfinite(prominence):
        raise ValueError("prominence must be finite and non-negative")
    signal, dense = dense_peak_prominence(features, smoothing_width)
    return peak_starts_from_signal(signal, prominence, max_span), dense


def peak_starts_from_signal(signal: np.ndarray, prominence: float, max_span: int) -> list[int]:
    signal = np.nan_to_num(np.asarray(signal, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if signal.ndim != 1 or not len(signal):
        raise ValueError("signal must be a non-empty vector")
    if prominence < 0 or not np.isfinite(prominence):
        raise ValueError("prominence must be finite and non-negative")
    peaks, _ = find_peaks(signal, prominence=float(prominence))
    selected = {int(index) for index in peaks if 0 < int(index) < len(signal)}
    starts = [0]
    for index in range(1, len(signal)):
        if index in selected or index - starts[-1] >= max_span:
            starts.append(index)
    return starts


def peak_budget_starts_from_signal(
    signal: np.ndarray,
    dense_prominence: np.ndarray,
    segments: int,
    max_span: int,
) -> tuple[list[int], float]:
    """Select an exact budget while prioritizing true local peaks.

    Strict local maxima cannot exceed half the input frame rate, so they cannot
    represent the 8 Hz condition on a 12.5 Hz grid. The smoothed change signal
    only breaks ties or backfills after all informative peaks are prioritized.
    """
    values = np.nan_to_num(np.asarray(signal, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    prominence = np.nan_to_num(
        np.asarray(dense_prominence, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    if values.ndim != 1 or prominence.shape != values.shape or not len(values):
        raise ValueError("signal and dense_prominence must be aligned non-empty vectors")
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
    starts, negative_score = minimum_cost_starts(costs, len(values), target, max_span)
    return starts, -negative_score


def no_merge_starts(length: int) -> list[int]:
    if length <= 0:
        raise ValueError("length must be positive")
    return list(range(length))


def uniform_starts(length: int, segments: int, max_span: int) -> list[int]:
    segments = feasible_segments(length, segments, max_span)
    base, remainder = divmod(length, segments)
    lengths = [base + (1 if idx < remainder else 0) for idx in range(segments)]
    starts = [0]
    for size in lengths[:-1]:
        starts.append(starts[-1] + size)
    return starts


def uniform_period_starts(times: np.ndarray, period_sec: float, max_span: int) -> list[int]:
    """Place boundaries on a fixed time clock without an utterance-level budget."""
    frame_times = np.asarray(times, dtype=np.float64).reshape(-1)
    if not len(frame_times) or not np.isfinite(frame_times).all():
        raise ValueError("times must be a finite non-empty vector")
    if period_sec <= 0 or not np.isfinite(period_sec):
        raise ValueError("period_sec must be finite and positive")
    if len(frame_times) == 1:
        return [0]
    boundaries = (frame_times[:-1] + frame_times[1:]) / 2.0
    left_edge = float(frame_times[0] - 0.5 * (frame_times[1] - frame_times[0]))
    next_time = left_edge + float(period_sec)
    starts = [0]
    for index, boundary_time in enumerate(boundaries, 1):
        due_to_clock = boundary_time >= next_time
        due_to_span = index - starts[-1] >= max_span
        if due_to_clock or due_to_span:
            starts.append(index)
            if next_time <= boundary_time:
                elapsed_periods = math.floor((boundary_time - next_time) / period_sec) + 1
                next_time += elapsed_periods * float(period_sec)
    return starts


def segment_sse_costs(features: np.ndarray, max_span: int) -> np.ndarray:
    x = clean_features(features).astype(np.float64)
    length = len(x)
    prefix = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(np.sum(x * x, axis=1))])
    costs = np.full((length, max_span + 1), np.inf, dtype=np.float64)
    for start in range(length):
        for size in range(1, min(max_span, length - start) + 1):
            end = start + size
            vector_sum = prefix[end] - prefix[start]
            costs[start, size] = max(0.0, prefix_sq[end] - prefix_sq[start] - float(vector_sum @ vector_sum) / size)
    return costs


def segment_l2_costs(features: np.ndarray, max_span: int) -> np.ndarray:
    """CodecSlime's sum of framewise L2 distances to the segment mean."""
    x = clean_features(features).astype(np.float64)
    length = len(x)
    prefix = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    costs = np.full((length, max_span + 1), np.inf, dtype=np.float64)
    for start in range(length):
        for size in range(1, min(max_span, length - start) + 1):
            end = start + size
            mean = (prefix[end] - prefix[start]) / size
            costs[start, size] = float(np.linalg.norm(x[start:end] - mean, axis=1).sum())
    return costs


def minimum_cost_starts(costs: np.ndarray, length: int, segments: int, max_span: int) -> tuple[list[int], float]:
    segments = feasible_segments(length, segments, max_span)
    dp = np.full((segments + 1, length + 1), np.inf, dtype=np.float64)
    parent = np.full((segments + 1, length + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    for group in range(1, segments + 1):
        for end in range(1, length + 1):
            for size in range(1, min(max_span, end) + 1):
                start = end - size
                previous = dp[group - 1, start]
                if not np.isfinite(previous):
                    continue
                candidate = previous + costs[start, size]
                if candidate < dp[group, end]:
                    dp[group, end] = candidate
                    parent[group, end] = start
    if not np.isfinite(dp[segments, length]):
        raise RuntimeError("no feasible minimum-cost segmentation")
    starts = []
    end = length
    for group in range(segments, 0, -1):
        start = int(parent[group, end])
        starts.append(start)
        end = start
    starts.reverse()
    return starts, float(dp[segments, length])


def penalized_minimum_cost_starts(
    costs: np.ndarray,
    length: int,
    max_span: int,
    segment_penalty: float,
) -> tuple[list[int], float]:
    """Global DP with a frozen per-segment penalty instead of exact K."""
    if segment_penalty < 0 or not np.isfinite(segment_penalty):
        raise ValueError("segment_penalty must be finite and non-negative")
    dp = np.full(length + 1, np.inf, dtype=np.float64)
    parent = np.full(length + 1, -1, dtype=np.int64)
    dp[0] = 0.0
    for end in range(1, length + 1):
        for size in range(1, min(max_span, end) + 1):
            start = end - size
            candidate = dp[start] + float(costs[start, size]) + float(segment_penalty)
            if candidate < dp[end]:
                dp[end] = candidate
                parent[end] = start
    if not np.isfinite(dp[length]):
        raise RuntimeError("no feasible penalized segmentation")
    starts = []
    end = length
    while end > 0:
        start = int(parent[end])
        if start < 0:
            raise RuntimeError("broken penalized-DP backtrace")
        starts.append(start)
        end = start
    starts.reverse()
    return starts, float(dp[length])


def split_gain_scores(features: np.ndarray, max_span: int) -> np.ndarray:
    x = clean_features(features)
    length = len(x)
    costs = segment_l2_costs(x, min(max_span * 2, max(1, length)))
    scores = np.zeros(length, dtype=np.float32)
    for boundary in range(1, length):
        left = max(0, boundary - max_span)
        right = min(length, boundary + max_span)
        whole_size = right - left
        if whole_size >= costs.shape[1]:
            whole = float(np.linalg.norm(x[left:right] - x[left:right].mean(axis=0), axis=1).sum())
        else:
            whole = costs[left, whole_size]
        left_cost = costs[left, boundary - left]
        right_cost = costs[boundary, right - boundary]
        scores[boundary] = max(0.0, float(whole - left_cost - right_cost))
    return scores


def elastic_time_greedy_starts(
    costs: np.ndarray,
    length: int,
    segments: int,
    max_span: int,
) -> tuple[list[int], float, bool]:
    """Elastic Time right-expansion greedy using incremental rollout error."""
    target = feasible_segments(length, segments, max_span)
    offsets = np.zeros(length, dtype=np.int64)
    errors = np.full(length, np.inf, dtype=np.float64)
    for index in range(1, length):
        if costs.shape[1] > 2 and np.isfinite(costs[index - 1, 2]):
            errors[index] = costs[index - 1, 2]

    for _ in range(length - target):
        selected = int(np.argmin(errors))
        if not np.isfinite(errors[selected]):
            starts, objective = minimum_cost_starts(costs, length, target, max_span)
            return starts, objective, True

        offsets[selected] = offsets[selected - 1] + 1
        errors[selected] = np.inf
        errors[selected - 1] = np.inf

        next_index = selected + 1
        if next_index >= length or not np.isfinite(errors[next_index]):
            continue
        anchor = selected - int(offsets[selected])
        current_size = int(offsets[selected]) + 1
        expanded_size = current_size + 1
        if expanded_size > max_span or expanded_size >= costs.shape[1]:
            errors[next_index] = np.inf
            continue
        errors[next_index] = max(0.0, float(costs[anchor, expanded_size] - costs[anchor, current_size]))

    starts = np.flatnonzero(offsets == 0).tolist()
    ends = [*starts[1:], length]
    objective = sum(costs[start, end - start] for start, end in zip(starts, ends))
    return starts, float(objective), False


def elastic_time_greedy_threshold_starts(
    costs: np.ndarray,
    length: int,
    max_span: int,
    difficulty_threshold: float,
) -> tuple[list[int], float]:
    """Elastic-Time greedy removal stopped by a frozen difficulty cutoff."""
    if difficulty_threshold < 0 or not np.isfinite(difficulty_threshold):
        raise ValueError("difficulty_threshold must be finite and non-negative")
    segments = [(index, index + 1) for index in range(length)]
    while True:
        candidates = []
        for index in range(len(segments) - 1):
            start, middle = segments[index]
            _, end = segments[index + 1]
            merged_size = end - start
            if merged_size > max_span or merged_size >= costs.shape[1]:
                continue
            left_size = middle - start
            right_size = end - middle
            merged = float(costs[start, merged_size])
            separate = float(costs[start, left_size] + costs[middle, right_size])
            if np.isfinite(merged) and np.isfinite(separate):
                candidates.append((max(0.0, merged - separate), index))
        if not candidates:
            break
        difficulty, selected = min(candidates)
        if difficulty > difficulty_threshold:
            break
        start, _ = segments[selected]
        _, end = segments[selected + 1]
        segments[selected : selected + 2] = [(start, end)]

    starts = [start for start, _ in segments]
    objective = sum(costs[start, end - start] for start, end in segments)
    return starts, float(objective)


greedy_cost_starts = elastic_time_greedy_starts


def select(
    name: str,
    features: np.ndarray,
    times: np.ndarray,
    target_segments: int | None,
    max_span: int = 8,
    elastic_model: ElasticPredictor | None = None,
    local_model: LocalPredictabilityModel | None = None,
    device: str = "cpu",
    precomputed: dict[str, np.ndarray] | None = None,
    control_parameter: float | None = None,
) -> SelectionResult:
    x = clean_features(features)
    length = len(x)
    frame_times = np.asarray(times, dtype=np.float64).reshape(-1)
    if frame_times.shape != (length,) or not np.isfinite(frame_times).all():
        raise ValueError("times must be a finite vector aligned with features")
    if length == 1:
        duration_sec = max(2.0 * float(frame_times[0]), 1e-12)
    else:
        left_step = float(frame_times[1] - frame_times[0])
        right_step = float(frame_times[-1] - frame_times[-2])
        duration_sec = max(float(frame_times[-1] + 0.5 * right_step - frame_times[0] + 0.5 * left_step), 1e-12)
    started = time.perf_counter()
    metadata: dict[str, object]
    if name == "no_merge":
        segments = length
        metadata = {"control_mode": "natural_rate"}
    elif name in DIRECT_RATE_SYSTEMS or name in {"elastic_time_greedy", "elastic_time_dp"}:
        if target_segments is None:
            raise ValueError(f"{name} requires target_segments")
        segments = feasible_segments(length, target_segments, max_span)
        metadata = {
            "control_mode": "direct_rate",
            "requested_segments": int(target_segments),
        }
    elif name in CALIBRATED_SYSTEMS or name == "dcdit_1d":
        if control_parameter is None or not np.isfinite(control_parameter):
            raise ValueError(f"{name} requires a finite validation-calibrated control_parameter")
        segments = None
        metadata = {
            "control_mode": "calibrated_parameter",
            "control_parameter": float(control_parameter),
        }
    else:
        raise KeyError(f"unknown selector: {name}")
    cached = precomputed or {}

    if name == "no_merge":
        metadata["implementation"] = "native_grid_no_merge_v1"
        scores = np.full(length, np.nan, dtype=np.float32)
        starts = no_merge_starts(length)
        objective = math.nan
    elif name == "uniform":
        metadata["implementation"] = "uniform_v1"
        scores = np.full(length, np.nan, dtype=np.float32)
        starts = uniform_starts(length, segments, max_span)
        objective = math.nan
    elif name == "similarity":
        metadata["implementation"] = "flexicodec_cosine_threshold_skip_normalize_v3"
        metadata["parameter_name"] = "similarity_threshold"
        scores = cached.get("cosine_scores", cosine_change(x))
        starts = threshold_starts(scores, 1.0 - float(control_parameter), max_span)
        objective = float(np.nansum(scores[starts[1:]]))
    elif name == "codecslime_dp":
        metadata["implementation"] = "codecslime_schedfr_l2_dp_v2"
        scores = split_gain_scores(x, max_span)
        starts, objective = minimum_cost_starts(segment_l2_costs(x, max_span), length, segments, max_span)
    elif name == "ple":
        metadata["implementation"] = "dtm_ple_fixed_tau_v2"
        metadata["parameter_name"] = "tau"
        scores = cached.get("cosine_scores", cosine_change(x))
        starts = ple_fixed_tau_scores(scores, float(control_parameter), max_span)
        objective = float(np.nansum(scores[starts[1:]]))
    elif name == "peak_detection":
        metadata["implementation"] = "word_segmentation_peak_exact_budget_v2"
        metadata["paper_faithful_signal"] = "utterance_zscore_cosine_change_moving_average_prominence"
        metadata["rate_control"] = "exact_k_peak_priority_with_signal_backfill"
        metadata["smoothing_width"] = 3
        signal = cached.get("peak_signal")
        scores = cached.get("peak_scores")
        if signal is None or scores is None:
            signal, scores = dense_peak_prominence(x, smoothing_width=3)
        starts, objective = peak_budget_starts_from_signal(signal, scores, segments, max_span)
    elif name in {"elastic_time_greedy", "elastic_time_dp"}:
        if elastic_model is None:
            raise ValueError(f"{name} requires an ElasticPredictor checkpoint")
        costs = cached.get("elastic_costs")
        if costs is None:
            costs = elastic_model.segment_costs(x, max_span, device)
        scores = np.zeros(length, dtype=np.float32)
        for idx in range(1, length):
            scores[idx] = float(costs[idx - 1, 2]) if np.isfinite(costs[idx - 1, 2]) else 0.0
        if name == "elastic_time_dp":
            metadata["implementation"] = "elastic_time_rollout_dp_v2"
            starts, objective = minimum_cost_starts(costs, length, segments, max_span)
        else:
            metadata["implementation"] = "elastic_time_right_expansion_greedy_v2"
            starts, objective, fallback = elastic_time_greedy_starts(costs, length, segments, max_span)
            metadata["dp_fallback"] = fallback
    elif name in {"dcdit_1d", "local_predictability"}:
        metadata["implementation"] = "dcdit_inspired_1d_masked_center_threshold_v2"
        metadata["parameter_name"] = "difficulty_threshold"
        if local_model is None:
            raise ValueError(f"{name} requires a LocalPredictabilityModel checkpoint")
        scores = cached.get("local_scores")
        if scores is None:
            scores = local_model.difficulty(x, device)
        starts = threshold_starts(scores, float(control_parameter), max_span)
        objective = float(np.sum(scores[starts[1:]]))
    result = result_from_starts(
        name,
        length,
        starts,
        scores,
        objective,
        (time.perf_counter() - started) * 1000.0,
        duration_sec,
        0,
        metadata,
    )
    result.metadata["actual_segments"] = len(result.starts)
    validate_result(result, length, segments, max_span)
    return result


def reconstructed_features(features: np.ndarray, result: SelectionResult) -> np.ndarray:
    x = clean_features(features)
    reconstructed = np.empty_like(x)
    for segment in range(len(result.starts)):
        mask = result.segment_ids == segment
        reconstructed[mask] = x[mask].mean(axis=0)
    return reconstructed


def normalized_mse(features: np.ndarray, result: SelectionResult) -> float:
    x = clean_features(features)
    reconstructed = reconstructed_features(x, result)
    denominator = float(np.mean(x * x))
    return float(np.mean((reconstructed - x) ** 2) / max(denominator, 1e-8))


def brute_force_minimum(costs: np.ndarray, length: int, segments: int, max_span: int) -> tuple[list[int], float]:
    """Reference implementation used only by tests."""
    best: tuple[list[int], float] | None = None
    for internal in combinations(range(1, length), segments - 1):
        starts = [0, *internal]
        ends = [*internal, length]
        lengths = [end - start for start, end in zip(starts, ends)]
        if max(lengths) > max_span:
            continue
        value = float(sum(costs[start, size] for start, size in zip(starts, lengths)))
        if best is None or value < best[1]:
            best = (starts, value)
    if best is None:
        raise RuntimeError("no feasible brute-force segmentation")
    return best


def ple_fixed_tau_scores(scores: np.ndarray, tau: float, max_span: int) -> list[int]:
    if tau <= 0:
        raise ValueError("tau must be positive")
    if max_span < 1:
        raise ValueError("max_span must be positive")
    starts = [0]
    cumulative = 0.0
    next_threshold = float(tau)
    values = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    for idx in range(1, len(values)):
        cumulative += max(0.0, float(values[idx]))
        crossed = cumulative >= next_threshold
        if crossed or idx - starts[-1] >= max_span:
            starts.append(idx)
            if crossed:
                next_threshold += float(tau)
    return starts


def ple_fixed_tau(features: np.ndarray, tau: float, max_span: int) -> list[int]:
    return ple_fixed_tau_scores(cosine_change(features), tau, max_span)
