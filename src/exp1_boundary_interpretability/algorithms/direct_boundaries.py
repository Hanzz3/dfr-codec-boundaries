"""Convert timestamp boundaries into valid FlexiCodec frame segmentations."""

from __future__ import annotations

import time

import numpy as np

from algorithm_types import SelectionResult, result_from_starts, validate_result


def frame_boundary_times(frame_times: np.ndarray) -> np.ndarray:
    times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("frame_times must be finite and strictly increasing")
    if len(times) == 1:
        return np.asarray([0.0, 2.0 * times[0]], dtype=np.float64)
    internal = (times[:-1] + times[1:]) / 2.0
    first = max(0.0, float(times[0] - 0.5 * (times[1] - times[0])))
    last = float(times[-1] + 0.5 * (times[-1] - times[-2]))
    return np.concatenate(([first], internal, [last]))


def snap_boundaries_to_starts(
    boundaries_sec: list[float] | np.ndarray,
    frame_times: np.ndarray,
    max_span: int = 8,
) -> tuple[list[int], int, int]:
    edges = frame_boundary_times(frame_times)
    length = len(edges) - 1
    values = np.asarray(boundaries_sec, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    values = values[(values > edges[0]) & (values < edges[-1])]
    snapped = {
        int(np.argmin(np.abs(edges[1:-1] - value))) + 1
        for value in values
    } if length > 1 else set()
    source_count = len(snapped)
    starts = [0]
    for candidate in sorted(snapped):
        while candidate - starts[-1] > max_span:
            starts.append(starts[-1] + max_span)
        if candidate > starts[-1]:
            starts.append(candidate)
    while length - starts[-1] > max_span:
        starts.append(starts[-1] + max_span)
    return starts, source_count, len(starts) - 1 - source_count


def select_direct_boundaries(
    name: str,
    features: np.ndarray,
    times: np.ndarray,
    boundaries_sec: list[float] | np.ndarray,
    max_span: int = 8,
) -> SelectionResult:
    x = np.asarray(features)
    if x.ndim != 2 or not len(x):
        raise ValueError("features must have shape [T, D]")
    started = time.perf_counter()
    starts, source_count, technical_count = snap_boundaries_to_starts(boundaries_sec, times, max_span)
    edges = frame_boundary_times(times)
    scores = np.full(len(x), np.nan, dtype=np.float32)
    result = result_from_starts(
        name,
        len(x),
        starts,
        scores,
        float("nan"),
        (time.perf_counter() - started) * 1000.0,
        float(edges[-1] - edges[0]),
        0,
        {
            "control_mode": "direct_natural_rate",
            "implementation": "timestamp_snap_collision_dedup_maxspan_v1",
            "source_boundaries_after_collision": source_count,
            "technical_maxspan_boundaries": technical_count,
        },
    )
    validate_result(result, len(x), None, max_span)
    return result
