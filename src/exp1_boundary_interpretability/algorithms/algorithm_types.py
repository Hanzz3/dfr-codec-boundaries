from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SelectionResult:
    """A complete temporal segmentation returned by a DFR algorithm."""

    name: str
    keep_mask: np.ndarray
    segment_ids: np.ndarray
    boundary_scores: np.ndarray
    objective: float
    runtime_ms: float
    realized_rate_hz: float = float("nan")
    memory_bytes: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def starts(self) -> np.ndarray:
        return np.flatnonzero(self.keep_mask)

    @property
    def segment_lengths(self) -> np.ndarray:
        return np.bincount(self.segment_ids, minlength=len(self.starts))


def feasible_segments(length: int, requested: int, max_span: int) -> int:
    if length <= 0:
        raise ValueError("feature sequence must contain at least one frame")
    if max_span <= 0:
        raise ValueError("max_span must be positive")
    minimum = (length + max_span - 1) // max_span
    return max(minimum, min(int(requested), length))


def result_from_starts(
    name: str,
    length: int,
    starts: list[int] | np.ndarray,
    scores: np.ndarray,
    objective: float,
    runtime_ms: float,
    duration_sec: float | None = None,
    memory_bytes: int = 0,
    metadata: dict[str, object] | None = None,
) -> SelectionResult:
    starts_arr = np.asarray(starts, dtype=np.int64)
    if starts_arr.ndim != 1 or not len(starts_arr) or starts_arr[0] != 0:
        raise ValueError("segment starts must be a one-dimensional array beginning at zero")
    if np.any(np.diff(starts_arr) <= 0) or starts_arr[-1] >= length:
        raise ValueError("segment starts must be strictly increasing and inside the sequence")
    keep = np.zeros(length, dtype=bool)
    keep[starts_arr] = True
    segment_ids = np.cumsum(keep.astype(np.int64)) - 1
    dense_scores = np.asarray(scores, dtype=np.float32)
    if dense_scores.shape != (length,):
        raise ValueError(f"boundary_scores must have shape ({length},), got {dense_scores.shape}")
    return SelectionResult(
        name=name,
        keep_mask=keep,
        segment_ids=segment_ids,
        boundary_scores=dense_scores,
        objective=float(objective),
        runtime_ms=float(runtime_ms),
        realized_rate_hz=(len(starts_arr) / duration_sec if duration_sec and duration_sec > 0 else float("nan")),
        memory_bytes=int(memory_bytes),
        metadata=metadata or {},
    )


def validate_result(result: SelectionResult, length: int, segments: int | None, max_span: int) -> None:
    if result.keep_mask.shape != (length,) or result.segment_ids.shape != (length,):
        raise AssertionError("selector output length mismatch")
    if not result.keep_mask[0] or result.segment_ids[0] != 0:
        raise AssertionError("the first frame must start segment zero")
    if segments is not None and len(result.starts) != segments:
        raise AssertionError(f"expected {segments} segments, found {len(result.starts)}")
    actual_segments = len(result.starts)
    if not np.array_equal(np.unique(result.segment_ids), np.arange(actual_segments)):
        raise AssertionError("segment ids are not contiguous")
    if np.any(result.segment_lengths <= 0) or int(result.segment_lengths.max()) > max_span:
        raise AssertionError("invalid segment length")
