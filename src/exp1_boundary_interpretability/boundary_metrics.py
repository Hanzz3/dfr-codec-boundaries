#!/usr/bin/env python3
"""Maximum-cardinality matching for EXP1 reference boundaries."""

from __future__ import annotations

from typing import Iterable

import numpy as np


PRIMARY_COLLAR_MS = 40.0
BOUNDARY_MATCHING_PROTOCOL = "maximum_cardinality_monotonic_v2"


def finite_sorted(values: Iterable[float], duration_sec: float | None = None) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if duration_sec is not None:
        array = array[(array >= 0.0) & (array <= duration_sec)]
    return np.sort(array)


def match_boundaries(
    predicted: Iterable[float],
    reference: Iterable[float],
    tolerance_sec: float,
) -> tuple[int, int, int, list[float]]:
    """Maximum-cardinality monotonic matching for sorted point events."""
    if tolerance_sec < 0.0:
        raise ValueError("tolerance_sec must be non-negative")
    pred = finite_sorted(predicted)
    ref = finite_sorted(reference)
    offsets: list[float] = []
    pred_index = 0
    ref_index = 0
    while pred_index < len(pred) and ref_index < len(ref):
        delta = float(pred[pred_index] - ref[ref_index])
        if delta < -tolerance_sec:
            pred_index += 1
        elif delta > tolerance_sec:
            ref_index += 1
        else:
            offsets.append(delta)
            pred_index += 1
            ref_index += 1
    tp = len(offsets)
    return tp, len(pred) - tp, len(ref) - tp, offsets
