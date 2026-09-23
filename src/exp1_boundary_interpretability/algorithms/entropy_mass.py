"""Waveform-entropy allocation used by the paper's Entropy-Mass baseline.

The temporal-entropy criterion is inspired by TFC, but this adapter is not a
reimplementation of TFC's learned multi-resolution codec. It measures soft
histogram entropy inside each native codec frame and uses exact-budget dynamic
programming to form contiguous spans with approximately equal entropy mass.
"""

from __future__ import annotations

import math

import numpy as np


SYSTEM = "tfc_style_entropy"
IMPLEMENTATION = "tfc_style_waveform_entropy_equal_mass_dp_v1"
MAX_SEGMENT_SECONDS = 0.640


def frame_edges(frame_times: np.ndarray, duration_sec: float) -> np.ndarray:
    times = np.asarray(frame_times, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.all(np.diff(times) > 0):
        raise ValueError("frame times must be a non-empty increasing vector")
    edges = np.empty(len(times) + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = float(duration_sec)
    if len(times) > 1:
        edges[1:-1] = (times[:-1] + times[1:]) / 2.0
    if not np.all(np.diff(edges) > 0):
        raise ValueError("frame supports are not strictly increasing")
    return edges


def waveform_frame_entropy(
    waveform: np.ndarray,
    sample_rate: int,
    frame_times: np.ndarray,
    duration_sec: float,
    bins: int = 64,
    sigma: float | None = None,
) -> np.ndarray:
    if bins < 2:
        raise ValueError("entropy bins must be at least two")
    sigma = 2.0 / (bins - 1) if sigma is None else float(sigma)
    if sigma <= 0:
        raise ValueError("entropy sigma must be positive")
    samples = np.asarray(waveform, dtype=np.float64)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or not len(samples):
        raise ValueError("waveform must contain at least one sample")
    samples = np.clip(np.nan_to_num(samples), -1.0, 1.0)
    centers = np.linspace(-1.0, 1.0, bins, dtype=np.float64)
    edges = frame_edges(frame_times, duration_sec)
    sample_edges = np.rint(edges * sample_rate).astype(np.int64)
    sample_edges = np.clip(sample_edges, 0, len(samples))
    sample_edges[0] = 0
    sample_edges[-1] = len(samples)

    entropy = np.empty(len(frame_times), dtype=np.float64)
    for index, (left, right) in enumerate(zip(sample_edges[:-1], sample_edges[1:])):
        if right <= left:
            midpoint = min(len(samples) - 1, max(0, int(left)))
            frame = samples[midpoint : midpoint + 1]
        else:
            frame = samples[left:right]
        affinity = np.exp(
            -0.5 * ((frame[:, None] - centers[None, :]) / sigma) ** 2
        )
        probability = affinity.mean(axis=0)
        probability /= max(float(probability.sum()), 1e-12)
        positive = probability > 0
        entropy[index] = -np.sum(
            probability[positive] * np.log2(probability[positive])
        )
    if not np.isfinite(entropy).all() or np.any(entropy <= 0):
        raise RuntimeError("non-finite waveform entropy")
    return entropy


def feasible_segments(frames: int, requested: int, max_span: int) -> int:
    if frames < 1 or max_span < 1:
        raise ValueError("frames and max_span must be positive")
    minimum = int(math.ceil(frames / max_span))
    return min(frames, max(minimum, int(requested)))


def equal_entropy_starts(
    entropy: np.ndarray,
    segments: int,
    max_span: int,
) -> list[int]:
    """Return the exact-K partition minimizing entropy-mass imbalance."""
    values = np.asarray(entropy, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("entropy must be a finite non-empty vector")
    frames = len(values)
    segments = feasible_segments(frames, segments, max_span)
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    target_mass = float(prefix[-1] / segments)
    dynamic = np.full((segments + 1, frames + 1), np.inf, dtype=np.float64)
    parent = np.full((segments + 1, frames + 1), -1, dtype=np.int64)
    dynamic[0, 0] = 0.0

    for group in range(1, segments + 1):
        for end in range(group, min(frames, group * max_span) + 1):
            minimum_start = max(group - 1, end - max_span)
            for start in range(minimum_start, end):
                previous = dynamic[group - 1, start]
                if not np.isfinite(previous):
                    continue
                mass = prefix[end] - prefix[start]
                candidate = previous + (mass - target_mass) ** 2
                if candidate < dynamic[group, end]:
                    dynamic[group, end] = candidate
                    parent[group, end] = start

    if not np.isfinite(dynamic[segments, frames]):
        raise RuntimeError("no feasible exact-budget entropy partition")
    starts: list[int] = []
    end = frames
    for group in range(segments, 0, -1):
        start = int(parent[group, end])
        if start < 0:
            raise RuntimeError("invalid dynamic-program backtrace")
        starts.append(start)
        end = start
    starts.reverse()
    return starts


def entropy_mass_starts(
    waveform: np.ndarray,
    sample_rate: int,
    frame_times: np.ndarray,
    duration_sec: float,
    target_rate_hz: float,
    *,
    bins: int = 64,
    sigma: float | None = None,
    max_segment_seconds: float = MAX_SEGMENT_SECONDS,
) -> tuple[list[int], np.ndarray]:
    """Compute Entropy-Mass starts and the per-frame entropy trajectory."""
    if duration_sec <= 0 or target_rate_hz <= 0:
        raise ValueError("duration and target rate must be positive")
    entropy = waveform_frame_entropy(
        waveform, sample_rate, frame_times, duration_sec, bins, sigma
    )
    native_rate = len(frame_times) / duration_sec
    max_span = max(1, int(math.floor(native_rate * max_segment_seconds + 1e-12)))
    requested = max(1, int(round(duration_sec * target_rate_hz)))
    return equal_entropy_starts(entropy, requested, max_span), entropy
