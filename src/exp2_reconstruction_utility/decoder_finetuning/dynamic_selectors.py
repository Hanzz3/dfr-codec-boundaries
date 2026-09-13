#!/usr/bin/env python3
"""Training-free selectors evaluated directly on FlexiCodec semantic frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Segment:
    start: int
    end: int
    mean: np.ndarray


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, 1e-12))


def atome_style_starts(features: np.ndarray, merge_ratio: float = 0.5) -> list[int]:
    """Run one official-style fixed-ratio adjacent matching stage."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("features must be a finite non-empty matrix")
    segments = [_Segment(i, i + 1, value.copy()) for i, value in enumerate(values)]
    merges = min(int(np.floor(len(segments) * merge_ratio)), len(segments) // 2)
    if merges == 0:
        return [0]

    scores = np.asarray(
        [_cosine(segments[i].mean, segments[i + 1].mean) for i in range(len(segments) - 1)]
    )
    dp = np.full((len(segments) + 1, merges + 1), -np.inf, dtype=np.float64)
    parent = np.zeros((len(segments) + 1, merges + 1), dtype=np.int8)
    dp[:, 0] = 0.0
    for consumed in range(1, len(segments) + 1):
        for count in range(1, min(merges, consumed // 2) + 1):
            skip = dp[consumed - 1, count]
            take = dp[consumed - 2, count - 1] + scores[consumed - 2]
            if take > skip + 1e-12:
                dp[consumed, count] = take
                parent[consumed, count] = 1
            else:
                dp[consumed, count] = skip
    if not np.isfinite(dp[len(segments), merges]):
        raise RuntimeError("A-ToMe matching is infeasible")

    pairs: set[int] = set()
    consumed, count = len(segments), merges
    while count:
        if parent[consumed, count]:
            pairs.add(consumed - 2)
            consumed -= 2
            count -= 1
        else:
            consumed -= 1
    output: list[_Segment] = []
    index = 0
    while index < len(segments):
        if index in pairs:
            left, right = segments[index], segments[index + 1]
            output.append(_Segment(left.start, right.end, (left.mean + right.mean) / 2.0))
            index += 2
        else:
            output.append(segments[index])
            index += 1
    starts = [segment.start for segment in output]
    _validate(starts, len(values), max_span=2)
    return starts


def tadpc_style_starts(
    features: np.ndarray,
    *,
    k: int = 10,
    beta: float = 0.2,
    threshold: float = 0.8561496734619141,
    max_span: int = 4,
) -> list[int]:
    """Run corrected official SequentialDPClusteringFixed equations."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("features must be a finite non-empty matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    similarity = (normalized @ normalized.T + 1.0) / 2.0
    neighbors = min(k, max(len(values) - 1, 0))
    if neighbors:
        ordered = np.sort(similarity, axis=1)[:, ::-1]
        rho = np.exp(ordered[:, 1 : neighbors + 1].mean(axis=1))
    else:
        rho = np.ones(1, dtype=np.float64)

    distance = 1.0 - similarity
    delta = np.empty(len(values), dtype=np.float64)
    for index in range(len(values)):
        higher = rho > rho[index]
        delta[index] = distance[index, higher].min() if np.any(higher) else distance[index].max()
    seed_score = rho * delta
    adjusted = similarity - beta * seed_score[None, :]
    assigned = np.zeros(len(values), dtype=bool)
    clusters: list[list[int]] = []
    while not assigned.all():
        seed = int(np.argmax(np.where(assigned, -np.inf, seed_score)))
        cluster = [seed]
        assigned[seed] = True
        for position in range(seed + 1, min(len(values), seed + max_span + 1)):
            if assigned[position] or len(cluster) >= max_span:
                break
            if adjusted[seed, position] <= threshold:
                break
            cluster.append(position)
            assigned[position] = True
        for position in range(seed - 1, max(-1, seed - max_span - 1), -1):
            if assigned[position] or len(cluster) >= max_span:
                break
            if adjusted[seed, position] <= threshold:
                break
            cluster.insert(0, position)
            assigned[position] = True
        clusters.append(cluster)

    clusters.sort(key=lambda cluster: cluster[0])
    cursor = 0
    starts: list[int] = []
    for cluster in clusters:
        if cluster != list(range(cluster[0], cluster[-1] + 1)) or cluster[0] != cursor:
            raise RuntimeError("TADPC did not produce a contiguous partition")
        starts.append(cluster[0])
        cursor = cluster[-1] + 1
    if cursor != len(values):
        raise RuntimeError("TADPC left frames unassigned")
    _validate(starts, len(values), max_span=max_span)
    return starts


def _validate(starts: list[int], frames: int, max_span: int) -> None:
    if not starts or starts[0] != 0 or any(a >= b for a, b in zip(starts, starts[1:])):
        raise RuntimeError("invalid selector starts")
    lengths = np.diff(np.asarray([*starts, frames], dtype=np.int64))
    if np.any(lengths <= 0) or int(lengths.max()) > max_span:
        raise RuntimeError("selector violated its span constraint")
