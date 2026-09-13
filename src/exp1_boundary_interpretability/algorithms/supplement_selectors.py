"""Training-free selector adapters used by the boundary supplement.

TADPC follows QwenAudio/FunResearch commit
ff0c6901600595a0aab37b4d651ff7bde9f26b2e, specifically the corrected
``SequentialDPClusteringFixed`` equations. Its rate is controlled only by the
official global similarity threshold. A-ToMe uses fixed-ratio adjacent merges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Selection:
    starts: list[int]
    native_segments: int


@dataclass
class _Segment:
    start: int
    end: int
    mean: np.ndarray

    @property
    def length(self) -> int:
        return self.end - self.start


def _normalize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("features must be a non-empty [frames, dimensions] array")
    if not np.isfinite(values).all():
        raise ValueError("features contain non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, 1e-12))


def atome_style_starts(
    features: np.ndarray, merge_ratios: tuple[float, ...] = (0.5, 0.25)
) -> Selection:
    """Apply stacked fixed-ratio adjacent token merging modules.

    This is A-ToMe's stacked fixed-merge-ratio control adapted to frozen L49.
    Each module uses path matching to choose the requested number of
    non-overlapping adjacent pairs with maximum total cosine similarity.
    """
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("features must be a finite non-empty matrix")
    if not merge_ratios or any(not 0.0 <= ratio <= 0.5 for ratio in merge_ratios):
        raise ValueError("each A-ToMe merge ratio must be inside [0, 0.5]")
    segments = [_Segment(index, index + 1, value.copy()) for index, value in enumerate(values)]
    for merge_ratio in merge_ratios:
        segments = _atome_fixed_ratio_stage(segments, merge_ratio)
    starts = [segment.start for segment in segments]
    _validate_starts(starts, len(values), 2 ** len(merge_ratios))
    return Selection(starts=starts, native_segments=len(starts))


def _atome_fixed_ratio_stage(
    segments: list[_Segment], merge_ratio: float
) -> list[_Segment]:
    merges = min(int(np.floor(len(segments) * merge_ratio)), len(segments) // 2)
    if merges == 0:
        return list(segments)

    scores = np.asarray(
        [
            _cosine(segments[index].mean, segments[index + 1].mean)
            for index in range(len(segments) - 1)
        ],
        dtype=np.float64,
    )
    negative = -np.inf
    dp = np.full((len(segments) + 1, merges + 1), negative, dtype=np.float64)
    parent = np.zeros((len(segments) + 1, merges + 1), dtype=np.int8)
    dp[:, 0] = 0.0
    for consumed in range(1, len(segments) + 1):
        upper = min(merges, consumed // 2)
        for count in range(1, upper + 1):
            skip = dp[consumed - 1, count]
            take = (
                dp[consumed - 2, count - 1] + scores[consumed - 2]
                if consumed >= 2
                else negative
            )
            if take > skip + 1e-12:
                dp[consumed, count] = take
                parent[consumed, count] = 1
            else:
                dp[consumed, count] = skip
    if not np.isfinite(dp[len(segments), merges]):
        raise RuntimeError("A-ToMe fixed-ratio matching is infeasible")

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
            output.append(
                _Segment(left.start, right.end, (left.mean + right.mean) / 2.0)
            )
            index += 2
        else:
            output.append(segments[index])
            index += 1
    if len(output) != len(segments) - merges:
        raise RuntimeError("A-ToMe fixed-ratio output count mismatch")
    return output


def _tadpc_native_clusters(
    features: np.ndarray,
    *,
    k: int = 10,
    beta: float = 0.2,
    threshold: float = 0.7,
    max_span: int = 4,
) -> list[_Segment]:
    """Numpy transcription of SequentialDPClusteringFixed for one sequence."""
    values = np.asarray(features, dtype=np.float64)
    normalized = _normalize(values)
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
        delta[index] = (
            distance[index, higher].min()
            if np.any(higher)
            else distance[index].max()
        )
    seed_score = rho * delta
    adjusted = similarity - beta * seed_score[None, :]

    assigned = np.zeros(len(values), dtype=bool)
    clusters: list[list[int]] = []
    while not assigned.all():
        available = np.where(assigned, -np.inf, seed_score)
        seed = int(np.argmax(available))
        cluster = [seed]
        assigned[seed] = True
        for position in range(seed + 1, min(len(values), seed + max_span + 1)):
            if assigned[position] or len(cluster) >= max_span:
                break
            if adjusted[seed, position] > threshold:
                cluster.append(position)
                assigned[position] = True
            else:
                break
        for position in range(seed - 1, max(-1, seed - max_span - 1), -1):
            if assigned[position] or len(cluster) >= max_span:
                break
            if adjusted[seed, position] > threshold:
                cluster.insert(0, position)
                assigned[position] = True
            else:
                break
        clusters.append(cluster)

    clusters.sort(key=lambda cluster: cluster[0])
    output: list[_Segment] = []
    cursor = 0
    for cluster in clusters:
        if cluster != list(range(cluster[0], cluster[-1] + 1)):
            raise RuntimeError("official TADPC produced a non-contiguous cluster")
        if cluster[0] != cursor:
            raise RuntimeError("official TADPC did not partition the sequence")
        start, end = cluster[0], cluster[-1] + 1
        output.append(_Segment(start, end, values[start:end].mean(axis=0)))
        cursor = end
    if cursor != len(values):
        raise RuntimeError("official TADPC left frames unassigned")
    return output


def tadpc_style_starts(
    features: np.ndarray,
    *,
    k: int = 10,
    beta: float = 0.2,
    threshold: float = 0.7,
    tadpc_max_span: int = 4,
) -> Selection:
    """Run corrected TADPC with official threshold-based rate control."""
    values = np.asarray(features, dtype=np.float64)
    segments = _tadpc_native_clusters(
        values,
        k=k,
        beta=beta,
        threshold=threshold,
        max_span=tadpc_max_span,
    )
    starts = [segment.start for segment in segments]
    _validate_starts(starts, len(values), tadpc_max_span)
    return Selection(starts=starts, native_segments=len(starts))


def _validate_starts(starts: list[int], frames: int, max_span: int) -> None:
    if not starts or starts[0] != 0:
        raise RuntimeError("starts must begin at zero")
    if any(left >= right for left, right in zip(starts, starts[1:])):
        raise RuntimeError("starts must be strictly increasing")
    if starts[-1] >= frames:
        raise RuntimeError("start exceeds the native frame grid")
    lengths = np.diff(np.asarray([*starts, frames], dtype=np.int64))
    if np.any(lengths <= 0) or int(lengths.max()) > max_span:
        raise RuntimeError("selector violated its maximum-span constraint")
