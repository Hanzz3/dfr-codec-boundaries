"""HuBERT-style offline K-means fitting, assignment, and repetition collapse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from algorithm_types import SelectionResult, result_from_starts, validate_result
from merging_selectors import clean_features
from rate_control import threshold_starts


PRIMARY_KMEANS_CLUSTERS = 500


@dataclass(frozen=True)
class KMeansCodebook:
    centers: np.ndarray
    random_state: int


def load_codebook(path, random_state: int = 42) -> KMeansCodebook:  # noqa: ANN001
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("architecture") != "hubert_style_lloyd_kmeans_v1":
        raise RuntimeError(f"unsupported K-means checkpoint architecture: {payload.get('architecture')}")
    centers = payload["centers"].float().numpy()
    if centers.ndim != 2 or not np.isfinite(centers).all():
        raise RuntimeError(f"invalid K-means centers in {path}")
    return KMeansCodebook(centers=centers, random_state=random_state)


def fit_codebook(
    feature_sequences: list[np.ndarray],
    clusters: int,
    *,
    random_state: int = 42,
    batch_size: int = 4096,
) -> KMeansCodebook:
    """Fit the optional codebook interface; the paper evaluation does not invoke it."""
    if clusters < 1:
        raise ValueError("clusters must be positive")
    if not feature_sequences:
        raise ValueError("feature_sequences cannot be empty")
    frames = np.concatenate([clean_features(item) for item in feature_sequences], axis=0)
    if len(frames) < clusters:
        raise ValueError("number of frames must be at least the number of clusters")
    model = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=random_state,
        batch_size=min(batch_size, len(frames)),
        n_init="auto",
    )
    model.fit(frames)
    return KMeansCodebook(model.cluster_centers_.astype(np.float32), random_state)


def assign_codes(features: np.ndarray, codebook: KMeansCodebook) -> np.ndarray:
    x = clean_features(features)
    centers = clean_features(codebook.centers)
    distances = (
        np.sum(x * x, axis=1, keepdims=True)
        - 2.0 * x @ centers.T
        + np.sum(centers * centers, axis=1)[None, :]
    )
    return np.argmin(distances, axis=1).astype(np.int64)


def code_change_scores(features: np.ndarray, codebook: KMeansCodebook) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest codes and confidence for each discrete code transition.

    A transition score is the squared-distance improvement of the new nearest
    centroid over the preceding frame's centroid. Non-transition positions are
    NaN so validation threshold calibration cannot invent boundaries that are
    not supported by a K-means token change.
    """
    x = clean_features(features)
    centers = clean_features(codebook.centers)
    distances = (
        np.sum(x * x, axis=1, keepdims=True)
        - 2.0 * x @ centers.T
        + np.sum(centers * centers, axis=1)[None, :]
    )
    codes = np.argmin(distances, axis=1).astype(np.int64)
    scores = np.full(len(x), np.nan, dtype=np.float32)
    if len(x) > 1:
        positions = np.flatnonzero(codes[1:] != codes[:-1]) + 1
        previous = codes[positions - 1]
        current = codes[positions]
        confidence = distances[positions, previous] - distances[positions, current]
        scores[positions] = np.maximum(confidence, 0.0).astype(np.float32)
    return codes, scores


def collapse_repetitions(codes: np.ndarray, max_span: int = 8) -> list[int]:
    values = np.asarray(codes, dtype=np.int64).reshape(-1)
    if not len(values):
        raise ValueError("codes cannot be empty")
    if max_span < 1:
        raise ValueError("max_span must be positive")
    starts = [0]
    for index in range(1, len(values)):
        if values[index] != values[index - 1] or index - starts[-1] >= max_span:
            starts.append(index)
    return starts


def select_kmeans(
    features: np.ndarray,
    times: np.ndarray,
    codebook: KMeansCodebook,
    max_span: int = 8,
    control_parameter: float | None = None,
) -> SelectionResult:
    x = clean_features(features)
    times = np.asarray(times, dtype=np.float64)
    if times.shape != (len(x),):
        raise ValueError("times must align with features")
    codes, scores = code_change_scores(x, codebook)
    starts = (
        collapse_repetitions(codes, max_span)
        if control_parameter is None
        else threshold_starts(scores, float(control_parameter), max_span)
    )
    quantization_error = float(np.mean((x - codebook.centers[codes]) ** 2))
    duration = 2.0 * float(times[0]) if len(times) == 1 else float(times[-1] - times[0] + (times[1] - times[0]))
    result = result_from_starts(
        "hubert_kmeans",
        len(x),
        starts,
        scores,
        quantization_error,
        0.0,
        duration,
        0,
        {
            "control_mode": "natural_rate" if control_parameter is None else "validation_threshold",
            "implementation": "hubert_kmeans_repetition_collapse_confidence_threshold_v2",
            "training_status": "trained_on_full_libritts_train",
            "clusters": int(len(codebook.centers)),
            "parameter_name": "code_change_confidence_threshold",
            "parameter_value": control_parameter,
        },
    )
    validate_result(result, len(x), None, max_span)
    return result
