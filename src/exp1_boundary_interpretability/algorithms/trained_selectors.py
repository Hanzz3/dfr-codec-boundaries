"""Native-grid trained merger adapters for direct semantic-token ASR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common_selectors import threshold_starts
from deferred_models import load_codebooks, load_normalization, load_predictors
from followup_selectors import Selection, requested_segments
from kmeans_selector import PRIMARY_KMEANS_CLUSTERS, code_change_scores
from merging_selectors import minimum_cost_starts, elastic_time_greedy_starts


BASE_SYSTEMS = (
    "uniform",
    "similarity",
    "codecslime_dp",
    "ple",
    "peak_detection",
)
TRAINED_SYSTEMS = (
    "elastic_time_greedy",
    "elastic_time_dp",
    "dcdit_1d",
    "hubert_kmeans",
)
MAIN_SYSTEMS = BASE_SYSTEMS + TRAINED_SYSTEMS


@dataclass(frozen=True)
class TrainedArtifacts:
    elastic: object
    dcdit: object
    kmeans: object
    mean: np.ndarray
    std: np.ndarray
    device: str


def load_trained_artifacts(
    predictor_dir: Path,
    normalization_path: Path,
    kmeans_dir: Path,
    device: str,
    feature_dim: int,
) -> TrainedArtifacts:
    elastic, dcdit = load_predictors(predictor_dir, device)
    codebooks = load_codebooks(kmeans_dir)
    if PRIMARY_KMEANS_CLUSTERS not in codebooks:
        raise RuntimeError(
            f"missing formal K-means K={PRIMARY_KMEANS_CLUSTERS} checkpoint"
        )
    mean, std = load_normalization(normalization_path, feature_dim)
    return TrainedArtifacts(
        elastic=elastic,
        dcdit=dcdit,
        kmeans=codebooks[PRIMARY_KMEANS_CLUSTERS],
        mean=mean,
        std=std,
        device=device,
    )


def normalize_features(features: np.ndarray, artifacts: TrainedArtifacts) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1:] != artifacts.mean.shape:
        raise ValueError(
            f"feature shape {values.shape} is incompatible with "
            f"normalization {artifacts.mean.shape}"
        )
    return (values - artifacts.mean) / artifacts.std


def frame_center_times(frames: int, duration_sec: float) -> np.ndarray:
    if frames < 1 or duration_sec <= 0:
        raise ValueError("frames and duration must be positive")
    return (
        (np.arange(frames, dtype=np.float64) + 0.5)
        * float(duration_sec)
        / frames
    )


def select_trained(
    system: str,
    normalized: np.ndarray,
    duration_sec: float,
    target_rate_hz: float,
    max_span: int,
    controls: dict,
    artifacts: TrainedArtifacts,
    cache: dict[str, np.ndarray],
) -> Selection:
    """Select one trained merger while sharing model scores across rates."""
    frames = len(normalized)
    segments = requested_segments(duration_sec, target_rate_hz)
    if system in {"elastic_time_greedy", "elastic_time_dp"}:
        costs = cache.get("elastic_costs")
        if costs is None:
            costs = artifacts.elastic.segment_costs(
                normalized, max_span, artifacts.device
            )
            cache["elastic_costs"] = costs
        dense = np.zeros(frames, dtype=np.float32)
        if frames > 1 and costs.shape[1] > 2:
            finite = np.isfinite(costs[:-1, 2])
            positions = np.flatnonzero(finite) + 1
            dense[positions] = costs[:-1, 2][finite].astype(np.float32)
        if system == "elastic_time_dp":
            starts, _ = minimum_cost_starts(
                costs, frames, segments, max_span
            )
        else:
            starts, _objective, _fallback = elastic_time_greedy_starts(
                costs, frames, segments, max_span
            )
        return Selection(starts=list(starts), dense_score=dense)

    if system == "dcdit_1d":
        scores = cache.get("dcdit_scores")
        if scores is None:
            scores = artifacts.dcdit.difficulty(normalized, artifacts.device)
            cache["dcdit_scores"] = scores
        starts = threshold_starts(
            scores, float(controls["dcdit_1d"]["value"]), max_span
        )
        return Selection(starts=list(starts), dense_score=scores)

    if system == "hubert_kmeans":
        scores = cache.get("kmeans_scores")
        if scores is None:
            _codes, scores = code_change_scores(normalized, artifacts.kmeans)
            cache["kmeans_scores"] = scores
        starts = threshold_starts(
            scores, float(controls["hubert_kmeans"]["value"]), max_span
        )
        return Selection(starts=list(starts), dense_score=scores)

    raise KeyError(f"unsupported trained merger: {system}")
