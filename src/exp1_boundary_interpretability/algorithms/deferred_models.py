"""Load and validate deferred predictor and K-means checkpoints."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from kmeans_selector import KMeansCodebook, load_codebook
from models import ElasticPredictor, LocalPredictabilityModel


def load_normalization(path: Path, feature_dim: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        mean = payload["mean"].astype(np.float32)
        std = payload["std"].astype(np.float32)
    if mean.ndim != 1 or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError(f"invalid normalization statistics: {path}")
    if feature_dim is not None and mean.shape != (feature_dim,):
        raise RuntimeError(f"normalization dimension {mean.shape} != {(feature_dim,)}")
    return mean, np.maximum(std, 1e-6)


def load_predictors(checkpoint_dir: Path, device: str):
    elastic_data = torch.load(checkpoint_dir / "elastic_predictor.pt", map_location="cpu", weights_only=True)
    if elastic_data.get("architecture") != "elastic_time_grucell_swiglu_v2":
        raise RuntimeError("Elastic Time checkpoint has the wrong architecture")
    elastic = ElasticPredictor(
        int(elastic_data["feature_dim"]),
        int(elastic_data["hidden_dim"]),
        int(elastic_data["num_layers"]),
    )
    elastic.load_state_dict(elastic_data["state_dict"])

    dcdit_data = torch.load(checkpoint_dir / "dcdit_1d_predictor.pt", map_location="cpu", weights_only=True)
    if dcdit_data.get("architecture") != "dcdit_1d_masked_center_v1":
        raise RuntimeError("DC-DiT-inspired checkpoint has the wrong architecture")
    dcdit = LocalPredictabilityModel(
        int(dcdit_data["feature_dim"]),
        int(dcdit_data["hidden_dim"]),
        int(dcdit_data["radius"]),
    )
    dcdit.load_state_dict(dcdit_data["state_dict"])
    return elastic.to(device).eval(), dcdit.to(device).eval()


def load_codebooks(checkpoint_dir: Path) -> dict[int, KMeansCodebook]:
    output = {}
    for path in sorted(checkpoint_dir.glob("kmeans_*.pt")):
        match = re.fullmatch(r"kmeans_(\d+)", path.stem)
        if match:
            output[int(match.group(1))] = load_codebook(path)
    if not output:
        raise RuntimeError(f"no trained K-means checkpoints under {checkpoint_dir}")
    return output
