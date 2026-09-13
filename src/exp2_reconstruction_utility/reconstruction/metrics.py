from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


SYSTEMS = (
    "uniform",
    "flexicodec_threshold",
    "codecslime_dp",
    "ple",
    "elastic_time_greedy",
    "elastic_time_dp",
    "dcdit_1d",
    "atome_style",
    "tadpc_style",
)
TARGETS = (
    "phoneme",
    "syllable",
    "bpe_subword",
    "word",
    "acoustic_event",
    "vuv",
)
LINGUISTIC = ("syllable", "bpe_subword", "word")
ACOUSTIC = ("phoneme", "acoustic_event", "vuv")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature(row: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(row["feature_path"]) as payload:
        features = payload["features"].astype(np.float32)
        times = payload["times"].astype(np.float32)
    if (
        features.ndim != 2
        or times.shape != (len(features),)
        or not len(features)
        or not np.isfinite(features).all()
        or not np.isfinite(times).all()
    ):
        raise RuntimeError(f"invalid cached feature row: {row['utt_id']}")
    return features, times


def select_starts(
    system: str,
    features: np.ndarray,
    *,
    atome_ratio: float,
    tadpc_threshold: float,
) -> list[int]:
    from supplement_selectors import atome_style_starts, tadpc_style_starts

    if system == "atome_style":
        return atome_style_starts(features, (atome_ratio,)).starts
    if system == "tadpc_style":
        return tadpc_style_starts(features, threshold=tadpc_threshold).starts
    raise KeyError(system)


def boundary_times(starts: list[int], frame_times: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (float(frame_times[start - 1]) + float(frame_times[start])) / 2.0
            for start in starts[1:]
        ],
        dtype=np.float32,
    )


def segment_lengths(starts: list[int], frames: int) -> np.ndarray:
    return np.diff(np.asarray([*starts, frames], dtype=np.int64))


def feature_nmse(features: np.ndarray, starts: list[int]) -> float:
    reconstructed = np.empty_like(features)
    ends = [*starts[1:], len(features)]
    for start, end in zip(starts, ends):
        reconstructed[start:end] = features[start:end].mean(axis=0)
    denominator = float(np.mean(features * features))
    return float(
        np.mean((reconstructed - features) ** 2) / max(denominator, 1e-8)
    )


def match_boundaries(
    predicted: np.ndarray, reference: list[float], tolerance: float
) -> tuple[int, int, int, list[float]]:
    pred = np.sort(np.asarray(predicted, dtype=np.float64).reshape(-1))
    ref = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    offsets: list[float] = []
    pred_index = 0
    ref_index = 0
    while pred_index < len(pred) and ref_index < len(ref):
        delta = float(pred[pred_index] - ref[ref_index])
        if delta < -tolerance:
            pred_index += 1
        elif delta > tolerance:
            ref_index += 1
        else:
            offsets.append(delta)
            pred_index += 1
            ref_index += 1
    tp = len(offsets)
    return tp, len(pred) - tp, len(ref) - tp, offsets


def normalize_words(value: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", value.lower()).split()


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for index, lhs in enumerate(left, 1):
        current = [index]
        for column, rhs in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (lhs != rhs),
                )
            )
        previous = current
    return previous[-1]


def word_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    return edit_distance(ref, hyp), len(ref)


def implementation(system: str) -> str:
    names = {
        "uniform": "uniform_codec_grid_v1",
        "flexicodec_threshold": "flexicodec_threshold_codec_grid_v1",
        "codecslime_dp": "codecslime_dp_codec_grid_v1",
        "ple": "ple_codec_grid_v1",
        "elastic_time_greedy": "elastic_time_right_expansion_greedy_v2",
        "elastic_time_dp": "elastic_time_dynamic_programming_v2",
        "dcdit_1d": "dcdit_1d_v2",
        "atome_style": "atome_adjacent_path_matching_fixed_ratio_codec_grid_v1",
        "tadpc_style": "varstok_sequential_dpc_fixed_global_threshold_codec_grid_v1",
    }
    return names[system]


def max_span(system: str) -> int:
    return 2 if system == "atome_style" else 4


def realized_rate(starts: list[int], duration: float) -> float:
    return len(starts) / max(float(duration), 1e-12)


def finite_or_none(value: float):
    return float(value) if math.isfinite(float(value)) else None
