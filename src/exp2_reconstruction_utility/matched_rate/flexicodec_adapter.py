"""Inject external segmentations into FlexiCodec without changing its default path."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F

from models import ElasticPredictor, LocalPredictabilityModel
from merging_selectors import DIRECT_RATE_SYSTEMS, select, selector_features
from direct_boundaries import snap_boundaries_to_starts
from algorithm_types import result_from_starts, validate_result


def prepare_pinned_flexicodec(
    flexicodec_repo,
    device: str,
    model_root="./models",
):  # noqa: ANN001
    """Load only the shared, revision-pinned checkpoints; never download implicitly."""
    from pathlib import Path
    import sys

    repository = Path(flexicodec_repo)
    root = Path(model_root)
    sys.path.insert(0, str(repository))
    # The pinned runtime is prepared before the experiment starts. FunASR's
    # remote-code loader otherwise invokes pip for the model requirements on
    # every process startup, which is unsafe when eight GPU workers load the
    # same checkpoint concurrently.
    from funasr.utils import install_model_requirements

    install_model_requirements.install_requirements = lambda _path: None
    from flexicodec.infer import prepare_model

    sensevoice = root / "SenseVoiceSmall"
    config = root / "flexicodec" / "12hz_v1_half_config.yaml"
    checkpoint = root / "flexicodec" / "12hz_v1_half.safetensors"
    missing = [str(path) for path in (sensevoice, config, checkpoint) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing pinned FlexiCodec assets: {missing}")
    return prepare_model(
        sensevoice_small_path=str(sensevoice),
        device=device,
        ckpt_path=str(checkpoint),
        config_path=str(config),
    )


def selection_to_alignment(segment_ids: np.ndarray, dtype=np.float32) -> np.ndarray:
    ids = np.asarray(segment_ids, dtype=np.int64)
    if ids.ndim != 1 or not len(ids) or ids[0] != 0:
        raise ValueError("segment_ids must be a non-empty vector starting at zero")
    groups = int(ids.max()) + 1
    if not np.array_equal(np.unique(ids), np.arange(groups)):
        raise ValueError("segment_ids must be contiguous")
    alignment = np.zeros((groups, len(ids)), dtype=dtype)
    alignment[ids, np.arange(len(ids))] = 1
    return alignment


class PreparedAlignmentProvider:
    """Inject saved boundary times into the codec's actual downsampled frame grid."""

    def __init__(self, max_span: int = 8) -> None:
        self.max_span = int(max_span)
        self.boundaries: list[list[float]] = []
        self.durations: list[float] = []
        self.no_merge: list[bool] = []
        self.last_results = []

    def set_batch(
        self,
        boundaries: list[list[float]],
        durations: list[float],
        no_merge: list[bool] | None = None,
    ) -> None:
        if len(boundaries) != len(durations):
            raise ValueError("boundaries and durations must have equal batch size")
        self.boundaries = [[float(value) for value in row] for row in boundaries]
        self.durations = [float(value) for value in durations]
        self.no_merge = list(no_merge or [False] * len(durations))

    def __call__(self, frames: torch.Tensor, x_lens: torch.Tensor | None = None):
        batch, width, _ = frames.shape
        if len(self.boundaries) != batch or len(self.durations) != batch or len(self.no_merge) != batch:
            raise RuntimeError("set_batch must be called before each FlexiCodec encode")
        lengths = [width] * batch if x_lens is None else [int(value) for value in x_lens.detach().cpu().tolist()]
        alignments = []
        counts = []
        results = []
        for index, (valid, duration) in enumerate(zip(lengths, self.durations)):
            times = ((np.arange(valid, dtype=np.float32) + 0.5) * duration / valid).astype(np.float32)
            if self.no_merge[index]:
                starts = list(range(valid))
                source_count = max(0, valid - 1)
                technical_count = 0
            else:
                starts, source_count, technical_count = snap_boundaries_to_starts(
                    self.boundaries[index], times, self.max_span
                )
            ids = np.zeros(valid, dtype=np.int64)
            keep = np.zeros(valid, dtype=np.int64)
            keep[starts] = 1
            ids = np.cumsum(keep) - 1
            alignment = selection_to_alignment(ids)
            padded = np.zeros((alignment.shape[0], width), dtype=np.float32)
            padded[:, :valid] = alignment
            alignments.append(torch.from_numpy(padded).to(frames.device))
            counts.append(alignment.shape[0])
            result = result_from_starts(
                "prepared_external_alignment",
                valid,
                starts,
                np.full(valid, np.nan, dtype=np.float32),
                float("nan"),
                0.0,
                duration,
                0,
                {
                    "control_mode": "prepared_boundary_times",
                    "source_boundaries_after_collision": source_count,
                    "technical_maxspan_boundaries": technical_count,
                },
            )
            validate_result(result, valid, None, self.max_span)
            results.append(result)
        max_groups = max(item.shape[0] for item in alignments)
        output = frames.new_zeros((batch, max_groups, width))
        for index, alignment in enumerate(alignments):
            output[index, : alignment.shape[0]] = alignment
        similarity = F.cosine_similarity(frames[:, :-1], frames[:, 1:], dim=-1)
        self.last_results = results
        return output, similarity, torch.tensor(counts, dtype=torch.long, device=frames.device)


class ExternalAlignmentProvider:
    def __init__(
        self,
        system: str,
        rate_hz: float,
        mean: np.ndarray,
        std: np.ndarray,
        elastic: ElasticPredictor,
        local: LocalPredictabilityModel,
        control_parameter: float | None = None,
        max_span: int = 8,
        device: str = "cuda",
    ) -> None:
        self.system = system
        self.rate_hz = float(rate_hz)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.elastic = elastic
        self.local = local
        self.control_parameter = control_parameter
        self.max_span = int(max_span)
        self.device = device
        self.durations: list[float] = []
        self.last_results = []

    def set_durations(self, durations: list[float]) -> None:
        self.durations = [float(value) for value in durations]

    def __call__(self, frames: torch.Tensor, x_lens: torch.Tensor | None = None):
        batch, width, _ = frames.shape
        lengths = [width] * batch if x_lens is None else [int(value) for value in x_lens.detach().cpu().tolist()]
        if len(self.durations) != batch:
            raise RuntimeError("set_durations must be called before each FlexiCodec encode")
        alignments = []
        results = []
        counts = []
        for index, (valid, duration) in enumerate(zip(lengths, self.durations)):
            feature = frames[index, :valid].detach().float().cpu().numpy()
            if feature.shape[1] != self.mean.shape[0]:
                raise RuntimeError(
                    f"SenseVoice feature dimension mismatch: callback={feature.shape[1]} cache={self.mean.shape[0]}"
                )
            normalized = (feature - self.mean) / self.std
            selection_features = selector_features(self.system, feature, normalized)
            times = ((np.arange(valid, dtype=np.float32) + 0.5) * duration / valid).astype(np.float32)
            is_direct = self.system in DIRECT_RATE_SYSTEMS
            target = max(1, int(round(duration * self.rate_hz))) if is_direct else None
            result = select(
                self.system,
                selection_features,
                times,
                target,
                self.max_span,
                self.elastic,
                self.local,
                self.device,
                control_parameter=self.control_parameter,
            )
            alignment = selection_to_alignment(result.segment_ids)
            padded = np.zeros((alignment.shape[0], width), dtype=np.float32)
            padded[:, :valid] = alignment
            # Leave padded columns unassigned. Folding them into the last real
            # group changes its mean and makes QueryTokenAggregator infer the
            # padded batch width as the utterance length.
            alignments.append(torch.from_numpy(padded).to(frames.device))
            counts.append(alignment.shape[0])
            results.append(result)
        max_groups = max(item.shape[0] for item in alignments)
        output = frames.new_zeros((batch, max_groups, width))
        for index, alignment in enumerate(alignments):
            output[index, : alignment.shape[0]] = alignment
        similarity = F.cosine_similarity(frames[:, :-1], frames[:, 1:], dim=-1)
        self.last_results = results
        return output, similarity, torch.tensor(counts, dtype=torch.long, device=frames.device)


@contextmanager
def external_alignment(model, provider: ExternalAlignmentProvider):
    """Temporarily override only the internal alignment selector."""
    method_name = "_perform_similarity_alignment_vectorized"
    had_instance_override = method_name in model.__dict__
    original_instance_value = model.__dict__.get(method_name)
    original = model._perform_similarity_alignment_vectorized

    def patched(this, h_frames_v, x_lens=None):  # noqa: ANN001
        del this
        return provider(h_frames_v, x_lens)

    model._perform_similarity_alignment_vectorized = MethodType(patched, model)
    try:
        yield provider
    finally:
        if had_instance_override:
            setattr(model, method_name, original_instance_value)
        else:
            delattr(model, method_name)
