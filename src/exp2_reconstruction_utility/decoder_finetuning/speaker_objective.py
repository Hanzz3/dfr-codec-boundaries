#!/usr/bin/env python3
"""Differentiable frozen-WavLM speaker objective with cached references."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F


class FrozenSpeakerEmbedder(torch.nn.Module):
    """Keep WavLM frozen while preserving gradients to the input waveform."""

    def __init__(self, model_path: Path, device: torch.device, crop_samples: int) -> None:
        super().__init__()
        if crop_samples < 16000:
            raise ValueError("speaker crop must be at least one second")
        from transformers import WavLMForXVector
        from transformers.models.wavlm import modeling_wavlm

        modeling_wavlm.is_peft_available = lambda: False
        self.model = WavLMForXVector.from_pretrained(model_path).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.crop_samples = int(crop_samples)
        self.minimum_samples = min(self.crop_samples, 32000)
        self.device = device

    def train(self, mode: bool = True):  # noqa: ANN201
        super().train(False)
        self.model.eval()
        return self

    def _prepare(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if waveforms.ndim == 3 and waveforms.shape[1] == 1:
            waveforms = waveforms[:, 0]
        if waveforms.ndim != 2 or lengths.ndim != 1 or len(waveforms) != len(lengths):
            raise ValueError("expected padded [batch, samples] waveforms and [batch] lengths")
        clips = []
        clip_lengths = []
        for index, raw_length in enumerate(lengths.detach().cpu().tolist()):
            length = min(int(raw_length), int(waveforms.shape[-1]))
            if length < 400:
                raise ValueError(f"speaker waveform is too short: {length} samples")
            width = min(length, self.crop_samples)
            start = max(0, (length - width) // 2)
            clip = waveforms[index, start : start + width].float()
            if width < self.minimum_samples:
                repeats = (self.minimum_samples + width - 1) // width
                clip = clip.repeat(repeats)[: self.minimum_samples]
                width = self.minimum_samples
            clips.append(clip)
            clip_lengths.append(width)
        padded = torch.nn.utils.rnn.pad_sequence(clips, batch_first=True)
        valid_lengths = torch.tensor(clip_lengths, device=padded.device, dtype=torch.long)
        positions = torch.arange(padded.shape[1], device=padded.device).unsqueeze(0)
        mask = positions < valid_lengths.unsqueeze(1)
        count = valid_lengths.to(padded.dtype).clamp_min(1).unsqueeze(1)
        mean = (padded * mask).sum(dim=1, keepdim=True) / count
        centered = (padded - mean) * mask
        variance = centered.square().sum(dim=1, keepdim=True) / count
        normalized = centered / torch.sqrt(variance + 1e-7)
        return normalized, mask.to(torch.long)

    def embeddings(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        values, attention_mask = self._prepare(waveforms, lengths)
        embeddings = self.model(input_values=values, attention_mask=attention_mask).embeddings
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("non-finite WavLM speaker embedding")
        return F.normalize(embeddings.float(), dim=-1)

    def loss(
        self,
        estimates: torch.Tensor,
        lengths: torch.Tensor,
        utt_ids: list[str],
        reference_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        estimate_embeddings = self.embeddings(estimates, lengths)
        try:
            references = torch.stack([reference_cache[utt_id] for utt_id in utt_ids])
        except KeyError as error:
            raise KeyError(f"missing cached speaker reference: {error.args[0]}") from error
        references = F.normalize(references.to(estimate_embeddings.device, dtype=torch.float32), dim=-1)
        loss = (1.0 - (estimate_embeddings * references).sum(dim=-1)).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite speaker loss")
        return loss


def load_reference_cache(cache_dir: Path, expected_rows: int | None = None) -> dict[str, torch.Tensor]:
    parts = sorted(cache_dir.glob("speaker_ref.part-*-of-*.pt"))
    if not parts:
        raise FileNotFoundError(f"no speaker reference cache parts in {cache_dir}")
    merged: dict[str, torch.Tensor] = {}
    metadata = None
    for path in parts:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        current = {
            "schema_version": int(payload["schema_version"]),
            "crop_samples": int(payload["crop_samples"]),
            "embedding_dim": int(payload["embedding_dim"]),
        }
        if metadata is None:
            metadata = current
        elif current != metadata:
            raise RuntimeError(f"speaker cache metadata mismatch in {path}")
        for utt_id, embedding in payload["embeddings"].items():
            if utt_id in merged:
                raise RuntimeError(f"duplicate speaker cache key: {utt_id}")
            if embedding.ndim != 1 or not torch.isfinite(embedding).all():
                raise RuntimeError(f"invalid speaker embedding: {utt_id}")
            merged[utt_id] = embedding.float()
    if expected_rows is not None and len(merged) != expected_rows:
        raise RuntimeError(f"speaker cache rows {len(merged)} != expected {expected_rows}")
    if not merged or metadata is None or metadata["schema_version"] != 1:
        raise RuntimeError("invalid speaker reference cache")
    return merged


def crop_samples(sample_rate: int, crop_seconds: float) -> int:
    value = int(round(sample_rate * crop_seconds))
    if not math.isfinite(crop_seconds) or value < sample_rate:
        raise ValueError("speaker crop must be finite and at least one second")
    return value
