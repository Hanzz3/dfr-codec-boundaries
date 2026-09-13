#!/usr/bin/env python3
"""Evaluate a tuned decoder on the frozen current-nine TIMIT boundaries."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

torch_site_packages = os.environ.get("BRIDGE_TORCH_SITEPACKAGES")
extra_site_packages = os.environ.get("BRIDGE_EXTRA_SITEPACKAGES")
if torch_site_packages and torch_site_packages not in sys.path:
    # This directory exposes only torch/torchaudio and their dist-info. Keeping it
    # first lets Transformers see torch 2.6 without replacing its tokenizers stack.
    sys.path.insert(0, torch_site_packages)
if extra_site_packages and extra_site_packages not in sys.path:
    sys.path.append(extra_site_packages)

import torch
import torchaudio  # noqa: F401

import librosa
import numpy as np
import soundfile as sf
from torch.nn import functional as F

from metrics import (
    SYSTEMS,
    feature_nmse,
    implementation,
    load_feature,
    normalize_words,
    read_jsonl,
    word_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--reference-tracks", "--ground-truth", dest="reference_tracks", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flexicodec-repo", type=Path, required=True)
    parser.add_argument("--target-rate-hz", type=float, default=6.25)
    parser.add_argument("--num-quantizers", type=int, default=8)
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--skip-heavy-metrics", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asr-model", default="openai/whisper-small.en")
    parser.add_argument("--speaker-model", default="microsoft/wavlm-base-plus-sv")
    return parser.parse_args()


def transcript(item: dict) -> str:
    path = (item.get("manifest_row") or {}).get("text")
    if not path:
        return ""
    line = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    fields = line.split(maxsplit=2)
    return fields[2] if len(fields) == 3 and fields[0].isdigit() else line


def load_audio(path: str | Path, target_rate: int = 16000):
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if sample_rate != target_rate:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=target_rate)
    mono = np.asarray(mono, dtype=np.float32)
    return torch.from_numpy(mono).unsqueeze(0), mono, target_rate


def match_audio_length(samples: np.ndarray, target_length: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    return samples[:target_length] if len(samples) >= target_length else np.pad(samples, (0, target_length - len(samples)))


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    length = min(len(reference), len(estimate))
    ref = reference[:length].astype(np.float64)
    est = estimate[:length].astype(np.float64)
    ref -= ref.mean()
    est -= est.mean()
    scale = float(np.dot(est, ref) / max(np.dot(ref, ref), 1e-12))
    target = scale * ref
    noise = est - target
    return float(10.0 * np.log10(max(np.dot(target, target), 1e-12) / max(np.dot(noise, noise), 1e-12)))


def mel_distance(reference: np.ndarray, estimate: np.ndarray) -> float:
    length = min(len(reference), len(estimate))
    ref = librosa.feature.melspectrogram(y=reference[:length], sr=16000, n_fft=1024, hop_length=160, n_mels=80)
    est = librosa.feature.melspectrogram(y=estimate[:length], sr=16000, n_fft=1024, hop_length=160, n_mels=80)
    return float(np.mean(np.abs(librosa.power_to_db(ref + 1e-8) - librosa.power_to_db(est + 1e-8))))


def optional_metrics(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    length = min(len(reference), len(estimate))
    try:
        from pesq import pesq

        pesq_value = float(pesq(16000, reference[:length], estimate[:length], "wb"))
    except Exception:
        pesq_value = math.nan
    try:
        from pystoi import stoi

        stoi_value = float(stoi(reference[:length], estimate[:length], 16000, extended=False))
    except Exception:
        stoi_value = math.nan
    return pesq_value, stoi_value


class HeavyMetrics:
    def __init__(self, device: str, asr_model: str, speaker_model: str) -> None:
        from transformers import AutoFeatureExtractor, AutoProcessor, WhisperForConditionalGeneration, WavLMForXVector

        self.processor = AutoProcessor.from_pretrained(asr_model)
        self.asr = WhisperForConditionalGeneration.from_pretrained(asr_model).to(device).eval()
        self.extractor = AutoFeatureExtractor.from_pretrained(speaker_model)
        self.speaker = WavLMForXVector.from_pretrained(speaker_model).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def transcribe_many(self, audio: list[np.ndarray], batch_size: int) -> list[str]:
        output: list[str] = []
        for offset in range(0, len(audio), batch_size):
            inputs = self.processor(audio[offset : offset + batch_size], sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding=True)
            generation = {"input_features": inputs.input_features.to(self.device)}
            if "attention_mask" in inputs:
                generation["attention_mask"] = inputs.attention_mask.to(self.device)
            token_ids = self.asr.generate(**generation)
            output.extend(self.processor.batch_decode(token_ids, skip_special_tokens=True))
        return [str(value) for value in output]

    @torch.inference_mode()
    def embeddings_many(self, audio: list[np.ndarray], batch_size: int) -> list[torch.Tensor]:
        output: list[torch.Tensor] = []
        for offset in range(0, len(audio), batch_size):
            inputs = self.extractor(audio[offset : offset + batch_size], sampling_rate=16000, return_tensors="pt", padding=True)
            values = self.speaker(**{key: value.to(self.device) for key, value in inputs.items()}).embeddings.cpu()
            output.extend(values.unbind(0))
        return output


class FrozenBoundaryProvider:
    def __init__(self, system: str, expected: list[int]) -> None:
        self.system = system
        self.expected = expected
        self.last_starts: list[int] | None = None

    def __call__(self, frames: torch.Tensor, x_lens: torch.Tensor | None = None):
        if frames.shape[0] != 1:
            raise RuntimeError("bridge evaluator requires FlexiCodec batch size one")
        valid = frames.shape[1] if x_lens is None else int(x_lens[0].item())
        starts = list(self.expected)
        if not starts or starts[0] != 0 or any(right <= left for left, right in zip(starts, starts[1:])):
            raise RuntimeError(f"invalid frozen starts for {self.system}: {starts}")
        if starts[-1] >= valid:
            raise RuntimeError(f"frozen starts exceed codec grid for {self.system}: last={starts[-1]}, valid={valid}")
        keep = np.zeros(valid, dtype=np.int64)
        keep[starts] = 1
        segment_ids = np.cumsum(keep) - 1
        groups = len(starts)
        alignment = np.zeros((groups, frames.shape[1]), dtype=np.float32)
        alignment[segment_ids, np.arange(valid)] = 1.0
        similarity = F.cosine_similarity(frames[:, :-1], frames[:, 1:], dim=-1)
        self.last_starts = starts
        return torch.from_numpy(alignment).unsqueeze(0).to(frames.device), similarity, torch.tensor([groups], dtype=torch.long, device=frames.device)


@contextmanager
def external_alignment(model, provider: FrozenBoundaryProvider):
    original = model._perform_similarity_alignment_vectorized

    def patched(this, h_frames_v, x_lens=None):
        del this
        return provider(h_frames_v, x_lens)

    model._perform_similarity_alignment_vectorized = MethodType(patched, model)
    try:
        yield
    finally:
        model._perform_similarity_alignment_vectorized = original


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def disable_broken_optional_torchvision() -> None:
    """Keep audio-only Transformers imports independent of optional torchvision."""
    import transformers.utils
    import transformers.utils.import_utils as import_utils

    import_utils.is_torchvision_available = lambda: False
    transformers.utils.is_torchvision_available = lambda: False


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    device = "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    disable_broken_optional_torchvision()
    sys.path.insert(0, str(args.flexicodec_repo))
    from flexicodec.infer import encode_flexicodec, prepare_model

    model_dict = prepare_model(
        sensevoice_small_path=str(args.model_root / "SenseVoiceSmall"),
        device=device,
        ckpt_path=str(args.model_root / "flexicodec" / "12hz_v1_half.safetensors"),
        config_path=str(args.model_root / "flexicodec" / "12hz_v1_half_config.yaml"),
    )
    codec = model_dict["model"]
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "generator" not in checkpoint or int(checkpoint.get("step", -1)) < 0:
        raise RuntimeError("checkpoint is not a stable-v2 decoder milestone")
    for name in ("convnext_decoder", "bottleneck_transformer", "dac_decoder"):
        module = codec.dac.decoder if name == "dac_decoder" else getattr(codec, name)
        module.load_state_dict(checkpoint["generator"][name], strict=True)
    codec.eval()
    checkpoint_step = int(checkpoint["step"])
    records = [row for row in read_jsonl(args.feature_index) if row.get("split") == "test"]
    all_predictions = {
        (row["utt_id"], row["system"]): row for row in read_jsonl(args.predictions)
    }
    prediction_utterances = {utt_id for utt_id, _system in all_predictions}
    source_utterances = {row["utt_id"] for row in records}
    unexpected_utterances = prediction_utterances - source_utterances
    if unexpected_utterances:
        raise RuntimeError(
            f"predictions contain {len(unexpected_utterances)} unknown utterances"
        )
    records = [row for row in records if row["utt_id"] in prediction_utterances]
    if args.max_utterances is not None:
        records = records[: args.max_utterances]
    indexed = [(index, row) for index, row in enumerate(records) if index % args.num_shards == args.shard_index]
    gt = {row["utt_id"]: row for row in read_jsonl(args.reference_tracks)}
    expected_prediction_keys = {
        (row["utt_id"], system) for row in records for system in SYSTEMS
    }
    missing = expected_prediction_keys - set(all_predictions)
    if missing:
        raise RuntimeError(
            f"prediction keys are missing: expected={len(expected_prediction_keys)}, "
            f"missing={len(missing)}"
        )
    predictions = {key: all_predictions[key] for key in expected_prediction_keys}
    heavy = None if args.skip_heavy_metrics else HeavyMetrics(device, args.asr_model, args.speaker_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial = args.output_dir / "reconstruction.partial.jsonl"
    final = args.output_dir / "reconstruction.jsonl"
    rows = list(read_jsonl(partial if partial.exists() else final)) if args.resume and (partial.exists() or final.exists()) else []
    completed = {(row["utt_id"], row["system"]) for row in rows}

    def persist(path: Path) -> None:
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        temporary.write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8")
        os.replace(temporary, path)

    for local_index, (test_index, record) in enumerate(indexed, 1):
        audio, reference, sample_rate = load_audio(record["audio"])
        codec_features, _codec_times = load_feature(record)
        reference_text = transcript(gt[record["utt_id"]])
        reference_embedding = heavy.embeddings_many([reference], 1)[0] if heavy else None
        pending: list[tuple[dict, np.ndarray]] = []
        for system in SYSTEMS:
            key = (record["utt_id"], system)
            if key in completed:
                continue
            expected = predictions[key]["starts"]
            provider = FrozenBoundaryProvider(system, expected)
            synchronize(device)
            encode_started = time.perf_counter()
            with torch.inference_mode(), external_alignment(codec, provider):
                encoded = encode_flexicodec(audio.to(device), model_dict, sample_rate, num_quantizers=args.num_quantizers, merging_threshold=0.91)
            synchronize(device)
            encode_seconds = time.perf_counter() - encode_started
            decode_started = time.perf_counter()
            with torch.inference_mode():
                decoded = codec.decode_from_codes(
                    semantic_codes=encoded["semantic_codes"],
                    acoustic_codes=encoded["acoustic_codes"],
                    token_lengths=encoded["token_lengths"],
                )
            synchronize(device)
            decode_seconds = time.perf_counter() - decode_started
            decoded = decoded.detach().float().cpu().squeeze().numpy()
            estimate = match_audio_length(decoded, len(reference))
            if not np.isfinite(estimate).all():
                raise RuntimeError(f"non-finite waveform: {record['utt_id']} / {system}")
            pesq_value, stoi_value = optional_metrics(reference, estimate)
            duration = float(record["duration_sec"])
            row = {
                "utt_id": record["utt_id"],
                "test_index": test_index,
                "system": system,
                "implementation": implementation(system),
                "decoder_checkpoint": str(args.checkpoint),
                "decoder_step": checkpoint_step,
                "rate_hz": args.target_rate_hz,
                "segments": len(expected),
                "realized_rate_hz": len(expected) / duration,
                "reference_samples": len(reference),
                "decoded_samples_before_length_fix": int(np.asarray(decoded).size),
                "decoded_samples_after_length_fix": len(estimate),
                "decoded_waveform_finite": True,
                "frozen_boundary_starts": expected,
                "feature_nmse": feature_nmse(codec_features, expected),
                "wer": math.nan,
                "word_errors": math.nan,
                "reference_words": len(normalize_words(reference_text)),
                "pesq_wb": pesq_value,
                "stoi": stoi_value,
                "si_sdr": si_sdr(reference, estimate),
                "mel_distance": mel_distance(reference, estimate),
                "speaker_similarity": math.nan,
                "encode_rtf": encode_seconds / duration,
                "decode_rtf": decode_seconds / duration,
                "codec_rtf": (encode_seconds + decode_seconds) / duration,
                "reference_text": reference_text,
                "hypothesis": "",
            }
            pending.append((row, estimate))
        if heavy and pending:
            hypotheses = heavy.transcribe_many([estimate for _, estimate in pending], args.metric_batch_size)
            embeddings = heavy.embeddings_many([estimate for _, estimate in pending], args.metric_batch_size)
            for (row, _estimate), hypothesis, embedding in zip(pending, hypotheses, embeddings):
                errors, words = word_counts(reference_text, hypothesis)
                row["hypothesis"] = hypothesis
                row["word_errors"] = errors
                row["reference_words"] = words
                row["wer"] = errors / max(words, 1)
                row["speaker_similarity"] = float(F.cosine_similarity(reference_embedding, embedding, dim=0))
        for row, _estimate in pending:
            rows.append(row)
            completed.add((row["utt_id"], row["system"]))
        if local_index % 5 == 0 or local_index == len(indexed):
            persist(partial)
            print(f"shard {args.shard_index}: reconstructed {local_index}/{len(indexed)}", flush=True)
    persist(final)
    if partial.exists():
        partial.unlink()
    (args.output_dir / "eval.exit").write_text("0\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
