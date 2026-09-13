#!/usr/bin/env python3
"""Evaluate original and algorithm-adapted decoders from identical cached codes."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from decoder_training import load_decoder_delta
from flexicodec_adapter import prepare_pinned_flexicodec
from io_utils import load_audio, read_jsonl


OUTPUT_COLUMNS = (
    "utt_id",
    "system",
    "condition",
    "decoder_mode",
    "target_rate_hz",
    "realized_rate_hz",
    "wer",
    "wer_errors",
    "wer_reference_words",
    "hypothesis",
    "pesq_wb",
    "stoi",
    "si_sdr",
    "log_mel_distance",
    "wavlm_speaker_similarity",
    "content_bps",
    "duration_bps",
    "total_bps",
    "encode_rtf",
    "decode_rtf",
    "total_rtf",
    "reference_samples",
    "decoded_samples",
)


def limit_unique_utterances(rows: list[dict], limit: int | None) -> list[dict]:
    if not limit:
        return rows
    allowed = []
    seen = set()
    for row in rows:
        utt_id = row["utt_id"]
        if utt_id not in seen:
            if len(seen) >= limit:
                continue
            seen.add(utt_id)
            allowed.append(utt_id)
    allowed_set = set(allowed)
    return [row for row in rows if row["utt_id"] in allowed_set]


def normalize_text(value: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", value.lower()).split()


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, lhs in enumerate(left, 1):
        current = [i]
        for j, rhs in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (lhs != rhs)))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    errors, reference_words = word_error_counts(reference, hypothesis)
    return errors / max(reference_words, 1)


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    return edit_distance(ref, hyp), len(ref)


def match_length(samples: np.ndarray, target: int) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    return values[:target] if len(values) >= target else np.pad(values, (0, target - len(values)))


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64) - float(np.mean(reference))
    estimate = estimate.astype(np.float64) - float(np.mean(estimate))
    scale = float(np.dot(estimate, reference) / max(np.dot(reference, reference), 1e-12))
    target = scale * reference
    noise = estimate - target
    return float(10.0 * np.log10(max(np.dot(target, target), 1e-12) / max(np.dot(noise, noise), 1e-12)))


def mel_distance(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = librosa.feature.melspectrogram(y=reference, sr=16000, n_fft=1024, hop_length=160, n_mels=80)
    est = librosa.feature.melspectrogram(y=estimate, sr=16000, n_fft=1024, hop_length=160, n_mels=80)
    return float(np.mean(np.abs(librosa.power_to_db(ref + 1e-8) - librosa.power_to_db(est + 1e-8))))


def pesq_stoi(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    from pesq import pesq
    from pystoi import stoi

    return float(pesq(16000, reference, estimate, "wb")), float(stoi(reference, estimate, 16000, extended=False))


class FrozenMetrics:
    def __init__(self, whisper_path: str, wavlm_path: str, device: str) -> None:
        from transformers import AutoFeatureExtractor, AutoProcessor, WavLMForXVector, WhisperForConditionalGeneration
        from transformers.models.wavlm import modeling_wavlm

        self.device = torch.device(device)
        # This frozen metric never loads PEFT adapters. Ignore an unrelated,
        # globally installed PEFT package that can trigger an incompatible
        # optional LoRA branch inside the Transformers WavLM TDNN layer.
        modeling_wavlm.is_peft_available = lambda: False
        self.whisper_processor = AutoProcessor.from_pretrained(whisper_path)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_path).to(self.device).eval()
        self.wavlm_processor = AutoFeatureExtractor.from_pretrained(wavlm_path)
        self.wavlm = WavLMForXVector.from_pretrained(wavlm_path).to(self.device).eval()

    @torch.inference_mode()
    def transcript(self, audio: np.ndarray) -> str:
        inputs = self.whisper_processor(audio, sampling_rate=16000, return_tensors="pt")
        ids = self.whisper.generate(
            inputs.input_features.to(self.device),
            language="en",
            task="transcribe",
        )
        return str(
            self.whisper_processor.batch_decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )

    @torch.inference_mode()
    def speaker_embedding(self, audio: np.ndarray) -> torch.Tensor:
        inputs = self.wavlm_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        return self.wavlm(**{key: value.to(self.device) for key, value in inputs.items()}).embeddings[0]


def code_width(module, fallback: int) -> int:  # noqa: ANN001
    size = int(getattr(module, "codebook_size", fallback))
    return int(math.ceil(math.log2(max(size, 2))))


def evaluate_mode(
    codec,
    rows,
    mode,
    metrics,
    device,
    semantic_width,
    acoustic_width,
    examples_dir,
    checkpoint=None,
    decode_fn=None,
):  # noqa: ANN001
    output = []
    for completed, row in enumerate(rows, 1):
        cached = torch.load(row["code_path"], map_location="cpu", weights_only=True)
        semantic = cached["semantic_codes"].unsqueeze(0).to(device)
        acoustic = None if cached["acoustic_codes"] is None else cached["acoustic_codes"].unsqueeze(0).to(device)
        token_lengths = cached["token_lengths"].unsqueeze(0).to(device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            decoded = (
                codec.decode_from_codes(semantic, acoustic, token_lengths)
                if decode_fn is None
                else decode_fn(semantic, acoustic, token_lengths)
            )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - started
        estimate = decoded.detach().float().cpu().numpy().reshape(-1)
        reference = load_audio(row["audio"])
        estimate = match_length(estimate, len(reference))
        if not np.isfinite(estimate).all() or len(estimate) != len(reference):
            raise RuntimeError(f"invalid waveform: {row['utt_id']} {row['system']} {row['condition']} {mode}")
        hypothesis = metrics.transcript(estimate)
        wer_errors, wer_reference_words = word_error_counts(
            row.get("transcript", ""), hypothesis
        )
        ref_embedding = metrics.speaker_embedding(reference)
        est_embedding = metrics.speaker_embedding(estimate)
        pesq_value, stoi_value = pesq_stoi(reference, estimate)
        segments = int(cached["speech_token_len"])
        duration = float(row["duration_sec"])
        semantic_bits = int(cached["semantic_codes"].numel()) * semantic_width
        acoustic_bits = 0 if cached["acoustic_codes"] is None else int(cached["acoustic_codes"].numel()) * acoustic_width
        duration_bits = segments * 3
        encode_seconds = float(cached.get("encode_seconds", row.get("encode_seconds", math.nan)))
        output.append(
            {
                "utt_id": row["utt_id"],
                "system": row["system"],
                "condition": row["condition"],
                "decoder_mode": mode,
                "target_rate_hz": row.get("target_rate_hz"),
                "realized_rate_hz": row["realized_rate_hz"],
                "wer": wer_errors / max(wer_reference_words, 1),
                "wer_errors": wer_errors,
                "wer_reference_words": wer_reference_words,
                "hypothesis": hypothesis,
                "pesq_wb": pesq_value,
                "stoi": stoi_value,
                "si_sdr": si_sdr(reference, estimate),
                "log_mel_distance": mel_distance(reference, estimate),
                "wavlm_speaker_similarity": float(torch.nn.functional.cosine_similarity(ref_embedding, est_embedding, dim=0)),
                "content_bps": (semantic_bits + acoustic_bits) / duration,
                "duration_bps": duration_bits / duration,
                "total_bps": (semantic_bits + acoustic_bits + duration_bits) / duration,
                "encode_rtf": encode_seconds / duration,
                "decode_rtf": decode_seconds / duration,
                "total_rtf": (encode_seconds + decode_seconds) / duration,
                "reference_samples": len(reference),
                "decoded_samples": len(estimate),
            }
        )
        if completed <= 20:
            examples_dir.mkdir(parents=True, exist_ok=True)
            sf.write(examples_dir / f"{row['utt_id']}__{row['condition']}__{mode}.wav", estimate, 16000)
        if completed % 10 == 0 or completed == len(rows):
            print(f"{row['system']} {mode}: {completed}/{len(rows)}", flush=True)
        if checkpoint is not None and (completed % 25 == 0 or completed == len(rows)):
            checkpoint(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-index", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--modes", nargs="+", choices=("original", "tuned"), default=("original", "tuned"))
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flexicodec-repo", type=Path, default=Path("./repos/FlexiCodec"))
    parser.add_argument("--whisper-model", default="./models/whisper-small")
    parser.add_argument("--wavlm-model", default="./models/wavlm-base-plus-sv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-utterances", type=int)
    args = parser.parse_args()

    requested_modes = tuple(mode for mode in ("original", "tuned") if mode in set(args.modes))
    if "tuned" in requested_modes and args.checkpoint is None:
        raise ValueError("--checkpoint is required when tuned mode is requested")

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid reconstruction shard")
    rows = [row for row in read_jsonl(args.code_index) if row["system"] == args.system and row["split"] == "test"]
    if args.conditions:
        allowed_conditions = set(args.conditions)
        rows = [row for row in rows if row["condition"] in allowed_conditions]
    rows = limit_unique_utterances(rows, args.max_utterances)
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"reconstruction.part-{args.shard_index:05d}-of-{args.num_shards:05d}.csv"
    exit_suffix = "original.exit" if requested_modes == ("original",) else "exit"
    exit_path = args.output_dir / f"reconstruction.part-{args.shard_index:05d}-of-{args.num_shards:05d}.{exit_suffix}"
    if not rows:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(path, index=False)
        exit_path.write_text("0\n", encoding="ascii")
        return 0
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(args.flexicodec_repo))
    model_dict = prepare_pinned_flexicodec(args.flexicodec_repo, str(device))
    codec = model_dict["model"].eval()
    semantic_width = code_width(codec.semantic_vq, 32768)
    acoustic_width = code_width(codec.dac, 4096)
    metrics = FrozenMetrics(args.whisper_model, args.wavlm_model, str(device))
    expected_keys = {(row["utt_id"], row["condition"]) for row in rows}
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not existing.empty:
        existing = existing[
            (existing["system"] == args.system)
            & (existing["decoder_mode"].isin(requested_modes))
        ].copy()
        if args.conditions:
            existing = existing[existing["condition"].isin(args.conditions)].copy()
        existing = existing[
            existing.apply(lambda row: (row["utt_id"], row["condition"]) in expected_keys, axis=1)
        ].copy()
    references = {row["utt_id"]: row.get("transcript", "") for row in rows}
    if not existing.empty and not {"wer_errors", "wer_reference_words"}.issubset(existing.columns):
        counts = [
            word_error_counts(references[row["utt_id"]], str(row["hypothesis"]))
            for _, row in existing.iterrows()
        ]
        existing["wer_errors"] = [errors for errors, _ in counts]
        existing["wer_reference_words"] = [words for _, words in counts]
        existing["wer"] = [errors / max(words, 1) for errors, words in counts]
    def mode_keys(mode: str) -> set[tuple[str, str]]:
        if existing.empty:
            return set()
        subset = existing[existing["decoder_mode"] == mode]
        return set(zip(subset["utt_id"], subset["condition"]))

    def write_existing(frame: pd.DataFrame) -> None:
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        frame[list(OUTPUT_COLUMNS)].to_csv(temporary, index=False)
        temporary.replace(path)

    if not existing.empty:
        write_existing(existing)

    for mode in requested_modes:
        observed = mode_keys(mode)
        if observed == expected_keys and len(existing[existing["decoder_mode"] == mode]) == len(expected_keys):
            print(f"{args.system} {mode}: already complete ({len(expected_keys)})", flush=True)
            continue
        if mode == "tuned":
            load_decoder_delta(codec, args.checkpoint)
            codec.eval()
        pending = [row for row in rows if (row["utt_id"], row["condition"]) not in observed]

        def checkpoint(partial_rows: list[dict]) -> None:
            partial = pd.DataFrame(partial_rows, columns=OUTPUT_COLUMNS)
            combined = pd.concat([existing, partial], ignore_index=True)
            combined = combined.drop_duplicates(
                ["utt_id", "system", "condition", "decoder_mode"], keep="last"
            )
            write_existing(combined)

        output = evaluate_mode(
            codec, pending, mode, metrics, device, semantic_width, acoustic_width,
            args.output_dir / "examples", checkpoint=checkpoint,
        )
        existing = pd.concat([existing, pd.DataFrame(output, columns=OUTPUT_COLUMNS)], ignore_index=True)
        existing = existing.drop_duplicates(["utt_id", "system", "condition", "decoder_mode"], keep="last")
        write_existing(existing)
    exit_path.write_text("0\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
