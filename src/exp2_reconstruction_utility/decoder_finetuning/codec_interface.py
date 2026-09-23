#!/usr/bin/env python3
"""Two-step codec-native GAN smoke test for the shared L49 decoder."""

from __future__ import annotations

import argparse
import json
import librosa
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from audiotools import AudioSignal
from dac.model import Discriminator
from dac.nn.loss import GANLoss, MelSpectrogramLoss

from paper_protocol import DECODER_TRAINING_SYSTEMS

SYSTEMS = DECODER_TRAINING_SYSTEMS
DYNAMIC_SYSTEMS = ("atome_style", "tadpc_style")
TRAINABLE_PREFIXES = ("convnext_decoder.", "bottleneck_transformer.", "dac.decoder.")


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_audio(path: Path) -> torch.Tensor:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = values.mean(axis=1)
    if sample_rate != 16000:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=16000)
    return torch.from_numpy(np.asarray(mono, dtype=np.float32)).float()


def map_starts(
    starts: list[int], source_frames: int, target_frames: int, max_span: int
) -> tuple[np.ndarray, float, float]:
    """Project starts to the codec grid while preserving count and span limits."""
    source = np.asarray(starts, dtype=np.int64)
    if source.ndim != 1 or not len(source) or source[0] != 0 or np.any(np.diff(source) <= 0):
        raise ValueError("source starts must be strictly increasing and begin at zero")
    if source[-1] >= source_frames or len(source) > target_frames:
        raise ValueError("source starts do not fit the target frame grid")
    count = len(source)
    raw = source.astype(np.float64) * target_frames / source_frames
    costs = np.full((count, target_frames), np.inf, dtype=np.float64)
    parents = np.full((count, target_frames), -1, dtype=np.int32)
    costs[0, 0] = raw[0] ** 2
    for index in range(1, count):
        lower = index
        upper = target_frames - (count - index)
        for position in range(lower, upper + 1):
            candidates = range(max(index - 1, position - max_span), position)
            parent = min(candidates, key=lambda item: costs[index - 1, item])
            if np.isfinite(costs[index - 1, parent]):
                costs[index, position] = costs[index - 1, parent] + (position - raw[index]) ** 2
                parents[index, position] = parent
    candidates = [
        position
        for position in range(count - 1, target_frames)
        if target_frames - position <= max_span and np.isfinite(costs[-1, position])
    ]
    if not candidates:
        raise RuntimeError(
            f"cannot map {count} segments from {source_frames} to {target_frames} frames"
        )
    position = min(candidates, key=lambda item: costs[-1, item])
    mapped = np.empty(count, dtype=np.int64)
    for index in range(count - 1, -1, -1):
        mapped[index] = position
        position = int(parents[index, position]) if index else -1
    lengths = np.diff(np.append(mapped, target_frames))
    if mapped[0] != 0 or np.any(np.diff(mapped) <= 0) or np.any(lengths > max_span):
        raise RuntimeError("invalid mapped alignment")
    error = np.abs(mapped - raw)
    return mapped, float(error.mean()), float(error.max())


class AlignmentProvider:
    def __init__(self, row: dict, max_span: int) -> None:
        self.row = row
        self.max_span = int(max_span)
        self.result: dict | None = None

    def __call__(self, frames: torch.Tensor, x_lens: torch.Tensor | None = None):
        batch, width, _ = frames.shape
        if batch != 1:
            raise RuntimeError("the smoke provider expects one variable-length item")
        valid = width if x_lens is None else int(x_lens[0].item())
        mapped, mean_error, max_error = map_starts(
            self.row["starts"], int(self.row["frames"]), valid, self.max_span
        )
        ends = np.append(mapped[1:], valid)
        alignment = frames.new_zeros((len(mapped), valid))
        for index, (start, end) in enumerate(zip(mapped, ends)):
            alignment[index, int(start) : int(end)] = 1.0
        output = frames.new_zeros((1, len(mapped), width))
        output[0, :, :valid] = alignment
        similarity = F.cosine_similarity(frames[:, :-1], frames[:, 1:], dim=-1)
        self.result = {
            "source_segments": len(self.row["starts"]),
            "mapped_segments": len(mapped),
            "codec_frames": valid,
            "mapped_max_span_frames": int(np.diff(np.append(mapped, valid)).max()),
            "mapping_mean_error_frames": mean_error,
            "mapping_max_error_frames": max_error,
        }
        return output, similarity, torch.tensor([len(mapped)], device=frames.device)


class DynamicAlignmentProvider:
    """Select boundaries on the exact codec-grid representation being encoded."""

    def __init__(self, row: dict) -> None:
        if row["system"] not in DYNAMIC_SYSTEMS:
            raise ValueError(f"not a dynamic selector: {row['system']}")
        self.row = row
        self.result: dict | None = None

    def __call__(self, frames: torch.Tensor, x_lens: torch.Tensor | None = None):
        from dynamic_selectors import atome_style_starts, tadpc_style_starts

        batch, width, _ = frames.shape
        if batch != 1:
            raise RuntimeError("dynamic selectors require one variable-length item")
        valid = width if x_lens is None else int(x_lens[0].item())
        values = frames[0, :valid].detach().float().cpu().numpy()
        if self.row["system"] == "atome_style":
            starts = atome_style_starts(values, merge_ratio=0.5)
            selector = "atome_fixed_ratio_0p5_codec_grid"
        else:
            starts = tadpc_style_starts(values, threshold=0.8561496734619141)
            selector = "official_tadpc_threshold_0p8561496734619141_codec_grid"
        mapped = np.asarray(starts, dtype=np.int64)
        ends = np.append(mapped[1:], valid)
        alignment = frames.new_zeros((len(mapped), valid))
        for index, (start, end) in enumerate(zip(mapped, ends)):
            alignment[index, int(start) : int(end)] = 1.0
        output = frames.new_zeros((1, len(mapped), width))
        output[0, :, :valid] = alignment
        similarity = F.cosine_similarity(frames[:, :-1], frames[:, 1:], dim=-1)
        lengths = np.diff(np.append(mapped, valid))
        self.result = {
            "source_segments": len(mapped),
            "mapped_segments": len(mapped),
            "codec_frames": valid,
            "mapped_max_span_frames": int(lengths.max()),
            "mapping_mean_error_frames": 0.0,
            "mapping_max_error_frames": 0.0,
            "selector": selector,
            "starts": starts,
        }
        return output, similarity, torch.tensor([len(mapped)], device=frames.device)


def alignment_provider(row: dict, max_span: int):  # noqa: ANN201
    if row["system"] in DYNAMIC_SYSTEMS:
        return DynamicAlignmentProvider(row)
    return AlignmentProvider(row, max_span)


@contextmanager
def external_alignment(model, provider: AlignmentProvider):  # noqa: ANN001
    name = "_perform_similarity_alignment_vectorized"
    original = getattr(model, name)

    def patched(this, frames, x_lens=None):  # noqa: ANN001
        del this
        return provider(frames, x_lens)

    setattr(model, name, MethodType(patched, model))
    try:
        yield
    finally:
        setattr(model, name, original)


def decoder_modules(codec: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    return codec.convnext_decoder, codec.bottleneck_transformer, codec.dac.decoder


def configure_trainable(codec: torch.nn.Module) -> list[str]:
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad = False
    for module in decoder_modules(codec):
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad = True
    names = [name for name, parameter in codec.named_parameters() if parameter.requires_grad]
    invalid = [name for name in names if not name.startswith(TRAINABLE_PREFIXES)]
    if invalid or not names:
        raise RuntimeError(f"unexpected trainable codec parameters: {invalid}")
    return names


def scheduled_systems(seed: int, max_steps: int, batch_size: int) -> list[str]:
    count = max_steps * batch_size
    labels = list(SYSTEMS) * (count // len(SYSTEMS)) + list(SYSTEMS[: count % len(SYSTEMS)])
    random.Random(seed).shuffle(labels)
    return labels


def snapshot(parameters) -> list[torch.Tensor]:  # noqa: ANN001
    return [parameter.detach().float().cpu().clone() for parameter in parameters]


def max_parameter_delta(before: list[torch.Tensor], parameters) -> float:  # noqa: ANN001
    return max(
        float((current.detach().float().cpu() - initial).abs().max())
        for initial, current in zip(before, parameters)
    )


def normalize_waveform(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        value = value.unsqueeze(1)
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != 1:
        raise RuntimeError(f"unexpected waveform shape: {tuple(value.shape)}")
    return value


def codebook_count(semantic_codes: torch.Tensor, acoustic_codes: torch.Tensor | None) -> int:
    semantic_books = 1 if semantic_codes.ndim == 2 else int(semantic_codes.shape[1])
    acoustic_books = 0
    if acoustic_codes is not None:
        acoustic_books = 1 if acoustic_codes.ndim == 2 else int(acoustic_codes.shape[1])
    return semantic_books + acoustic_books


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flexicodec-repo", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--speaker-model", type=Path, required=True)
    parser.add_argument("--speaker-weight", type=float, choices=(0.0, 0.1), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-span", type=int, default=8)
    args = parser.parse_args()

    if args.steps != 2 or args.batch_size != 4:
        raise ValueError("formal smoke protocol is exactly two batch-4 steps")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    rows = read_jsonl(args.manifest)
    grouped = {system: [] for system in SYSTEMS}
    for row in rows:
        system = str(row["system"])
        if system not in grouped:
            raise RuntimeError(f"unexpected system: {system}")
        local = dict(row)
        local["audio"] = str(args.audio_root / Path(row["audio"]).name)
        grouped[system].append(local)
    if any(len(grouped[system]) != 4 for system in SYSTEMS):
        raise RuntimeError({system: len(grouped[system]) for system in SYSTEMS})

    formal_labels = scheduled_systems(args.seed, 10000, args.batch_size)
    formal_counts = {system: formal_labels.count(system) for system in SYSTEMS}
    if max(formal_counts.values()) - min(formal_counts.values()) > 1:
        raise RuntimeError(f"unbalanced formal sampler: {formal_counts}")
    smoke_labels = formal_labels[: args.steps * args.batch_size]
    selection_rng = random.Random(args.seed + 1)
    selected = [selection_rng.choice(grouped[system]) for system in smoke_labels]

    sys.path.insert(0, str(args.flexicodec_repo))
    from funasr.utils import install_model_requirements

    install_model_requirements.install_requirements = lambda _path: None
    from flexicodec.infer import encode_flexicodec, prepare_model

    model_dict = prepare_model(
        sensevoice_small_path=str(args.model_root / "SenseVoiceSmall"),
        device=str(device),
        ckpt_path=str(args.model_root / "flexicodec" / "12hz_v1_half.safetensors"),
        config_path=str(args.model_root / "flexicodec" / "12hz_v1_half_config.yaml"),
    )
    codec = model_dict["model"]
    codec.eval()

    prepared = []
    mapping = []
    for row in selected:
        reference = load_audio(Path(row["audio"])).to(device)
        provider = AlignmentProvider(row, args.max_span)
        with torch.inference_mode(), external_alignment(codec, provider):
            encoded = encode_flexicodec(
                reference.unsqueeze(0),
                model_dict,
                16000,
                num_quantizers=8,
                merging_threshold=0.91,
                audio_lens=[len(reference)],
            )
        if provider.result is None or provider.result["source_segments"] != provider.result["mapped_segments"]:
            raise RuntimeError("codec-grid mapping did not preserve the token budget")
        acoustic = encoded.get("acoustic_codes")
        books = codebook_count(encoded["semantic_codes"], acoustic)
        if books != 8:
            raise RuntimeError(
                "num_quantizers=8 did not produce eight codebooks: "
                f"semantic={tuple(encoded['semantic_codes'].shape)}, "
                f"acoustic={None if acoustic is None else tuple(acoustic.shape)}"
            )
        prepared.append(
            {
                "row": row,
                "reference": reference,
                "semantic_codes": encoded["semantic_codes"].detach().clone(),
                "acoustic_codes": None if acoustic is None else acoustic.detach().clone(),
                "token_lengths": encoded["token_lengths"].detach().clone(),
            }
        )
        mapping.append({"utt_id": row["utt_id"], "system": row["system"], **provider.result})

    first = prepared[0]
    with torch.inference_mode():
        original = normalize_waveform(
            codec.decode_from_codes(
                first["semantic_codes"], first["acoustic_codes"], first["token_lengths"]
            )
        ).float()
    trainable_names = configure_trainable(codec)
    codec.eval()
    for module in decoder_modules(codec):
        module.eval()
    with torch.inference_mode():
        step_zero = normalize_waveform(
            codec.decode_from_codes(
                first["semantic_codes"], first["acoustic_codes"], first["token_lengths"]
            )
        ).float()
    step_zero_error = float((original - step_zero).abs().max())
    if step_zero_error > 1e-5:
        raise RuntimeError(f"step-zero decoder mismatch: {step_zero_error}")
    for module in decoder_modules(codec):
        module.train()

    generator_parameters = [parameter for parameter in codec.parameters() if parameter.requires_grad]
    frozen_trainable = sum(
        parameter.numel()
        for name, parameter in codec.named_parameters()
        if parameter.requires_grad and not name.startswith(TRAINABLE_PREFIXES)
    )
    discriminator = Discriminator(sample_rate=16000).to(device).train()
    gan = GANLoss(discriminator)
    spectral = MelSpectrogramLoss(
        pow=1.0,
        mag_weight=0.0,
        log_weight=2.0,
        n_mels=[5, 10, 20, 40, 80, 160, 320],
        window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
    ).to(device)
    optimizer_g = torch.optim.AdamW(generator_parameters, lr=1e-5, betas=(0.8, 0.9))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.8, 0.9))
    generator_before = snapshot(generator_parameters)
    discriminator_before = snapshot(discriminator.parameters())

    speaker = None
    speaker_references: dict[str, torch.Tensor] = {}
    if args.speaker_weight:
        sys.path.insert(0, str(Path(__file__).parent))
        from speaker_objective import FrozenSpeakerEmbedder

        speaker = FrozenSpeakerEmbedder(args.speaker_model, device, 64000)
        unique = {item["row"]["utt_id"]: item["reference"] for item in prepared}
        for utt_id, reference in unique.items():
            with torch.inference_mode():
                speaker_references[utt_id] = speaker.embeddings(
                    reference.unsqueeze(0), torch.tensor([len(reference)], device=device)
                )[0].half().cpu()

    history = []
    started = time.perf_counter()
    for step in range(args.steps):
        items = prepared[step * args.batch_size : (step + 1) * args.batch_size]
        generated = []
        for item in items:
            # The original codec trainer uses A100 bfloat16 mixed precision.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                estimate = codec.decode_from_codes(
                    item["semantic_codes"], item["acoustic_codes"], item["token_lengths"]
                )
            estimate = normalize_waveform(estimate).float()
            reference = item["reference"].reshape(1, 1, -1).float()
            width = min(estimate.shape[-1], reference.shape[-1])
            generated.append((item, estimate[..., :width], reference[..., :width]))

        optimizer_d.zero_grad(set_to_none=True)
        discriminator_loss = sum(
            gan.discriminator_loss(
                AudioSignal(fake.detach(), 16000), AudioSignal(real, 16000)
            )
            for _, fake, real in generated
        ) / args.batch_size
        if not torch.isfinite(discriminator_loss):
            raise RuntimeError("non-finite discriminator loss")
        discriminator_loss.backward()
        discriminator_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), 1.0, error_if_nonfinite=True
            )
        )
        optimizer_d.step()

        for parameter in discriminator.parameters():
            parameter.requires_grad = False
        optimizer_g.zero_grad(set_to_none=True)
        adversarial = torch.zeros((), device=device)
        feature = torch.zeros((), device=device)
        reconstruction = torch.zeros((), device=device)
        speaker_loss = torch.zeros((), device=device)
        for item, fake, real in generated:
            current_adversarial, current_feature = gan.generator_loss(
                AudioSignal(fake, 16000), AudioSignal(real, 16000)
            )
            current_reconstruction = spectral(AudioSignal(real, 16000), AudioSignal(fake, 16000))
            adversarial = adversarial + current_adversarial / args.batch_size
            feature = feature + current_feature / args.batch_size
            reconstruction = reconstruction + current_reconstruction / args.batch_size
            if speaker is not None:
                speaker_loss = speaker_loss + speaker.loss(
                    fake[:, 0],
                    torch.tensor([fake.shape[-1]], device=device),
                    [str(item["row"]["utt_id"])],
                    speaker_references,
                ) / args.batch_size
        generator_loss = adversarial + 2.0 * feature + 15.0 * reconstruction
        total = generator_loss + args.speaker_weight * speaker_loss
        if not torch.isfinite(total):
            raise RuntimeError("non-finite generator loss")
        total.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                generator_parameters, 1.0, error_if_nonfinite=True
            )
        )
        optimizer_g.step()
        for parameter in discriminator.parameters():
            parameter.requires_grad = True
        history.append(
            {
                "step": step + 1,
                "systems": [item["row"]["system"] for item in items],
                "discriminator": float(discriminator_loss.detach()),
                "discriminator_gradient_norm": discriminator_gradient_norm,
                "adversarial": float(adversarial.detach()),
                "feature_matching": float(feature.detach()),
                "reconstruction": float(reconstruction.detach()),
                "codec_generator": float(generator_loss.detach()),
                "speaker": float(speaker_loss.detach()),
                "total": float(total.detach()),
                "generator_gradient_norm": gradient_norm,
            }
        )
        print(json.dumps(history[-1]), flush=True)

    acoustic_shapes = [
        None if item["acoustic_codes"] is None else list(item["acoustic_codes"].shape)
        for item in prepared
    ]
    result = {
        "status": "passed",
        "purpose": "two_step_smoke_only_not_for_selection",
        "speaker_weight": args.speaker_weight,
        "mixed_precision": "bfloat16",
        "discriminator_initialization": "new_random_seed_42_no_checkpoint_keys_available",
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "num_quantizers_requested": 8,
        "codebook_counts": [
            codebook_count(item["semantic_codes"], item["acoustic_codes"])
            for item in prepared
        ],
        "semantic_code_shapes": [list(item["semantic_codes"].shape) for item in prepared],
        "acoustic_code_shapes": acoustic_shapes,
        "formal_10000_step_system_counts": formal_counts,
        "formal_sampler_max_count_difference": max(formal_counts.values()) - min(formal_counts.values()),
        "smoke_systems": smoke_labels,
        "mapping": mapping,
        "step_zero_max_abs_error": step_zero_error,
        "codec_trainable_parameter_count": sum(parameter.numel() for parameter in generator_parameters),
        "unexpected_frozen_codec_trainable_parameters": frozen_trainable,
        "trainable_parameter_tensors": len(trainable_names),
        "discriminator_parameter_count": sum(parameter.numel() for parameter in discriminator.parameters()),
        "generator_max_parameter_delta": max_parameter_delta(generator_before, generator_parameters),
        "discriminator_max_parameter_delta": max_parameter_delta(
            discriminator_before, discriminator.parameters()
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    numeric = [
        result["step_zero_max_abs_error"],
        result["generator_max_parameter_delta"],
        result["discriminator_max_parameter_delta"],
        *[
            value
            for row in history
            for key, value in row.items()
            if key not in {"step", "systems"}
        ],
    ]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise RuntimeError("smoke result contains non-finite values")
    if result["generator_max_parameter_delta"] <= 0 or result["discriminator_max_parameter_delta"] <= 0:
        raise RuntimeError("generator or discriminator did not update")
    atomic_json(result, args.output_dir / "smoke.json")
    (args.output_dir / "smoke.exit").write_text("0\n", encoding="ascii")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
