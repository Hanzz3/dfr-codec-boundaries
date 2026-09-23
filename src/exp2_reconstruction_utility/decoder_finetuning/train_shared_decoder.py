#!/usr/bin/env python3
"""Train the decoder with the historical nine-allocation GAN protocol."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from audiotools import AudioSignal
from dac.model import Discriminator
from dac.nn.loss import GANLoss, MelSpectrogramLoss

from codec_interface import (
    DYNAMIC_SYSTEMS,
    SYSTEMS,
    alignment_provider,
    atomic_json,
    codebook_count,
    configure_trainable,
    decoder_modules,
    external_alignment,
    load_audio,
    normalize_waveform,
    scheduled_systems,
)
from paper_protocol import DECODER_TRAINING_RATES_HZ


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="ascii")
    temporary.replace(path)


def manifest_handle(path: Path):  # noqa: ANN201
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_audio(row: dict, audio_root: Path) -> Path:
    original = Path(row["audio"])
    if original.is_file():
        return original
    source_split = str(row["source_split"])
    parts = original.parts
    if source_split not in parts:
        raise RuntimeError(f"cannot map audio path for {row['utt_id']}: {original}")
    index = parts.index(source_split)
    local = audio_root / source_split / Path(*parts[index + 1 :])
    return local


def rate_tag(rate: float) -> str:
    return f"{rate:g}".replace(".", "p")

RATES = DECODER_TRAINING_RATES_HZ
CONDITIONS = tuple(f"{system}@{rate_tag(rate)}hz" for rate in RATES for system in SYSTEMS)

def scheduled_conditions(seed: int, max_steps: int, batch_size: int) -> list[str]:
    count = max_steps * batch_size
    labels = list(CONDITIONS) * (count // len(CONDITIONS)) + list(CONDITIONS[: count % len(CONDITIONS)])
    random.Random(seed).shuffle(labels)
    return labels

def balanced_sample(
    manifest: Path,
    labels: list[str],
    audio_root: Path,
    seed: int,
    excluded_keys: set[tuple[str, str]] | None = None,
) -> tuple[list[dict], dict[str, int], dict[str, int], dict[str, int], dict[str, int], list[dict]]:
    excluded_keys = excluded_keys or set()
    required = {condition: labels.count(condition) for condition in CONDITIONS}
    manifest_counts = {condition: 0 for condition in CONDITIONS}
    eligible_counts = {condition: 0 for condition in CONDITIONS}
    available_counts = {condition: 0 for condition in CONDITIONS}
    seen = {condition: set() for condition in CONDITIONS}
    excluded = []
    reservoirs = {condition: [] for condition in CONDITIONS}
    rngs = {condition: random.Random(seed + 1000 + i) for i, condition in enumerate(CONDITIONS)}
    with manifest_handle(manifest) as handle:
        for line in handle:
            if not line.strip(): continue
            row = json.loads(line)
            system = str(row.get("system")); rate = float(row.get("target_rate_hz"))
            condition = str(row.get("condition_key", f"{system}@{rate_tag(rate)}hz"))
            if condition not in reservoirs: raise RuntimeError(f"unexpected condition in train manifest: {condition}")
            utt_id = str(row["utt_id"])
            if utt_id in seen[condition]: raise RuntimeError(f"duplicate {condition}/{utt_id}")
            seen[condition].add(utt_id); manifest_counts[condition] += 1
            codec_capacity = max(1, int(int(row["frames"]) / 1.33333))
            segment_count = len(row["starts"])
            if segment_count > codec_capacity:
                excluded.append({"condition": condition, "utt_id": utt_id, "source_frames": int(row["frames"]), "segments": segment_count, "codec_capacity": codec_capacity, "excess_segments": segment_count - codec_capacity})
                continue
            eligible_counts[condition] += 1
            if (condition, utt_id) in excluded_keys: continue
            available_counts[condition] += 1
            local = dict(row); local["audio"] = str(resolve_audio(row, audio_root))
            reservoir = reservoirs[condition]; need = required[condition]
            if len(reservoir) < need: reservoir.append(local)
            elif need:
                replacement = rngs[condition].randrange(available_counts[condition])
                if replacement < need: reservoir[replacement] = local
    for condition in CONDITIONS:
        if manifest_counts[condition] == 0: raise RuntimeError(f"manifest missing condition: {condition}")
        if len(reservoirs[condition]) != required[condition]: raise RuntimeError(f"insufficient rows for {condition}: {len(reservoirs[condition])}/{required[condition]}")
        random.Random(seed + 2000 + CONDITIONS.index(condition)).shuffle(reservoirs[condition])
    cursors = {condition: 0 for condition in CONDITIONS}; selected = []
    for condition in labels:
        selected.append(reservoirs[condition][cursors[condition]]); cursors[condition] += 1
    return selected, manifest_counts, eligible_counts, available_counts, required, excluded

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generator_state(codec: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "convnext_decoder": codec.convnext_decoder.state_dict(),
        "bottleneck_transformer": codec.bottleneck_transformer.state_dict(),
        "dac_decoder": codec.dac.decoder.state_dict(),
    }


def load_generator_state(codec: torch.nn.Module, state: dict) -> None:
    codec.convnext_decoder.load_state_dict(state["convnext_decoder"], strict=True)
    codec.bottleneck_transformer.load_state_dict(
        state["bottleneck_transformer"], strict=True
    )
    codec.dac.decoder.load_state_dict(state["dac_decoder"], strict=True)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])


def atomic_checkpoint(payload: dict, milestone: Path, last_state: Path) -> None:
    milestone.parent.mkdir(parents=True, exist_ok=True)
    temporary = milestone.with_name(milestone.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(milestone)
    link = last_state.with_name(last_state.name + f".tmp.{os.getpid()}")
    if link.exists():
        link.unlink()
    os.link(milestone, link)
    link.replace(last_state)


def checkpoint_payload(
    *,
    step: int,
    codec: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    metadata: dict,
) -> dict:
    return {
        "schema_version": 1,
        "step": step,
        "metadata": metadata,
        "generator": generator_state(codec),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "rng": rng_state(),
    }


def append_jsonl(payload: dict, path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def encode_item(row: dict, codec, model_dict: dict, max_span: int, device):  # noqa: ANN001, ANN201
    from flexicodec.infer import encode_flexicodec

    reference = load_audio(Path(row["audio"])).to(device)
    provider = alignment_provider(row, max_span)
    with torch.inference_mode(), external_alignment(codec, provider):
        encoded = encode_flexicodec(
            reference.unsqueeze(0),
            model_dict,
            16000,
            num_quantizers=8,
            merging_threshold=0.91,
            audio_lens=[len(reference)],
        )
    if provider.result is None:
        raise RuntimeError("missing codec-grid mapping result")
    if provider.result["source_segments"] != provider.result["mapped_segments"]:
        raise RuntimeError("codec-grid mapping changed the token count")
    acoustic = encoded.get("acoustic_codes")
    if codebook_count(encoded["semantic_codes"], acoustic) != 8:
        raise RuntimeError("formal training did not produce eight codebooks")
    return {
        "row": row,
        "reference": reference,
        "semantic_codes": encoded["semantic_codes"].detach(),
        "acoustic_codes": None if acoustic is None else acoustic.detach(),
        "token_lengths": encoded["token_lengths"].detach(),
        "mapping": provider.result,
    }


def distributed_context() -> tuple[int, int, int, bool]:
    """Initialize torchrun process group when requested."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        dist.init_process_group(backend="nccl", init_method="env://")
        if rank < 0 or rank >= world_size:
            raise RuntimeError(f"invalid rank {rank}/{world_size}")
    return rank, world_size, local_rank, world_size > 1


def sync_gradients(parameters, world_size: int) -> None:
    if world_size <= 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(world_size))


def sync_scalar(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1:
        return value
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value.div_(float(world_size))
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flexicodec-repo", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--speaker-model", type=Path, required=True)
    parser.add_argument("--speaker-weight", type=float, choices=(0.0, 0.1), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-span", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--protocol-mode",
        choices=("formal", "formal_extension", "smoke"),
        default="formal",
    )
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--parent-step", type=int, default=10000)
    args = parser.parse_args()

    if args.batch_size != 4 or args.seed != 42:
        raise ValueError("formal_v2 requires seed42 and batch4")
    if args.protocol_mode == "formal" and (args.steps != 10000 or args.save_every != 500):
        raise ValueError("formal mode requires 10000 steps and save_every=500")
    if args.protocol_mode == "formal_extension":
        if args.steps != 30000 or args.parent_step != 10000 or args.save_every != 500:
            raise ValueError(
                "formal_extension requires parent_step=10000, steps=30000, and save_every=500"
            )
        if args.initialize_from is None or not args.initialize_from.is_file():
            raise ValueError("formal_extension requires an existing --initialize-from checkpoint")
    if args.protocol_mode == "smoke" and not (1 <= args.steps <= 12):
        raise ValueError("smoke mode permits 1-12 steps")
    if args.protocol_mode != "formal_extension" and args.initialize_from is not None:
        raise ValueError("--initialize-from is reserved for formal_extension")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rank, world_size, local_rank, distributed = distributed_context()
    if distributed:
        if args.batch_size % world_size != 0:
            raise ValueError("global batch size must be divisible by WORLD_SIZE")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    local_batch_size = args.batch_size // world_size
    loss_divisor = local_batch_size if distributed else args.batch_size
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(device)
    set_seed(args.seed)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]

    parent_checkpoint_sha256 = None
    parent_exposure_counts = {condition: 0 for condition in CONDITIONS}
    parent_keys: set[tuple[str, str]] = set()
    selection_seed = args.seed
    selection_start_step = 0
    if args.protocol_mode == "formal_extension":
        parent_labels = scheduled_conditions(args.seed, args.parent_step, args.batch_size)
        (
            parent_selected,
            _,
            _,
            _,
            parent_exposure_counts,
            _,
        ) = balanced_sample(args.manifest, parent_labels, args.audio_root, args.seed)
        parent_keys = {
            (str(row.get("condition_key")), str(row["utt_id"])) for row in parent_selected
        }
        if len(parent_keys) != len(parent_selected):
            raise RuntimeError("parent multirate sampler contains duplicate condition/utterance keys")
        selection_start_step = args.parent_step
        extension_steps = args.steps - args.parent_step
        label_seed = args.seed + args.parent_step
        selection_seed = args.seed + 2 * args.parent_step
        labels = scheduled_conditions(label_seed, extension_steps, args.batch_size)
        parent_checkpoint_sha256 = file_sha256(args.initialize_from)
    else:
        labels = scheduled_conditions(args.seed, args.steps, args.batch_size)

    (
        selected,
        manifest_counts,
        eligible_counts,
        available_counts,
        exposure_counts,
        excluded,
    ) = balanced_sample(
        args.manifest,
        labels,
        args.audio_root,
        selection_seed,
        excluded_keys=parent_keys,
    )
    manifest_sha256 = file_sha256(args.manifest)
    coverage = {
        condition: exposure_counts[condition] / max(available_counts[condition], 1)
        for condition in CONDITIONS
    }
    sampler = {
        "status": "passed",
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "manifest_counts": manifest_counts,
        "codec_grid_eligible_counts": eligible_counts,
        "available_after_parent_exclusion": available_counts,
        "codec_grid_excluded_count": len(excluded),
        "codec_grid_excluded": excluded,
        "codec_grid_eligibility_rule": "segments<=max(1,int(source_frames/1.33333))",
        "exposure_counts": exposure_counts,
        "unique_exposures": exposure_counts,
        "coverage_fraction": coverage,
        "max_exposure_difference": max(exposure_counts.values())
        - min(exposure_counts.values()),
        "selection": "deterministic_per_system_reservoir_without_replacement",
        "selection_seed": selection_seed,
        "label_seed": (
            args.seed + args.parent_step
            if args.protocol_mode == "formal_extension"
            else args.seed
        ),
        "selection_start_step": selection_start_step,
        "parent_exposure_counts": parent_exposure_counts,
        "parent_overlap_count": sum(
            (str(row.get("condition_key")), str(row["utt_id"])) in parent_keys
            for row in selected
        ),
    }
    if sampler["max_exposure_difference"] > 1:
        raise RuntimeError(f"unbalanced condition sampler: {exposure_counts}")
    if sampler["parent_overlap_count"]:
        raise RuntimeError("formal extension reuses parent system/utterance records")
    if rank == 0:
        atomic_json(sampler, args.output_dir / "sampler_validation.json")
    if distributed:
        dist.barrier()

    sys.path.insert(0, str(args.flexicodec_repo))
    from funasr.utils import install_model_requirements

    install_model_requirements.install_requirements = lambda _path: None
    from flexicodec.infer import prepare_model

    model_dict = prepare_model(
        sensevoice_small_path=str(args.model_root / "SenseVoiceSmall"),
        device=str(device),
        ckpt_path=str(args.model_root / "flexicodec" / "12hz_v1_half.safetensors"),
        config_path=str(args.model_root / "flexicodec" / "12hz_v1_half_config.yaml"),
    )
    codec = model_dict["model"]
    codec.eval()

    first = encode_item(selected[0], codec, model_dict, args.max_span, device)
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

    generator_parameters = [p for p in codec.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in generator_parameters)
    if trainable_count != 148538162:
        raise RuntimeError(f"unexpected trainable parameter count: {trainable_count}")
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

    speaker = None
    speaker_references: dict[str, torch.Tensor] = {}
    if args.speaker_weight:
        sys.path.insert(0, str(Path(__file__).parent))
        from speaker_objective import FrozenSpeakerEmbedder

        speaker = FrozenSpeakerEmbedder(args.speaker_model, device, 64000)

    metadata = {
        "manifest_sha256": manifest_sha256,
        "speaker_weight": args.speaker_weight,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "max_span_codec_frames": args.max_span,
        "num_quantizers": 8,
        "generator_lr": 1e-5,
        "discriminator_lr": 1e-4,
        "mixed_precision": args.amp_dtype,
        "source_protocol": "accepted_formal_v2_immediate_full_gan_v1_multirate",
        "protocol_mode": args.protocol_mode,
        "multirate_conditions": list(CONDITIONS),
        "target_rates_hz": list(RATES),
        "v100_precision_adaptation": args.amp_dtype == "float16",
        "discriminator_initialization": "new_random_seed_42_no_public_checkpoint_keys",
        "trainable_parameters": trainable_count,
        "trainable_names": trainable_names,
        "loss": "adv+2*feature_matching+15*multiscale_logmel+speaker_weight*speaker_cosine",
        "distributed_world_size": world_size,
        "distributed_backend": "nccl" if distributed else "none",
    }
    if args.protocol_mode == "formal_extension":
        metadata.update(
            {
                "parent_step": args.parent_step,
                "parent_checkpoint": str(args.initialize_from.resolve()),
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
                "continuation_sampling": (
                    "balanced_without_replacement_excluding_exact_parent_exposures"
                ),
            }
        )
    if rank == 0:
        atomic_json(
            {"status": "passed", "step_zero_max_abs_error": step_zero_error, **metadata},
            args.output_dir / "trainable_validation.json",
        )
    if distributed:
        dist.barrier()

    milestones = args.output_dir / "milestones"
    last_state = args.output_dir / "last_state.pt"
    start_step = 0
    if last_state.exists():
        saved = torch.load(last_state, map_location=device, weights_only=False)
        if saved["metadata"] != metadata:
            raise RuntimeError("resume metadata does not match the formal protocol")
        load_generator_state(codec, saved["generator"])
        discriminator.load_state_dict(saved["discriminator"], strict=True)
        optimizer_g.load_state_dict(saved["optimizer_g"])
        optimizer_d.load_state_dict(saved["optimizer_d"])
        restore_rng_state(saved["rng"])
        start_step = int(saved["step"])
    elif args.protocol_mode == "formal_extension":
        saved = torch.load(args.initialize_from, map_location=device, weights_only=False)
        parent_metadata = saved.get("metadata", {})
        expected_parent = {
            "steps": args.parent_step,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "max_span_codec_frames": args.max_span,
            "speaker_weight": args.speaker_weight,
            "generator_lr": 1e-5,
            "discriminator_lr": 1e-4,
            "source_protocol": "accepted_formal_v2_immediate_full_gan_v1_multirate",
            "protocol_mode": "formal",
        }
        mismatches = {
            key: (parent_metadata.get(key), value)
            for key, value in expected_parent.items()
            if parent_metadata.get(key) != value
        }
        if int(saved.get("step", -1)) != args.parent_step or mismatches:
            raise RuntimeError(
                f"formal extension parent validation failed: step={saved.get('step')}, "
                f"metadata_mismatches={mismatches}"
            )
        load_generator_state(codec, saved["generator"])
        discriminator.load_state_dict(saved["discriminator"], strict=True)
        optimizer_g.load_state_dict(saved["optimizer_g"])
        optimizer_d.load_state_dict(saved["optimizer_d"])
        restore_rng_state(saved["rng"])
        start_step = args.parent_step
        if rank == 0:
            atomic_checkpoint(
                checkpoint_payload(
                    step=start_step,
                    codec=codec,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    metadata=metadata,
                ),
                milestones / f"step_{start_step:05d}.pt",
                last_state,
            )
        if distributed:
            dist.barrier()
    else:
        set_seed(args.seed + 3000)
        if rank == 0:
            atomic_checkpoint(
                checkpoint_payload(
                    step=0,
                    codec=codec,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    metadata=metadata,
                ),
                milestones / "step_00000.pt",
                last_state,
            )
        if distributed:
            dist.barrier()

    log_path = args.output_dir / "train.jsonl"
    if rank == 0 and log_path.exists() and start_step:
        retained = []
        with log_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip() and int(json.loads(line)["step"]) <= start_step:
                    retained.append(line)
        temporary = log_path.with_name(log_path.name + f".tmp.{os.getpid()}")
        temporary.write_text("".join(retained), encoding="utf-8")
        temporary.replace(log_path)
    started = time.perf_counter()
    max_mapping_error = 0.0
    max_mapping_span = 0
    for step in range(start_step, args.steps):
        selection_step = step - selection_start_step
        global_batch_rows = selected[
            selection_step * args.batch_size : (selection_step + 1) * args.batch_size
        ]
        if len(global_batch_rows) != args.batch_size:
            raise RuntimeError(
                f"incomplete batch at global step {step}: {len(global_batch_rows)}/{args.batch_size}"
            )
        if distributed:
            batch_rows = global_batch_rows[rank * local_batch_size : (rank + 1) * local_batch_size]
        else:
            batch_rows = global_batch_rows
        items = [encode_item(row, codec, model_dict, args.max_span, device) for row in batch_rows]
        generated = []
        for item in items:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                estimate = codec.decode_from_codes(
                    item["semantic_codes"], item["acoustic_codes"], item["token_lengths"]
                )
            estimate = normalize_waveform(estimate).float()
            reference = item["reference"].reshape(1, 1, -1).float()
            width = min(estimate.shape[-1], reference.shape[-1])
            generated.append((item, estimate[..., :width], reference[..., :width]))
            max_mapping_error = max(
                max_mapping_error, float(item["mapping"]["mapping_max_error_frames"])
            )
            max_mapping_span = max(
                max_mapping_span, int(item["mapping"]["mapped_max_span_frames"])
            )

        optimizer_d.zero_grad(set_to_none=True)
        discriminator_loss = torch.zeros((), device=device)
        for _, fake, real in generated:
            current_discriminator = gan.discriminator_loss(
                AudioSignal(fake.detach(), 16000), AudioSignal(real, 16000)
            ) / loss_divisor
            if not torch.isfinite(current_discriminator):
                raise RuntimeError("non-finite discriminator loss")
            current_discriminator.backward()
            discriminator_loss = discriminator_loss + current_discriminator.detach()
        sync_gradients(discriminator.parameters(), world_size)
        discriminator_loss = sync_scalar(discriminator_loss, world_size)
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
            current_adv, current_feature = gan.generator_loss(
                AudioSignal(fake, 16000), AudioSignal(real, 16000)
            )
            current_reconstruction = spectral(AudioSignal(real, 16000), AudioSignal(fake, 16000))
            current_speaker = torch.zeros((), device=device)
            if speaker is not None:
                utt_id = str(item["row"]["utt_id"])
                if utt_id not in speaker_references:
                    with torch.inference_mode():
                        speaker_references[utt_id] = speaker.embeddings(
                            real[:, 0], torch.tensor([real.shape[-1]], device=device)
                        )[0].half().cpu()
                current_speaker = speaker.loss(
                    fake[:, 0],
                    torch.tensor([fake.shape[-1]], device=device),
                    [utt_id],
                    speaker_references,
                )
            scaled_adv = current_adv / loss_divisor
            scaled_feature = current_feature / loss_divisor
            scaled_reconstruction = current_reconstruction / loss_divisor
            scaled_speaker = current_speaker / loss_divisor
            current_codec_generator = (
                scaled_adv + 2.0 * scaled_feature + 15.0 * scaled_reconstruction
            )
            current_total = current_codec_generator + args.speaker_weight * scaled_speaker
            if not torch.isfinite(current_total):
                raise RuntimeError("non-finite generator loss")
            current_total.backward()
            adversarial = adversarial + scaled_adv.detach()
            feature = feature + scaled_feature.detach()
            reconstruction = reconstruction + scaled_reconstruction.detach()
            speaker_loss = speaker_loss + scaled_speaker.detach()
        sync_gradients(generator_parameters, world_size)
        for metric in (adversarial, feature, reconstruction, speaker_loss):
            sync_scalar(metric, world_size)
        codec_generator = adversarial + 2.0 * feature + 15.0 * reconstruction
        total = codec_generator + args.speaker_weight * speaker_loss
        generator_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                generator_parameters, 1.0, error_if_nonfinite=True
            )
        )
        optimizer_g.step()
        for parameter in discriminator.parameters():
            parameter.requires_grad = True

        completed = step + 1
        elapsed = time.perf_counter() - started
        record = {
            "step": completed,
            "systems": [item["row"]["system"] for item in items],
            "discriminator": float(discriminator_loss.detach()),
            "discriminator_gradient_norm": discriminator_gradient_norm,
            "adversarial": float(adversarial.detach()),
            "feature_matching": float(feature.detach()),
            "reconstruction": float(reconstruction.detach()),
            "codec_generator": float(codec_generator.detach()),
            "speaker": float(speaker_loss.detach()),
            "total": float(total.detach()),
            "generator_gradient_norm": generator_gradient_norm,
            "steps_per_second_since_resume": (completed - start_step) / max(elapsed, 1e-9),
            "eta_seconds": (args.steps - completed)
            * elapsed
            / max(completed - start_step, 1),
            "max_mapping_error_frames": max_mapping_error,
            "max_mapping_span_frames": max_mapping_span,
        }
        numeric = [value for key, value in record.items() if key not in {"systems"}]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise RuntimeError("non-finite training record")
        if rank == 0:
            append_jsonl(record, log_path)
            print(json.dumps(record), flush=True)

        if completed % args.save_every == 0:
            if rank == 0:
                atomic_checkpoint(
                    checkpoint_payload(
                        step=completed,
                        codec=codec,
                        discriminator=discriminator,
                        optimizer_g=optimizer_g,
                        optimizer_d=optimizer_d,
                        metadata=metadata,
                    ),
                    milestones / f"step_{completed:05d}.pt",
                    last_state,
                )
            if distributed:
                dist.barrier()

    summary = {
        "status": "passed",
        "protocol": (
            "current_nine_formal_v2_immediate_full_gan_10k_to_30k_extension_v1"
            if args.protocol_mode == "formal_extension"
            else "current_nine_multirate_formal_v2_immediate_full_gan_v1"
        ),
        "completed_steps": args.steps,
        "speaker_weight": args.speaker_weight,
        "milestones": args.steps // args.save_every + 1,
        "max_mapping_error_frames": max_mapping_error,
        "max_mapping_span_frames": max_mapping_span,
        "sampler": sampler,
    }
    if rank == 0:
        atomic_json(summary, args.output_dir / "train_summary.json")
        atomic_text("0\n", args.output_dir / "train.exit")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
