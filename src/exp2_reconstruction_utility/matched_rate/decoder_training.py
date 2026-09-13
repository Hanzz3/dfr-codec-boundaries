#!/usr/bin/env python3
"""Fine-tune only the FlexiCodec decoder stack from fixed algorithm-specific codes."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from io_utils import load_audio, read_jsonl
from flexicodec_adapter import prepare_pinned_flexicodec


TRAINABLE_PREFIXES = ("convnext_decoder.", "bottleneck_transformer.", "dac.decoder.")


def decoder_modules(codec: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    return codec.convnext_decoder, codec.bottleneck_transformer, codec.dac.decoder


def set_decoder_training_mode(codec: torch.nn.Module, training: bool) -> None:
    """Keep the frozen codec in eval mode and toggle only trainable decoder modules."""
    codec.eval()
    for module in decoder_modules(codec):
        module.train(training)


class CodeDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        codes = torch.load(row["code_path"], map_location="cpu", weights_only=True)
        reference = torch.from_numpy(load_audio(row["audio"])).float()
        return {"row": row, "codes": codes, "reference": reference}


def pad_last(tensors: list[torch.Tensor], value: int = 0) -> torch.Tensor:
    width = max(item.shape[-1] for item in tensors)
    shape = (len(tensors), *tensors[0].shape[:-1], width)
    output = tensors[0].new_full(shape, value)
    for index, tensor in enumerate(tensors):
        output[index, ..., : tensor.shape[-1]] = tensor
    return output


def collate(batch: list[dict]) -> dict:
    semantic = pad_last([item["codes"]["semantic_codes"] for item in batch])
    acoustic_items = [item["codes"]["acoustic_codes"] for item in batch]
    if any(item is None for item in acoustic_items):
        acoustic = None
    else:
        acoustic = pad_last(acoustic_items)
    token_lengths = pad_last([item["codes"]["token_lengths"].unsqueeze(0) for item in batch])[:, 0]
    sample_lengths = torch.tensor([len(item["reference"]) for item in batch], dtype=torch.long)
    references = torch.zeros(len(batch), int(sample_lengths.max()), dtype=torch.float32)
    for index, item in enumerate(batch):
        references[index, : len(item["reference"])] = item["reference"]
    return {
        "semantic_codes": semantic,
        "acoustic_codes": acoustic,
        "token_lengths": token_lengths,
        "references": references,
        "sample_lengths": sample_lengths,
        "rows": [item["row"] for item in batch],
    }


def stft_loss(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    losses = []
    for n_fft in (512, 1024, 2048):
        if reference.numel() < n_fft:
            continue
        window = torch.hann_window(n_fft, device=reference.device, dtype=reference.dtype)
        ref = torch.stft(reference, n_fft, hop_length=n_fft // 4, window=window, return_complex=True).abs()
        est = torch.stft(estimate, n_fft, hop_length=n_fft // 4, window=window, return_complex=True).abs()
        spectral_convergence = torch.linalg.vector_norm(est - ref) / torch.linalg.vector_norm(ref).clamp_min(1e-6)
        log_magnitude = F.l1_loss(torch.log(est + 1e-5), torch.log(ref + 1e-5))
        losses.append(spectral_convergence + log_magnitude)
    return torch.stack(losses).mean() if losses else reference.new_zeros(())


def log_mel_loss(reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    import torchaudio

    key = (str(reference.device), str(reference.dtype), sample_rate)
    cache = getattr(log_mel_loss, "_cache", {})
    if key not in cache:
        cache[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=160,
            n_mels=80,
            power=1.0,
        ).to(reference.device, reference.dtype)
        log_mel_loss._cache = cache
    transform = cache[key]
    ref = torch.log(transform(reference.unsqueeze(0)).clamp_min(1e-5))
    est = torch.log(transform(estimate.unsqueeze(0)).clamp_min(1e-5))
    return F.l1_loss(est, ref)


def reconstruction_loss(
    output: torch.Tensor,
    references: torch.Tensor,
    lengths: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    values = {"waveform": [], "stft": [], "mel": []}
    for index, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        estimate = output[index].reshape(-1)
        if len(estimate) < length:
            estimate = F.pad(estimate, (0, length - len(estimate)))
        estimate = estimate[:length]
        reference = references[index, :length]
        values["waveform"].append(F.l1_loss(estimate, reference))
        values["stft"].append(stft_loss(reference.float(), estimate.float()))
        values["mel"].append(log_mel_loss(reference.float(), estimate.float()))
    means = {name: torch.stack(items).mean() for name, items in values.items()}
    total = sum(float(weights[name]) * means[name] for name in means)
    return total, {name: float(value.detach()) for name, value in means.items()}


def configure_trainable(codec: torch.nn.Module) -> list[str]:
    for parameter in codec.parameters():
        parameter.requires_grad = False
    for module in decoder_modules(codec):
        for parameter in module.parameters():
            parameter.requires_grad = True
    set_decoder_training_mode(codec, True)
    names = [name for name, parameter in codec.named_parameters() if parameter.requires_grad]
    invalid = [name for name in names if not name.startswith(TRAINABLE_PREFIXES)]
    if invalid or not names:
        raise RuntimeError(f"unexpected trainable parameters: {invalid}; count={len(names)}")
    return names


def decoder_delta(codec: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "convnext_decoder": {key: value.detach().cpu() for key, value in codec.convnext_decoder.state_dict().items()},
        "bottleneck_transformer": {
            key: value.detach().cpu() for key, value in codec.bottleneck_transformer.state_dict().items()
        },
        "dac_decoder": {key: value.detach().cpu() for key, value in codec.dac.decoder.state_dict().items()},
    }


def load_decoder_delta(codec: torch.nn.Module, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    codec.convnext_decoder.load_state_dict(payload["modules"]["convnext_decoder"])
    codec.bottleneck_transformer.load_state_dict(payload["modules"]["bottleneck_transformer"])
    codec.dac.decoder.load_state_dict(payload["modules"]["dac_decoder"])


def run_epoch(codec, loader, device, weights, optimizer=None, scaler=None, scheduler=None, clip=1.0):  # noqa: ANN001
    training = optimizer is not None
    set_decoder_training_mode(codec, training)
    totals = []
    components = []
    started = time.perf_counter()
    for batch in loader:
        semantic = batch["semantic_codes"].to(device)
        acoustic = None if batch["acoustic_codes"] is None else batch["acoustic_codes"].to(device)
        token_lengths = batch["token_lengths"].to(device)
        references = batch["references"].to(device)
        lengths = batch["sample_lengths"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            output = codec.decode_from_codes(semantic, acoustic, token_lengths)
            loss, parts = reconstruction_loss(output, references, lengths, weights)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite decoder loss")
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([parameter for parameter in codec.parameters() if parameter.requires_grad], clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        totals.append(float(loss.detach()))
        components.append(parts)
    metrics = {"loss": float(np.mean(totals)), "seconds": time.perf_counter() - started}
    for name in ("waveform", "stft", "mel"):
        metrics[name] = float(np.mean([row[name] for row in components]))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-index", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flexicodec-repo", type=Path, default=Path("./repos/FlexiCodec"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-utterances", type=int)
    parser.add_argument("--max-val-utterances", type=int)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(args.flexicodec_repo))
    rows = [row for row in read_jsonl(args.code_index) if row["system"] == args.system]
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if args.max_train_utterances:
        train_rows = train_rows[: args.max_train_utterances]
    if args.max_val_utterances:
        val_rows = val_rows[: args.max_val_utterances]
    if not train_rows or not val_rows:
        raise RuntimeError(f"empty train/validation code cache: train={len(train_rows)}, val={len(val_rows)}")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        CodeDataset(train_rows), batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        CodeDataset(val_rows), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=device.type == "cuda",
    )
    model_dict = prepare_pinned_flexicodec(args.flexicodec_repo, str(device))
    codec = model_dict["model"]
    trainable_names = configure_trainable(codec)
    trainable = [parameter for parameter in codec.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["decoder_training"]["learning_rate"]),
        weight_decay=float(config["decoder_training"]["weight_decay"]),
    )
    total_steps = max(1, args.max_epochs * len(train_loader))
    warmup = max(1, int(round(total_steps * float(config["decoder_training"]["warmup_fraction"]))))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    weights = {key: float(value) for key, value in config["decoder_training"]["loss_weights"].items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trainable_parameters.json").write_text(
        json.dumps({"count": len(trainable_names), "names": trainable_names}, indent=2) + "\n", encoding="utf-8"
    )
    history = []
    best = math.inf
    stale = 0
    for epoch in range(1, args.max_epochs + 1):
        train_metrics = run_epoch(
            codec, train_loader, device, weights, optimizer, scaler, scheduler,
            float(config["decoder_training"]["gradient_clip"]),
        )
        with torch.no_grad():
            val_metrics = run_epoch(codec, val_loader, device, weights)
        row = {
            "system": args.system,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "decoder_training_history.csv", index=False)
        payload = {
            "system": args.system,
            "epoch": epoch,
            "validation_loss": val_metrics["loss"],
            "modules": decoder_delta(codec),
            "config": config,
        }
        torch.save(payload, args.output_dir / "last.pt")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            stale = 0
            torch.save(payload, args.output_dir / "best.pt")
        else:
            stale += 1
        print(json.dumps(row), flush=True)
        if stale >= args.patience:
            break
    (args.output_dir / "train.exit").write_text("0\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
