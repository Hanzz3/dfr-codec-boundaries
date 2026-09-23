#!/usr/bin/env python3
"""Train the shared Qwen2.5-0.5B direct q1 ASR probe."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from core import (
    ASRManifestDataset,
    SUPPORTED_CHECKPOINT_FORMATS,
    BalancedSystemBatchSampler,
    build_probe,
    build_training_tensors,
    cosine_schedule,
    identity_collate,
    load_clean_state_dict,
    save_checkpoint,
    trainable_parameter_audit,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def fixed_validation_records(
    dataset: ASRManifestDataset,
    per_system: int,
    systems: tuple[str, ...],
) -> list[int]:
    selected = {system: [] for system in systems}
    for index, record in enumerate(dataset.records):
        system = record["system"]
        if system in selected and len(selected[system]) < per_system:
            selected[system].append(index)
    missing = {
        system: len(values)
        for system, values in selected.items()
        if len(values) != per_system
    }
    if missing:
        raise RuntimeError(f"insufficient validation rows: {missing}")
    return [index for system in systems for index in selected[system]]


@torch.no_grad()
def validate(
    probe,
    tokenizer,
    dataset: ASRManifestDataset,
    indexes: list[int],
    *,
    batch_size: int,
    device: torch.device,
    max_text_tokens: int,
    systems: tuple[str, ...],
) -> tuple[float, dict[str, float]]:
    probe.eval()
    losses = []
    by_system = {system: [] for system in systems}
    for offset in range(0, len(indexes), batch_size):
        items = [dataset[index] for index in indexes[offset : offset + batch_size]]
        inputs, attention, labels = build_training_tensors(
            probe,
            tokenizer,
            items,
            device=device,
            max_text_tokens=max_text_tokens,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = probe.qwen(
                inputs_embeds=inputs,
                attention_mask=attention,
                labels=labels,
                use_cache=False,
            )
        value = float(output.loss)
        losses.append(value)
        for item in items:
            by_system[item["system"]].append(value)
    probe.train()
    return float(np.mean(losses)), {
        system: float(np.mean(values)) for system, values in by_system.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=0,
        help="Split each effective batch into smaller forward/backward passes; 0 disables splitting.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--validation-per-system", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-text-tokens", type=int, default=192)
    parser.add_argument("--projector-lr", type=float, default=2e-4)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--system",
        action="append",
        help="Train the selected merger. Repeat for a subset; default is every system in the manifest.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    micro_batch_size = args.micro_batch_size or args.batch_size
    if not 1 <= micro_batch_size <= args.batch_size:
        raise ValueError(
            f"micro_batch_size must be in [1, batch_size], got "
            f"{micro_batch_size} for batch_size={args.batch_size}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    train_dataset = ASRManifestDataset(args.train_manifest)
    validation_dataset = ASRManifestDataset(args.validation_manifest)
    manifest_systems = tuple(dict.fromkeys(row["system"] for row in train_dataset.records))
    systems = tuple(args.system or manifest_systems)
    if len(systems) != len(set(systems)):
        raise RuntimeError(f"duplicate systems: {systems}")
    feature_dims = {int(row["feature_dim"]) for row in train_dataset.records}
    if len(feature_dims) != 1:
        raise RuntimeError(f"mixed input dimensions: {feature_dims}")
    input_dim = feature_dims.pop()

    probe, tokenizer = build_probe(
        args.model,
        input_dim=input_dim,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=device,
    )
    audit = trainable_parameter_audit(probe)
    (args.output_dir / "trainable_parameters.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        **vars(args),
        "model": str(args.model),
        "train_manifest": str(args.train_manifest),
        "validation_manifest": str(args.validation_manifest),
        "output_dir": str(args.output_dir),
        "systems": list(systems),
        "input_dim": input_dim,
        "device": str(device),
        "effective_batch_size": args.batch_size,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": int(
            np.ceil(args.batch_size / micro_batch_size)
        ),
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )

    projector_parameters = list(probe.projector.parameters())
    lora_parameters = [
        parameter
        for name, parameter in probe.named_parameters()
        if parameter.requires_grad and not name.startswith("projector.")
    ]
    optimizer = torch.optim.AdamW(
        (
            {"params": projector_parameters, "lr": args.projector_lr},
            {"params": lora_parameters, "lr": args.lora_lr},
        ),
        weight_decay=args.weight_decay,
    )
    warmup_steps = int(round(args.max_steps * args.warmup_fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(step, args.max_steps, warmup_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_step = 0
    best_validation = float("inf")
    history: list[dict] = []
    if args.resume:
        history_path = args.output_dir / "training_history.csv"
        if history_path.is_file():
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("format") not in SUPPORTED_CHECKPOINT_FORMATS:
            raise RuntimeError("invalid resume checkpoint")
        load_clean_state_dict(probe, payload["trainable_state"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        start_step = int(payload["step"])
        best_validation = float(payload["best_validation_loss"])

    sampler = BalancedSystemBatchSampler(
        train_dataset.records,
        systems=systems,
        batch_size=args.batch_size,
        steps=args.max_steps,
        seed=args.seed,
        start_step=start_step,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    validation_indexes = fixed_validation_records(
        validation_dataset,
        args.validation_per_system,
        systems,
    )

    probe.train()
    running_losses = []
    started = time.perf_counter()
    last_validation = float("nan")
    for step, items in enumerate(loader, start=start_step + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for offset in range(0, len(items), micro_batch_size):
            micro_items = items[offset : offset + micro_batch_size]
            inputs, attention, labels = build_training_tensors(
                probe,
                tokenizer,
                micro_items,
                device=device,
                max_text_tokens=args.max_text_tokens,
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = probe.qwen(
                    inputs_embeds=inputs,
                    attention_mask=attention,
                    labels=labels,
                    use_cache=False,
                )
                loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            weight = len(micro_items) / len(items)
            scaler.scale(loss * weight).backward()
            step_loss += float(loss.detach()) * weight
            del output, loss, inputs, attention, labels
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in probe.parameters() if parameter.requires_grad),
            args.gradient_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_losses.append(step_loss)

        should_eval = step % args.eval_every == 0 or step == args.max_steps
        if should_eval:
            last_validation, per_system = validate(
                probe,
                tokenizer,
                validation_dataset,
                validation_indexes,
                batch_size=micro_batch_size,
                device=device,
                max_text_tokens=args.max_text_tokens,
                systems=systems,
            )
            row = {
                "step": step,
                "train_loss_window": float(np.mean(running_losses)),
                "validation_loss": last_validation,
                "learning_rate_projector": optimizer.param_groups[0]["lr"],
                "learning_rate_lora": optimizer.param_groups[1]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            for system, value in per_system.items():
                row[f"validation_{system}"] = value
            history.append(row)
            write_history(args.output_dir / "training_history.csv", history)
            print(json.dumps(row, sort_keys=True), flush=True)
            running_losses.clear()
            if last_validation < best_validation:
                best_validation = last_validation
                save_checkpoint(
                    args.output_dir / "best.pt",
                    probe,
                    step=step,
                    best_validation_loss=best_validation,
                    validation_loss=last_validation,
                    config=config,
                )
            if step in {1000, 3000, 5000, 10000, 20000, 30000}:
                save_checkpoint(
                    args.output_dir / f"step_{step:06d}.pt",
                    probe,
                    step=step,
                    best_validation_loss=best_validation,
                    validation_loss=last_validation,
                    config=config,
                )
            save_checkpoint(
                args.output_dir / "last_state.pt",
                probe,
                step=step,
                best_validation_loss=best_validation,
                validation_loss=last_validation,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
        elif step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_loss": float(np.mean(running_losses)),
                        "system": items[0]["system"],
                        "seconds_per_step": (time.perf_counter() - started)
                        / max(1, step - start_step),
                    }
                ),
                flush=True,
            )

    save_checkpoint(
        args.output_dir / "last.pt",
        probe,
        step=args.max_steps,
        best_validation_loss=best_validation,
        validation_loss=last_validation,
        config=config,
    )
    (args.output_dir / "train.exit").write_text("0\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
