#!/usr/bin/env python3
"""Greedy-decode a direct q1 ASR checkpoint and report corpus WER by algorithm."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from core import (
    ASRManifestDataset,
    PROMPT_AFTER,
    PROMPT_BEFORE,
    SYSTEMS,
    SUPPORTED_CHECKPOINT_FORMATS,
    build_probe,
    edit_counts,
    load_clean_state_dict,
    normalize_transcript,
    tokenize_text,
)


@torch.no_grad()
def greedy_decode(
    probe,
    tokenizer,
    item: dict,
    *,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    embedding = probe.qwen.get_input_embeddings()
    before_ids = tokenize_text(tokenizer, PROMPT_BEFORE, 64)
    after_ids = tokenize_text(tokenizer, PROMPT_AFTER, 32)
    audio = probe.projector(
        item["features"].to(device), item["segment_lengths"].to(device)
    ).to(dtype=embedding.weight.dtype)
    before = embedding(torch.tensor(before_ids, device=device))
    after = embedding(torch.tensor(after_ids, device=device))
    prefix = torch.cat((before, audio, after), dim=0).unsqueeze(0)
    attention = torch.ones(1, prefix.shape[1], dtype=torch.long, device=device)
    probe.qwen.config.use_cache = True
    output = probe.qwen(
        inputs_embeds=prefix,
        attention_mask=attention,
        use_cache=True,
    )
    cache = output.past_key_values
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = []
    for _ in range(max_new_tokens):
        token = int(next_token.item())
        if token == int(tokenizer.eos_token_id):
            break
        generated.append(token)
        attention = torch.cat(
            (attention, torch.ones(1, 1, dtype=torch.long, device=device)), dim=1
        )
        output = probe.qwen(
            input_ids=next_token,
            attention_mask=attention,
            past_key_values=cache,
            use_cache=True,
        )
        cache = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    probe.qwen.config.use_cache = False
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-per-system", type=int)
    parser.add_argument("--system", action="append")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") not in SUPPORTED_CHECKPOINT_FORMATS:
        raise RuntimeError("invalid LM-ASR checkpoint")
    config = checkpoint["config"]
    systems = tuple(args.system or SYSTEMS)
    if len(systems) != len(set(systems)):
        raise RuntimeError(f"duplicate systems: {systems}")
    probe, tokenizer = build_probe(
        args.model,
        input_dim=int(config["input_dim"]),
        lora_rank=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        device=device,
    )
    load_clean_state_dict(probe, checkpoint["trainable_state"])
    probe.qwen.gradient_checkpointing_disable()
    probe.eval()

    dataset = ASRManifestDataset(args.manifest)
    selected = []
    counts = Counter()
    for index, record in enumerate(dataset.records):
        system = record["system"]
        if system not in systems:
            continue
        if args.max_per_system is not None and counts[system] >= args.max_per_system:
            continue
        selected.append(index)
        counts[system] += 1

    output_path = args.output_dir / "predictions.jsonl"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    totals = defaultdict(lambda: Counter(reference_words=0, substitutions=0, deletions=0, insertions=0))
    started = time.perf_counter()
    with temporary.open("w", encoding="utf-8") as handle:
        for completed, index in enumerate(selected, 1):
            item = dataset[index]
            hypothesis = greedy_decode(
                probe,
                tokenizer,
                item,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
            substitutions, deletions, insertions = edit_counts(
                item["transcript"], hypothesis
            )
            reference_words = len(normalize_transcript(item["transcript"]).split())
            system_totals = totals[item["system"]]
            system_totals["reference_words"] += reference_words
            system_totals["substitutions"] += substitutions
            system_totals["deletions"] += deletions
            system_totals["insertions"] += insertions
            row = {
                "utt_id": item["utt_id"],
                "system": item["system"],
                "reference": item["transcript"],
                "hypothesis": hypothesis,
                "realized_rate_hz": item["realized_rate_hz"],
                "reference_words": reference_words,
                "substitutions": substitutions,
                "deletions": deletions,
                "insertions": insertions,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if completed % 20 == 0 or completed == len(selected):
                print(f"decoded {completed}/{len(selected)}", flush=True)
    temporary.replace(output_path)

    summary = []
    for system in systems:
        values = totals[system]
        errors = values["substitutions"] + values["deletions"] + values["insertions"]
        words = values["reference_words"]
        summary.append(
            {
                "system": system,
                "utterances": counts[system],
                **dict(values),
                "errors": errors,
                "wer": errors / words if words else None,
            }
        )
    report = {
        "status": "passed",
        "checkpoint": str(args.checkpoint),
        "systems": list(systems),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    }
    (args.output_dir / "wer_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "eval.exit").write_text("0\n", encoding="ascii")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
