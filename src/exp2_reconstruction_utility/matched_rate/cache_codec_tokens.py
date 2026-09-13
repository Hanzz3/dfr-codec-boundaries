#!/usr/bin/env python3
"""Cache fixed FlexiCodec codes for one algorithm-specific decoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from flexicodec_adapter import PreparedAlignmentProvider, external_alignment, prepare_pinned_flexicodec
from io_utils import load_audio, read_jsonl, stable_name, write_jsonl


def encoded_token_count(encoded: dict, batch_index: int = 0) -> int:
    """Normalize FlexiCodec's current `total_frames` and older field names."""
    value = encoded.get("total_frames", encoded.get("speech_token_len"))
    if value is not None:
        if isinstance(value, torch.Tensor):
            value = value.reshape(-1)[batch_index].item()
        elif isinstance(value, (list, tuple)):
            value = value[batch_index]
        return int(value)
    lengths = encoded["token_lengths"][batch_index]
    return int(torch.count_nonzero(lengths).item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--flexicodec-repo", type=Path, default=Path("./repos/FlexiCodec"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-quantizers", type=int, default=8)
    parser.add_argument("--max-span", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--max-train-utterances", type=int)
    parser.add_argument("--max-val-utterances", type=int)
    parser.add_argument("--max-test-utterances", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(args.flexicodec_repo))
    from flexicodec.infer import encode_flexicodec

    predictions = [row for row in read_jsonl(args.predictions) if row["system"] == args.system]
    if args.max_utterances:
        keep_ids = []
        seen = set()
        for row in predictions:
            if row["utt_id"] not in seen and len(seen) >= args.max_utterances:
                continue
            seen.add(row["utt_id"])
            keep_ids.append(row)
        predictions = keep_ids
    split_limits = {
        "train": args.max_train_utterances,
        "val": args.max_val_utterances,
        "test": args.max_test_utterances,
    }
    if any(value for value in split_limits.values()):
        kept = []
        seen_by_split: dict[str, set[str]] = {split: set() for split in split_limits}
        for row in predictions:
            split = row.get("split", "test")
            limit = split_limits.get(split)
            seen = seen_by_split.setdefault(split, set())
            if row["utt_id"] not in seen and limit and len(seen) >= limit:
                continue
            seen.add(row["utt_id"])
            kept.append(row)
        predictions = kept
    selected = [row for index, row in enumerate(predictions) if index % args.num_shards == args.shard_index]
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_root / "records"
    cache_dir.mkdir(exist_ok=True)
    model_dict = prepare_pinned_flexicodec(args.flexicodec_repo, device)
    codec = model_dict["model"]
    codec.eval()
    provider = PreparedAlignmentProvider(args.max_span)
    rows = []
    for completed, row in enumerate(selected, 1):
        key = f"{row['utt_id']}__{row['condition']}"
        path = cache_dir / stable_name(key, suffix=".pt")
        if args.force or not path.exists():
            samples = load_audio(row["audio"])
            audio = torch.from_numpy(samples).unsqueeze(0).to(device)
            provider.set_batch(
                [row["boundary_times"]],
                [float(row["duration_sec"])],
                [args.system == "no_merge"],
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            encode_started = time.perf_counter()
            with torch.inference_mode(), external_alignment(codec, provider):
                encoded = encode_flexicodec(
                    audio,
                    model_dict,
                    16000,
                    num_quantizers=args.num_quantizers,
                    merging_threshold=0.91,
                )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            encode_seconds = time.perf_counter() - encode_started
            payload = {
                "semantic_codes": encoded["semantic_codes"][0].detach().cpu().long(),
                "acoustic_codes": (
                    None if encoded.get("acoustic_codes") is None else encoded["acoustic_codes"][0].detach().cpu().long()
                ),
                "token_lengths": encoded["token_lengths"][0].detach().cpu().long(),
                "speech_token_len": encoded_token_count(encoded, 0),
                "reference_samples": int(len(samples)),
                "encode_seconds": float(encode_seconds),
            }
            temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
            torch.save(payload, temporary)
            temporary.replace(path)
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if int(cached["token_lengths"].sum()) < 1 or int(cached["speech_token_len"]) < 1:
            raise RuntimeError(f"invalid cached codes: {path}")
        rows.append(
            {
                **row,
                "code_path": str(path),
                "semantic_shape": list(cached["semantic_codes"].shape),
                "acoustic_shape": None if cached["acoustic_codes"] is None else list(cached["acoustic_codes"].shape),
                "token_length_shape": list(cached["token_lengths"].shape),
                "reference_samples": int(cached["reference_samples"]),
                "encode_seconds": float(cached.get("encode_seconds", float("nan"))),
            }
        )
        if completed % 25 == 0 or completed == len(selected):
            print(f"{args.system} code shard {args.shard_index}: {completed}/{len(selected)}", flush=True)
    index = args.output_root / f"index.part-{args.shard_index:05d}-of-{args.num_shards:05d}.jsonl"
    write_jsonl(index, rows)
    (args.output_root / f"codes.part-{args.shard_index:05d}-of-{args.num_shards:05d}.exit").write_text("0\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
