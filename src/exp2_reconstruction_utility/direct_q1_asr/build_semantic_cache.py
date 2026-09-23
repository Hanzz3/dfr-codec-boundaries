#!/usr/bin/env python3
"""Build TIMIT q1-sem latents from a shared q8 FlexiCodec encode."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import librosa
import numpy as np
import soundfile as sf
import torch

from paper_protocol import PAPER_SYSTEMS

SYSTEMS = PAPER_SYSTEMS


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def transcript(item: dict) -> str:
    path = (item.get("manifest_row") or {}).get("text")
    if not path:
        return ""
    line = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    fields = line.split(maxsplit=2)
    return fields[2] if len(fields) == 3 and fields[0].isdigit() else line


def load_audio(path: str | Path) -> torch.Tensor:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if sample_rate != 16000:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=16000)
    return torch.from_numpy(np.asarray(mono, dtype=np.float32))


class FrozenAlignment:
    def __init__(self, starts: list[int]) -> None:
        self.starts = [int(value) for value in starts]

    def __call__(self, frames: torch.Tensor, x_lens: torch.Tensor | None = None):
        if frames.shape[0] != 1:
            raise RuntimeError("frozen TIMIT alignment requires batch size one")
        valid = frames.shape[1] if x_lens is None else int(x_lens[0].item())
        if not self.starts or self.starts[0] != 0 or any(
            right <= left for left, right in zip(self.starts, self.starts[1:])
        ):
            raise RuntimeError(f"invalid starts: {self.starts}")
        if self.starts[-1] >= valid:
            raise RuntimeError(f"frozen starts exceed codec grid: {self.starts[-1]} >= {valid}")
        keep = np.zeros(valid, dtype=np.int64)
        keep[self.starts] = 1
        segment_ids = np.cumsum(keep) - 1
        alignment = np.zeros((len(self.starts), frames.shape[1]), dtype=np.float32)
        alignment[segment_ids, np.arange(valid)] = 1.0
        similarity = torch.nn.functional.cosine_similarity(frames[:, :-1], frames[:, 1:], dim=-1)
        return (
            torch.from_numpy(alignment).unsqueeze(0).to(frames.device),
            similarity,
            torch.tensor([len(self.starts)], dtype=torch.long, device=frames.device),
        )


@contextmanager
def external_alignment(model, provider: FrozenAlignment):
    name = "_perform_similarity_alignment_vectorized"
    original = getattr(model, name)

    def patched(this, frames, x_lens=None):
        del this
        return provider(frames, x_lens)

    setattr(model, name, MethodType(patched, model))
    try:
        yield
    finally:
        setattr(model, name, original)


def atomic_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--reference-tracks", "--ground-truth", dest="reference_tracks", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flexicodec-repo", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--expected-utterances", type=int, default=1680)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")

    records = [row for row in read_jsonl(args.feature_index) if row.get("split") == "test"]
    if len(records) != args.expected_utterances:
        raise RuntimeError(f"unexpected TEST coverage: {len(records)}")
    selected = [row for index, row in enumerate(records) if index % args.num_shards == args.shard_index]
    if args.max_utterances is not None:
        selected = selected[: args.max_utterances]
    predictions = {(row["utt_id"], row["system"]): row for row in read_jsonl(args.predictions)}
    expected_keys = {(row["utt_id"], system) for row in records for system in SYSTEMS}
    if set(predictions) != expected_keys:
        raise RuntimeError("final paper prediction key coverage mismatch")
    ground_truth = {row["utt_id"]: row for row in read_jsonl(args.reference_tracks)}

    device = torch.device(args.device)
    torch.cuda.set_device(device)
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
    codec = model_dict["model"].eval()
    for parameter in codec.parameters():
        parameter.requires_grad = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = args.output_dir / "items"
    items.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    started = time.perf_counter()
    for completed, record in enumerate(selected, 1):
        utt_id = str(record["utt_id"])
        destination = items / f"{utt_id}.npz"
        arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, dict] = {}
        if destination.is_file():
            with np.load(destination) as payload:
                arrays = {key: payload[key] for key in payload.files if key.startswith("q1__")}
                metadata = json.loads(str(payload["metadata"].item()))
        else:
            audio = load_audio(record["audio"]).to(device)
            for system in SYSTEMS:
                starts = predictions[(utt_id, system)]["starts"]
                provider = FrozenAlignment(starts)
                with torch.inference_mode(), external_alignment(codec, provider):
                    encoded = encode_flexicodec(
                        audio.unsqueeze(0),
                        model_dict,
                        16000,
                        num_quantizers=8,
                        merging_threshold=0.91,
                        audio_lens=[len(audio)],
                    )
                    semantic_latent = codec.semantic_vq.from_codes(encoded["semantic_codes"])[0]
                    q1 = codec.convnext_decoder(semantic_latent)
                matrix = q1[0].transpose(0, 1).detach().float().cpu().numpy().astype(np.float16)
                if matrix.ndim != 2 or matrix.shape[1] != 512 or not np.isfinite(matrix).all():
                    raise RuntimeError(f"invalid q1-sem latent: {utt_id} / {system}")
                token_lengths = encoded["token_lengths"][0].detach().float().cpu().numpy()
                duration = float(record["duration_sec"])
                durations = token_lengths * (duration / max(float(token_lengths.sum()), 1e-12))
                arrays[f"q1__{system}"] = matrix
                metadata[system] = {
                    "segments": len(matrix),
                    "durations": durations.tolist(),
                    "realized_rate_hz": len(matrix) / duration,
                    "codec_grid_starts": [int(value) for value in starts],
                }
            atomic_npz(destination, {**arrays, "metadata": np.array(json.dumps(metadata))})

        reference = transcript(ground_truth[utt_id])
        for system in SYSTEMS:
            matrix = arrays[f"q1__{system}"]
            meta = metadata[system]
            manifest_rows.append(
                {
                    "utt_id": utt_id,
                    "system": system,
                    "transcript": reference,
                    "feature_path": str(destination.resolve()),
                    "feature_key": f"q1__{system}",
                    "feature_dim": int(matrix.shape[1]),
                    "starts": list(range(int(matrix.shape[0]))),
                    "segment_durations_sec": meta["durations"],
                    "realized_rate_hz": meta["realized_rate_hz"],
                    "num_quantizers": 1,
                    "quantized_latent": True,
                    "latent_definition": "semantic_component_from_shared_q8_encode",
                    "codec_grid_starts": meta["codec_grid_starts"],
                }
            )
        if completed % 10 == 0 or completed == len(selected):
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "completed": completed,
                        "total": len(selected),
                        "rows": len(manifest_rows),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    manifest = args.output_dir / f"q1_sem.part-{args.shard_index:05d}-of-{args.num_shards:05d}.jsonl"
    atomic_text(manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows))
    acceptance = {
        "status": "passed",
        "protocol": "timit_paper_seven_q1_semantic_component_from_shared_q8_encode_v1",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "utterances": len(selected),
        "records": len(manifest_rows),
        "systems": list(SYSTEMS),
        "feature_dim": 512,
        "finite": True,
    }
    atomic_text(
        args.output_dir / f"acceptance.shard_{args.shard_index:05d}.json",
        json.dumps(acceptance, indent=2) + "\n",
    )
    atomic_text(args.output_dir / f"cache.shard_{args.shard_index:05d}.exit", "0\n")
    print(json.dumps(acceptance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
