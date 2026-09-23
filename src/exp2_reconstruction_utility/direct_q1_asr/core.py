"""Shared data, model, checkpoint, and WER utilities for the LM-ASR probe."""

from __future__ import annotations

import json
import math
import random
import re
import sys
from array import array
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler

from paper_protocol import PAPER_SYSTEMS

SYSTEMS = PAPER_SYSTEMS
CHECKPOINT_FORMAT = "direct_q1_asr_probe_v1"
# Backward-compatible names used by earlier checkpoints.
SUPPORTED_CHECKPOINT_FORMATS = {CHECKPOINT_FORMAT, "exp3_lm_asr_probe_v1"}
PROMPT_BEFORE = "Transcribe the following speech into English.\nAudio:"
PROMPT_AFTER = "\nTranscript:"


def patch_transformers_torchvision() -> None:
    """Avoid an unrelated broken torchvision build in the AIStation image."""

    import transformers
    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as import_utils

    import_utils._torchvision_available = False
    import_utils.is_torchvision_available = lambda: False
    transformers_utils.is_torchvision_available = lambda: False


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_feature(path: str | Path, key: str = "features") -> np.ndarray:
    with np.load(path) as payload:
        values = payload[key].astype(np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise RuntimeError(f"invalid feature matrix: {path}")
    return values


def pool_segments(features: np.ndarray, starts: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    frames = int(features.shape[0])
    points = [int(value) for value in starts]
    if not points or points[0] != 0 or points != sorted(set(points)):
        raise ValueError("starts must be strictly increasing and begin at zero")
    if points[-1] >= frames:
        raise ValueError("segment start exceeds feature length")
    ends = [*points[1:], frames]
    pooled = np.stack(
        [features[start:end].mean(axis=0) for start, end in zip(points, ends)]
    ).astype(np.float32)
    lengths = np.asarray([end - start for start, end in zip(points, ends)], dtype=np.float32)
    if int(lengths.sum()) != frames:
        raise RuntimeError("segment lengths do not cover the input")
    return pooled, lengths


class ASRManifestDataset(Dataset):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.records = []
        self.offsets = array("Q")
        self._handle: BinaryIO | None = None
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                metadata = {
                    "system": sys.intern(str(record["system"])),
                    "utt_id": str(record["utt_id"]),
                }
                if "feature_dim" in record:
                    metadata["feature_dim"] = int(record["feature_dim"])
                self.offsets.append(offset)
                self.records.append(metadata)
        if not self.records:
            raise RuntimeError(f"empty manifest: {path}")

    def _record(self, index: int) -> dict:
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("rb")
        self._handle.seek(self.offsets[index])
        line = self._handle.readline()
        if not line:
            raise RuntimeError(f"manifest offset exceeds file: {self.path}:{index}")
        return json.loads(line)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self._record(index)
        pooled, lengths = pool_segments(
            load_feature(
                record["feature_path"],
                str(record.get("feature_key", "features")),
            ),
            record["starts"],
        )
        duration_values = record.get("segment_durations_sec")
        if duration_values is None:
            duration_values = lengths
            duration_unit = "frames"
        else:
            duration_values = np.asarray(duration_values, dtype=np.float32)
            if (
                duration_values.shape != lengths.shape
                or not np.isfinite(duration_values).all()
                or np.any(duration_values <= 0)
            ):
                raise RuntimeError(
                    f"invalid segment durations for {record.get('utt_id', index)}"
                )
            duration_unit = "seconds"
        return {
            "utt_id": record["utt_id"],
            "system": record["system"],
            "transcript": record["transcript"],
            "features": torch.from_numpy(pooled),
            "segment_lengths": torch.from_numpy(duration_values),
            "duration_unit": duration_unit,
            "realized_rate_hz": float(record["realized_rate_hz"]),
        }


class BalancedSystemBatchSampler(Sampler[list[int]]):
    """Yield one-system batches while cycling evenly through all systems."""

    def __init__(
        self,
        records: Sequence[dict],
        *,
        systems: Sequence[str],
        batch_size: int,
        steps: int,
        seed: int,
        start_step: int = 0,
    ):
        self.systems = tuple(systems)
        self.batch_size = int(batch_size)
        self.steps = int(steps)
        self.seed = int(seed)
        self.start_step = int(start_step)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            groups[record["system"]].append(index)
        missing = [system for system in self.systems if not groups[system]]
        if missing:
            raise RuntimeError(f"missing systems in training manifest: {missing}")
        self.groups = dict(groups)

    def __len__(self) -> int:
        return max(0, self.steps - self.start_step)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        pools = {}
        positions = {}
        for system in self.systems:
            values = list(self.groups[system])
            rng.shuffle(values)
            pools[system] = values
            positions[system] = 0

        for step in range(self.steps):
            system = self.systems[step % len(self.systems)]
            batch = []
            for _ in range(self.batch_size):
                if positions[system] >= len(pools[system]):
                    rng.shuffle(pools[system])
                    positions[system] = 0
                batch.append(pools[system][positions[system]])
                positions[system] += 1
            if step >= self.start_step:
                yield batch


def identity_collate(items: list[dict]) -> list[dict]:
    return items


class AcousticProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.feature_norm = nn.LayerNorm(input_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.duration_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        feature_values = self.feature_mlp(self.feature_norm(features))
        duration_values = self.duration_mlp(torch.log1p(lengths).unsqueeze(-1))
        return feature_values + duration_values


class QwenASRProbe(nn.Module):
    def __init__(self, qwen: nn.Module, projector: AcousticProjector):
        super().__init__()
        self.qwen = qwen
        self.projector = projector


def build_probe(
    model_path: Path,
    *,
    input_dim: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    device: torch.device,
) -> tuple[QwenASRProbe, object]:
    patch_transformers_torchvision()
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    qwen.config.use_cache = False
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        bias="none",
    )
    qwen = get_peft_model(qwen, config)
    qwen.gradient_checkpointing_enable()
    hidden_size = int(qwen.config.hidden_size)
    probe = QwenASRProbe(qwen, AcousticProjector(input_dim, hidden_size)).to(device)
    return probe, tokenizer


def tokenize_text(tokenizer, text: str, max_tokens: int) -> list[int]:  # noqa: ANN001
    values = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )["input_ids"]
    return [int(value) for value in values]


def build_training_tensors(
    probe: QwenASRProbe,
    tokenizer,
    items: list[dict],
    *,
    device: torch.device,
    max_text_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embedding = probe.qwen.get_input_embeddings()
    before_ids = tokenize_text(tokenizer, PROMPT_BEFORE, 64)
    after_ids = tokenize_text(tokenizer, PROMPT_AFTER, 32)
    sequences = []
    labels = []
    for item in items:
        features = item["features"].to(device)
        lengths = item["segment_lengths"].to(device)
        audio = probe.projector(features, lengths).to(dtype=embedding.weight.dtype)
        target_ids = tokenize_text(tokenizer, item["transcript"], max_text_tokens - 1)
        target_ids.append(int(tokenizer.eos_token_id))
        before = embedding(torch.tensor(before_ids, device=device))
        after = embedding(torch.tensor(after_ids, device=device))
        target_tensor = torch.tensor(target_ids, device=device)
        target = embedding(target_tensor)
        sequence = torch.cat((before, audio, after, target), dim=0)
        ignored = len(before_ids) + len(audio) + len(after_ids)
        label = torch.tensor(
            [-100] * ignored + target_ids, dtype=torch.long, device=device
        )
        sequences.append(sequence)
        labels.append(label)

    width = max(sequence.shape[0] for sequence in sequences)
    hidden = sequences[0].shape[1]
    batch = torch.zeros(
        len(sequences), width, hidden, dtype=sequences[0].dtype, device=device
    )
    attention = torch.zeros(len(sequences), width, dtype=torch.long, device=device)
    padded_labels = torch.full(
        (len(sequences), width), -100, dtype=torch.long, device=device
    )
    for index, (sequence, label) in enumerate(zip(sequences, labels)):
        length = sequence.shape[0]
        batch[index, :length] = sequence
        attention[index, :length] = 1
        padded_labels[index, :length] = label
    return batch, attention, padded_labels


def trainable_parameter_audit(probe: QwenASRProbe) -> dict:
    names = [name for name, parameter in probe.named_parameters() if parameter.requires_grad]
    forbidden = [
        name
        for name in names
        if not name.startswith("projector.") and "lora_" not in name
    ]
    if forbidden:
        raise RuntimeError(f"unexpected trainable Qwen parameters: {forbidden[:20]}")
    return {
        "parameter_tensors": len(names),
        "parameter_count": sum(
            parameter.numel()
            for parameter in probe.parameters()
            if parameter.requires_grad
        ),
        "projector_parameter_count": sum(
            parameter.numel() for parameter in probe.projector.parameters()
        ),
        "names": names,
        "frozen_qwen_base": True,
    }


def clean_state_dict(probe: QwenASRProbe) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in probe.named_parameters()
        if parameter.requires_grad
    }


def load_clean_state_dict(probe: QwenASRProbe, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(probe.named_parameters())
    missing = [name for name in state if name not in parameters]
    if missing:
        raise RuntimeError(f"checkpoint parameters are missing from model: {missing[:10]}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))


def save_checkpoint(
    path: Path,
    probe: QwenASRProbe,
    *,
    step: int,
    best_validation_loss: float,
    validation_loss: float,
    config: dict,
    optimizer=None,  # noqa: ANN001
    scheduler=None,  # noqa: ANN001
    scaler=None,  # noqa: ANN001
) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "best_validation_loss": float(best_validation_loss),
        "validation_loss": float(validation_loss),
        "config": config,
        "trainable_state": clean_state_dict(probe),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
        payload["scaler"] = scaler.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def normalize_transcript(text: str) -> str:
    value = text.lower().replace("’", "'")
    value = re.sub(r"[^a-z0-9' ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def edit_counts(reference: str, hypothesis: str) -> tuple[int, int, int]:
    ref = normalize_transcript(reference).split()
    hyp = normalize_transcript(hypothesis).split()
    rows = len(ref) + 1
    cols = len(hyp) + 1
    cost = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0] = (i, 0, i, 0)
    for j in range(1, cols):
        cost[0][j] = (j, 0, 0, j)
    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
                continue
            substitution = cost[i - 1][j - 1]
            deletion = cost[i - 1][j]
            insertion = cost[i][j - 1]
            candidates = (
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            )
            cost[i][j] = min(candidates)
    _, substitutions, deletions, insertions = cost[-1][-1]
    return substitutions, deletions, insertions


def cosine_schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
