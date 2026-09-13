"""Small deterministic I/O helpers shared by boundary-evaluation stages."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def stable_name(utt_id: str, suffix: str = ".npz") -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", utt_id)
    digest = hashlib.sha1(utt_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe}_{digest}{suffix}"


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    samples, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if source_rate != sample_rate:
        mono = librosa.resample(mono, orig_sr=source_rate, target_sr=sample_rate)
    return np.asarray(mono, dtype=np.float32)


def transcript_from_record(record: dict) -> str:
    direct = record.get("transcript") or record.get("text_value")
    if direct:
        return str(direct).strip()
    nested = record.get("manifest_row") or {}
    direct = nested.get("transcript") or nested.get("text_value")
    if direct:
        return str(direct).strip()
    text_path = record.get("text") or nested.get("text")
    if not text_path:
        raise KeyError(f"record {record.get('utt_id')} does not contain a transcript")
    line = Path(text_path).read_text(encoding="utf-8", errors="ignore").strip()
    parts = line.split(maxsplit=2)
    if len(parts) == 3 and (parts[0].isdigit() or "_" in parts[0].lower()):
        return parts[2]
    return line


def normalize_manifest_record(record: dict) -> dict:
    nested = record.get("manifest_row") or {}
    utt_id = str(record.get("utt_id") or nested.get("utt_id"))
    audio = str(record.get("audio") or nested.get("audio"))
    if not utt_id or not audio:
        raise ValueError("manifest rows require utt_id and audio")
    duration = float(record.get("duration_sec") or nested.get("duration_sec") or 0.0)
    if duration <= 0:
        duration = len(load_audio(audio)) / 16000.0
    return {
        "utt_id": utt_id,
        "audio": audio,
        "duration_sec": duration,
        "split": str(record.get("split") or nested.get("split") or "test"),
        "speaker": str(record.get("speaker") or nested.get("speaker") or "unknown"),
        "transcript": transcript_from_record(record),
    }


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
