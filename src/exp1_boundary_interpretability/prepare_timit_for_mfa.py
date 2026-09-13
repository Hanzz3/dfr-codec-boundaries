#!/usr/bin/env python3
"""Prepare TIMIT into MFA-ready corpus and export TIMIT reference boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import soundfile as sf

TIMIT_SAMPLE_RATE = 16000.0


@dataclass
class UtterancePaths:
    split: str
    wav: Path
    phn: Path
    wrd: Path
    txt: Path
    split_root: Path

def normalize_split(split: str) -> str:
    split = split.strip().lower()
    if split not in {"train", "test", "all"}:
        raise ValueError(f"Unsupported split: {split}")
    return split


def resolve_split_root(timit_root: Path, split: str) -> Path:
    candidates = [timit_root / split.upper(), timit_root / split.lower(), timit_root / split]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand
    raise FileNotFoundError(f"Could not find split directory for '{split}' under {timit_root}")


def find_companion(base: Path, suffix: str) -> Optional[Path]:
    candidates = [
        base.with_suffix(suffix.upper()),
        base.with_suffix(suffix.lower()),
        Path(str(base) + suffix.upper()),
        Path(str(base) + suffix.lower()),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def collect_utterances(timit_root: Path, split: str) -> Iterable[UtterancePaths]:
    splits = ["train", "test"] if split == "all" else [split]
    for sp in splits:
        split_root = resolve_split_root(timit_root, sp)
        wav_files = list(split_root.rglob("*.WAV")) + list(split_root.rglob("*.wav"))
        wav_files = sorted(set(wav_files))
        for wav in wav_files:
            base = wav.with_suffix("")
            phn = find_companion(base, ".PHN")
            wrd = find_companion(base, ".WRD")
            txt = find_companion(base, ".TXT")
            if not phn or not wrd or not txt:
                continue
            yield UtterancePaths(split=sp, wav=wav, phn=phn, wrd=wrd, txt=txt, split_root=split_root)


def parse_timit_txt(txt_path: Path) -> str:
    line = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not line:
        return ""
    parts = line.split()
    if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
        return " ".join(parts[2:]).strip().lower()
    return line.lower()


def parse_boundary_file(path: Path, sample_rate: float = TIMIT_SAMPLE_RATE) -> List[Dict[str, object]]:
    intervals: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 3:
            continue
        if not (parts[0].isdigit() and parts[1].isdigit()):
            continue
        start = int(parts[0]) / sample_rate
        end = int(parts[1]) / sample_rate
        label = " ".join(parts[2:])
        intervals.append({"start": start, "end": end, "label": label})
    return intervals

def ensure_pcm16_wav(src_path: Path, dst_path: Path) -> None:
    audio, sr = sf.read(str(src_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    sf.write(str(dst_path), audio, sr, subtype="PCM_16")


def safe_symlink_or_copy(src: Path, dst: Path, copy_audio: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_audio:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def compute_utt_id(split: str, rel_noext: Path) -> str:
    rel = rel_noext.as_posix().replace("/", "_").replace("\\", "_").lower()
    return f"{split}_{rel}"


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--max-utts", type=int, default=0, help="0 means no limit")
    parser.add_argument("--copy-audio", action="store_true", help="Copy instead of symlink when not converting")
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip conversion and directly symlink/copy original wav files",
    )
    return parser

def main() -> None:
    args = build_parser().parse_args()
    split = normalize_split(args.split)

    output_root = args.output_root
    corpus_dir = output_root / "mfa_corpus"
    manifest_path = output_root / "prep_manifest.csv"
    reference_path = output_root / "timit_reference.jsonl"

    corpus_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    reference_rows: List[Dict[str, object]] = []

    count = 0
    for utt in collect_utterances(args.timit_root, split):
        rel_noext = utt.wav.relative_to(utt.split_root).with_suffix("")
        out_base = corpus_dir / utt.split / rel_noext
        out_base.parent.mkdir(parents=True, exist_ok=True)

        out_wav = out_base.with_suffix(".wav")
        out_lab = out_base.with_suffix(".lab")

        if args.skip_convert:
            safe_symlink_or_copy(utt.wav, out_wav, args.copy_audio)
        else:
            ensure_pcm16_wav(utt.wav, out_wav)

        transcript = parse_timit_txt(utt.txt)
        out_lab.write_text(transcript + "\n", encoding="utf-8")

        phn_intervals = parse_boundary_file(utt.phn)
        wrd_intervals = parse_boundary_file(utt.wrd)

        utt_id = compute_utt_id(utt.split, rel_noext)
        rel_key = (Path(utt.split) / rel_noext).as_posix()

        manifest_rows.append(
            {
                "utt_id": utt_id,
                "split": utt.split,
                "rel_key": rel_key,
                "wav_path": str(out_wav.resolve()),
                "lab_path": str(out_lab.resolve()),
                "transcript": transcript,
            }
        )
        reference_rows.append(
            {
                "utt_id": utt_id,
                "split": utt.split,
                "rel_key": rel_key,
                "wav_path": str(out_wav.resolve()),
                "transcript": transcript,
                "timit_phone_intervals": phn_intervals,
                "timit_word_intervals": wrd_intervals,
            }
        )

        count += 1
        if args.max_utts > 0 and count >= args.max_utts:
            break

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utt_id", "split", "rel_key", "wav_path", "lab_path", "transcript"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    write_jsonl(reference_path, reference_rows)

    print(f"[prepare_timit_for_mfa] prepared utterances: {len(manifest_rows)}")
    print(f"[prepare_timit_for_mfa] corpus_dir: {corpus_dir}")
    print(f"[prepare_timit_for_mfa] manifest: {manifest_path}")
    print(f"[prepare_timit_for_mfa] reference: {reference_path}")


if __name__ == "__main__":
    main()
