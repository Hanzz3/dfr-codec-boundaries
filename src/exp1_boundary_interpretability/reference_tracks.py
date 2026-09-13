"""Construct the six reference tracks used by EXP1 and Table 1."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd

SILENCE = {"", "<eps>", "h#", "pau", "epi", "sil", "sp", "spn"}
TIMIT_VOWELS = {
    "aa", "ae", "ah", "ao", "aw", "ax", "ax-h", "axr", "ay", "eh", "el",
    "er", "ey", "ih", "ix", "iy", "ow", "oy", "uh", "uw", "ux",
}
IPA_VOWEL_CHARS = set("aeiouɑɒɔəɚɝæɛɪʊʌøœɜɐɞɶyɨʉɯɤ")


@dataclass(frozen=True)
class Interval:
    begin: float
    end: float
    label: str
    source: str


@dataclass(frozen=True)
class BoundarySet:
    phoneme: list[float]
    syllable: list[float]
    bpe_subword: list[float]
    word: list[float]
    acoustic_event: list[float]
    vuv: list[float]
    silence_phone: list[float]


@dataclass(frozen=True)
class GroundTruth:
    boundaries: BoundarySet
    records: list[dict]
    stats: dict


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, index: bool = False) -> str:
    return frame.to_markdown(index=index)


def load_audio(path: str) -> tuple[np.ndarray, int]:
    waveform, sample_rate = librosa.load(path, sr=16000, mono=True)
    return waveform.astype(np.float32), sample_rate


def normalize_label(label: str) -> str:
    return label.lower().strip().replace("ː", "")


def is_silence(label: str) -> bool:
    return normalize_label(label) in SILENCE


def is_vowel(label: str) -> bool:
    normalized = normalize_label(label)
    return normalized in TIMIT_VOWELS or any(char in IPA_VOWEL_CHARS for char in normalized)


def read_timit_intervals(path: str | None, source: str) -> list[Interval]:
    if not path or not Path(path).is_file():
        return []
    intervals = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split(maxsplit=2)
            if len(fields) == 3:
                intervals.append(Interval(int(fields[0]) / 16000, int(fields[1]) / 16000, fields[2], source))
    return intervals


def find_mfa_csv(mfa_dir: Path | None, utterance_id: str) -> Path | None:
    if mfa_dir is None:
        return None
    direct = mfa_dir / f"{utterance_id}.csv"
    if direct.is_file():
        return direct
    matches = list(mfa_dir.rglob(f"{utterance_id}.csv"))
    return matches[0] if len(matches) == 1 else None


def read_mfa_csv(path: Path | None) -> tuple[list[Interval], list[Interval]]:
    if path is None or not path.is_file():
        return [], []
    rows = pd.read_csv(path)
    names = {name.lower(): name for name in rows.columns}
    phones, words = [], []
    for _, row in rows.iterrows():
        interval = Interval(float(row[names["begin"]]), float(row[names["end"]]), str(row[names["label"]]), "mfa")
        kind = str(row[names["type"]]).lower()
        if "phone" in kind:
            phones.append(interval)
        elif "word" in kind:
            words.append(interval)
    return phones, words


def dedupe_sorted(values: Iterable[float], resolution: float = 0.002) -> list[float]:
    output = []
    for value in sorted(float(item) for item in values if np.isfinite(float(item))):
        if not output or abs(value - output[-1]) > resolution:
            output.append(value)
    return output


def interval_boundaries(intervals: list[Interval], include_silence: bool = True) -> tuple[list[float], list[float]]:
    boundaries, silence = [], []
    ordered = sorted(intervals, key=lambda item: (item.begin, item.end))
    for left, right in zip(ordered, ordered[1:]):
        if right.begin <= 0:
            continue
        time = float(right.begin)
        if is_silence(left.label) or is_silence(right.label):
            silence.append(time)
        if include_silence or (not is_silence(left.label) and not is_silence(right.label)):
            boundaries.append(time)
    return dedupe_sorted(boundaries), dedupe_sorted(silence)


def phones_for_word(phones: list[Interval], word: Interval) -> list[Interval]:
    selected = []
    for phone in phones:
        center = (phone.begin + phone.end) / 2
        overlap = max(0.0, min(phone.end, word.end) - max(phone.begin, word.begin))
        if word.begin - 1e-3 <= center <= word.end + 1e-3 or overlap > 0.01:
            selected.append(phone)
    return selected


def syllable_spans_for_word(word_phones: list[Interval], word: Interval) -> list[Interval]:
    phones = [phone for phone in word_phones if not is_silence(phone.label)]
    if not phones:
        return [Interval(word.begin, word.end, word.label, "syllable_heuristic")]
    vowel_indices = [index for index, phone in enumerate(phones) if is_vowel(phone.label)]
    if len(vowel_indices) <= 1:
        return [Interval(max(word.begin, phones[0].begin), min(word.end, phones[-1].end), word.label, "syllable_heuristic")]
    starts = [0]
    for previous, current in zip(vowel_indices, vowel_indices[1:]):
        consonants = [index for index in range(previous + 1, current) if not is_vowel(phones[index].label)]
        boundary = consonants[-1] if consonants else current
        if boundary > starts[-1]:
            starts.append(boundary)
    output = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(phones)
        chunk = phones[start:end]
        if chunk:
            output.append(Interval(max(word.begin, chunk[0].begin), min(word.end, chunk[-1].end), f"{word.label}:{position + 1}", "syllable_heuristic"))
    return output


def build_syllable_boundaries(phones: list[Interval], words: list[Interval]) -> tuple[list[float], list[dict]]:
    boundaries, records = [], []
    for word in words:
        for index, span in enumerate(syllable_spans_for_word(phones_for_word(phones, word), word)):
            if span.begin > 0:
                boundaries.append(span.begin)
            records.append({"begin": span.begin, "end": span.end, "label": span.label, "source": span.source, "index_in_word": index})
    return dedupe_sorted(boundaries), records


def load_bpe_tokenizer(name: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    return tokenizer, f"transformers:{name}"


def word_for_bpe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9']+", "", label).lower()


def bpe_offsets(tokenizer, word: str) -> list[tuple[int, int, str]]:
    clean = word_for_bpe(word)
    encoded = tokenizer(clean, add_special_tokens=False, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    return [(int(start), int(end), str(token)) for (start, end), token in zip(encoded["offset_mapping"], tokens) if end > start]


def build_bpe_boundaries(words: list[Interval], tokenizer, tokenizer_source: str) -> tuple[list[float], list[dict]]:
    boundaries, records = [], []
    for word in words:
        clean = word_for_bpe(word.label)
        pieces = bpe_offsets(tokenizer, word.label)
        if not clean or not pieces:
            continue
        starts = [word.begin]
        for start, _, _ in pieces[1:]:
            starts.append(float(np.clip(word.begin + start / len(clean) * (word.end - word.begin), starts[-1] + 1e-6, word.end - 1e-6)))
        for index, ((_, _, token), begin) in enumerate(zip(pieces, starts)):
            end = starts[index + 1] if index + 1 < len(starts) else word.end
            if begin > 0:
                boundaries.append(begin)
            records.append({"begin": begin, "end": end, "label": token, "source": f"bpe_from_word_text_offset:{tokenizer_source}", "word": word.label, "index_in_word": index})
    return dedupe_sorted(boundaries), records


def smooth_binary(values: np.ndarray, min_run: int = 4) -> np.ndarray:
    output = values.copy()
    start = 0
    while start < len(output):
        end = start + 1
        while end < len(output) and output[end] == output[start]:
            end += 1
        if end - start < min_run:
            left = output[start - 1] if start else None
            right = output[end] if end < len(output) else None
            if left is not None and right is not None and left == right:
                output[start:end] = left
        start = end
    return output


def detect_vuv_boundaries(waveform: np.ndarray, sample_rate: int, hop: int = 160) -> list[float]:
    rms = librosa.feature.rms(y=waveform, frame_length=400, hop_length=hop, center=True)[0]
    zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=400, hop_length=hop, center=True)[0]
    if not len(rms):
        return []
    db = librosa.amplitude_to_db(rms + 1e-8, ref=np.max)
    voiced = smooth_binary((db > max(-45, float(np.percentile(db, 30)))) & (zcr < min(.22, float(np.percentile(zcr, 75)))))
    changes = np.flatnonzero(voiced[1:] != voiced[:-1]) + 1
    duration = len(waveform) / sample_rate
    return [float(time) for time in librosa.frames_to_time(changes, sr=sample_rate, hop_length=hop) if .05 <= time <= duration - .05]


def _energy_spectral_features(waveform: np.ndarray, sample_rate: int, hop: int = 160) -> tuple[np.ndarray, np.ndarray]:
    rms = librosa.feature.rms(y=waveform, frame_length=400, hop_length=hop, center=True)[0]
    zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=400, hop_length=hop, center=True)[0]
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate, n_fft=1024, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=waveform, n_fft=1024, hop_length=hop)[0]
    values = np.stack([np.log(rms + 1e-8), zcr, centroid / sample_rate, flatness], axis=1)
    times = librosa.frames_to_time(np.arange(len(values)), sr=sample_rate, hop_length=hop)
    return values.astype(np.float32), times.astype(np.float32)


def _local_peak_indices(times: np.ndarray, scores: np.ndarray, min_gap: float) -> list[int]:
    if not len(scores):
        return []
    if len(scores) < 3:
        candidates = list(range(len(scores)))
    else:
        candidates = [
            index
            for index in range(1, len(scores) - 1)
            if scores[index] >= scores[index - 1] and scores[index] >= scores[index + 1]
        ]
        if not candidates:
            candidates = list(range(len(scores)))
    selected = []
    for index in sorted(candidates, key=lambda item: float(scores[item]), reverse=True):
        if all(abs(float(times[index]) - float(times[other])) >= min_gap for other in selected):
            selected.append(index)
    return selected


def _select_rate_peaks(times: np.ndarray, scores: np.ndarray, count: int, min_gap: float) -> np.ndarray:
    if count <= 0 or not len(scores):
        return np.asarray([], dtype=np.float32)
    selected = _local_peak_indices(times, scores, min_gap)[:count]
    return np.asarray(sorted(float(times[index]) for index in selected), dtype=np.float32)


def detect_acoustic_event_boundaries(waveform: np.ndarray, sample_rate: int, target_rate_hz: float = 5) -> list[float]:
    features, times = _energy_spectral_features(waveform, sample_rate)
    standardized = (features - features.mean(axis=0, keepdims=True)) / np.where(
        features.std(axis=0, keepdims=True) < 1e-5,
        1.0,
        features.std(axis=0, keepdims=True),
    )
    norms = np.linalg.norm(standardized, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    normalized = standardized / norms
    scores = np.clip(1 - np.sum(normalized[:-1] * normalized[1:], axis=1), 0.0, 2.0)
    if scores.size and float(scores.max()) > float(scores.min()):
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    score_times = (times[:-1] + times[1:]) / 2
    duration = len(waveform) / sample_rate
    peaks = _select_rate_peaks(score_times, scores, max(1, round(duration * target_rate_hz)), .05)
    events = [*peaks, *detect_vuv_boundaries(waveform, sample_rate)]
    return [time for time in dedupe_sorted(events, .02) if .04 <= time <= duration - .04]


def build_ground_truth(row: dict, waveform: np.ndarray, sample_rate: int, mfa_dir: Path | None, tokenizer, tokenizer_source: str) -> GroundTruth:
    phones, words = read_mfa_csv(find_mfa_csv(mfa_dir, row["utt_id"]))
    if not phones or not words:
        raise RuntimeError(f"MFA phone/word intervals missing for {row['utt_id']}")
    phoneme, silence = interval_boundaries(phones, True)
    word, _ = interval_boundaries(words, False)
    syllable, syllable_records = build_syllable_boundaries(phones, words)
    bpe, bpe_records = build_bpe_boundaries(words, tokenizer, tokenizer_source)
    vuv = detect_vuv_boundaries(waveform, sample_rate)
    acoustic = detect_acoustic_event_boundaries(waveform, sample_rate)
    boundary_set = BoundarySet(phoneme, syllable, bpe, word, acoustic, vuv, silence)
    sources = {"phoneme": "mfa", "syllable": "mfa_phones_heuristic", "bpe_subword": f"mfa_words_text_offsets_{tokenizer_source}", "word": "mfa", "acoustic_event": "energy_spectrum_prosody", "vuv": "energy_zcr_vuv_detector"}
    records = [{"utt_id": row["utt_id"], "boundary_type": level, "time_sec": float(time), "source": sources[level]} for level, values in (("phoneme", phoneme), ("syllable", syllable), ("bpe_subword", bpe), ("word", word), ("acoustic_event", acoustic), ("vuv", vuv)) for time in values]
    duration = len(waveform) / sample_rate
    stats = {"utt_id": row["utt_id"], "duration_sec": duration, "phone_source": "mfa", "word_source": "mfa", "num_phone_intervals": len(phones), "num_word_intervals": len(words), "syllable_segments": len(syllable_records), "bpe_segments": len(bpe_records)}
    for level, values in (("phoneme", phoneme), ("syllable", syllable), ("bpe_subword", bpe), ("word", word), ("acoustic_event", acoustic), ("vuv", vuv)):
        stats[f"num_{level}_boundaries"] = len(values)
    return GroundTruth(boundary_set, records, stats)


def summarize_ground_truth(stats: list[dict], manifest_rows_total: int, tokenizer_source: str) -> dict:
    duration = sum(row["duration_sec"] for row in stats)
    summary = {"manifest_rows_total": manifest_rows_total, "analysis_rows": len(stats), "total_duration_sec": duration, "bpe_source": tokenizer_source, "phone_sources": dict(Counter(row["phone_source"] for row in stats)), "word_sources": dict(Counter(row["word_source"] for row in stats))}
    for level in ("phoneme", "syllable", "bpe_subword", "word", "acoustic_event", "vuv"):
        count = sum(row[f"num_{level}_boundaries"] for row in stats)
        summary[f"total_{level}_boundaries"] = count
        summary[f"{level}_boundary_rate_hz"] = count / duration
    return summary
