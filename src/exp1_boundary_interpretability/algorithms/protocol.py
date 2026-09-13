"""Fixed protocol definitions for the direct semantic-token ASR matrix."""

from __future__ import annotations

from dataclasses import dataclass


TARGET_RATE_HZ = 6.25
MAX_SEGMENT_SECONDS = 0.640
SEED = 42

COMMON_SYSTEMS = (
    "uniform",
    "similarity",
    "codecslime_dp",
    "ple",
    "peak_detection",
)

EXTENDED_SYSTEMS = COMMON_SYSTEMS + (
    "hubert_kmeans",
    "dcdit_1d",
    "elastic_time_greedy",
    "elastic_time_dp",
)


@dataclass(frozen=True)
class EncoderSpec:
    encoder_id: str
    family: str
    representation: str
    model_path: str | None


ENCODERS = {
    "sensevoice_l49": EncoderSpec(
        "sensevoice_l49", "sensevoice", "sensevoice_main_l49", "models/SenseVoiceSmall"
    ),
    "whisper_tiny_l4": EncoderSpec(
        "whisper_tiny_l4", "whisper", "whisper_tiny_l4", "models/whisper-tiny"
    ),
    "hubert_base_l8": EncoderSpec(
        "hubert_base_l8", "hubert", "hubert_base_l8", "models/hubert-base-ls960"
    ),
    "wavlm_base_l11": EncoderSpec(
        "wavlm_base_l11", "wavlm", "wavlm_base_l11", "models/wavlm-base-plus-sv"
    ),
    "encodec": EncoderSpec(
        "encodec", "encodec", "encodec_24khz_encoder", None
    ),
    "dac": EncoderSpec(
        "dac", "dac", "dac_24khz_8kbps_latents", None
    ),
    "mimi": EncoderSpec(
        "mimi", "mimi", "mimi_prequantizer_latents", "models/mimi"
    ),
    "logmel": EncoderSpec(
        "logmel", "logmel", "logmel_10ms", None
    ),
    "sensevoice_l0": EncoderSpec(
        "sensevoice_l0", "sensevoice", "sensevoice_main_l0", "models/SenseVoiceSmall"
    ),
    "sensevoice_l9": EncoderSpec(
        "sensevoice_l9", "sensevoice", "sensevoice_main_l9", "models/SenseVoiceSmall"
    ),
    "sensevoice_l19": EncoderSpec(
        "sensevoice_l19", "sensevoice", "sensevoice_main_l19", "models/SenseVoiceSmall"
    ),
    "sensevoice_l29": EncoderSpec(
        "sensevoice_l29", "sensevoice", "sensevoice_main_l29", "models/SenseVoiceSmall"
    ),
    "sensevoice_l39": EncoderSpec(
        "sensevoice_l39", "sensevoice", "sensevoice_main_l39", "models/SenseVoiceSmall"
    ),
}

SENSEVOICE_CACHE_LAYERS = (0, 9, 19, 29, 39, 49)


def physical_max_span(frames: int, duration_sec: float) -> int:
    if frames < 1 or duration_sec <= 0:
        raise ValueError("frames and duration must be positive")
    return max(1, int(round(frames / duration_sec * MAX_SEGMENT_SECONDS)))


def requested_segments(duration_sec: float) -> int:
    if duration_sec <= 0:
        raise ValueError("duration must be positive")
    return max(1, int(round(duration_sec * TARGET_RATE_HZ)))
