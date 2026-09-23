from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXP1 = ROOT / "src/exp1_boundary_interpretability"
ALGORITHMS = EXP1 / "algorithms"
RECONSTRUCTION = ROOT / "src/exp2_reconstruction_utility/reconstruction"
for path in (EXP1, ALGORITHMS, RECONSTRUCTION):
    sys.path.insert(0, str(path))

import boundary_metrics
import reference_tracks
from entropy_mass import equal_entropy_starts, waveform_frame_entropy
from metrics import feature_nmse
from paper_protocol import DECODER_TRAINING_SYSTEMS, PAPER_SYSTEMS, PAPER_SYSTEM_LABELS
from supplement_selectors import atome_style_starts, tadpc_style_starts


def test_boundary_matcher_is_monotonic_and_one_to_one():
    matcher = boundary_metrics.match_boundaries
    assert matcher([0.05, 0.13], [0.0, 0.08], 0.06)[:3] == (2, 0, 0)
    assert matcher([0.125, 0.126], [0.125], 0.04)[:3] == (1, 1, 0)
    assert matcher([], [0.1], 0.04)[:3] == (0, 0, 1)
    assert matcher([0.125], [0.0], 0.125)[:3] == (1, 0, 0)


def test_reference_interval_boundary_uses_next_interval_start():
    intervals = [
        reference_tracks.Interval(0.0, 0.09, "aa", "synthetic"),
        reference_tracks.Interval(0.10, 0.19, "t", "synthetic"),
    ]
    boundaries, _ = reference_tracks.interval_boundaries(intervals)
    assert boundaries == [0.10]


def test_syllable_final_consonant_rule():
    intervals = [
        reference_tracks.Interval(index * 0.1, (index + 1) * 0.1, label, "synthetic")
        for index, label in enumerate(["aa", "t", "r", "iy"])
    ]
    word = reference_tracks.Interval(0, 0.4, "synthetic", "synthetic")
    spans = reference_tracks.syllable_spans_for_word(intervals, word)
    assert [span.begin for span in spans] == pytest.approx([0, 0.2])


def test_short_voicing_run_smoothing():
    short = np.array([True] * 4 + [False] * 3 + [True] * 4)
    assert reference_tracks.smooth_binary(short, min_run=4).all()
    retained = np.array([True] * 4 + [False] * 4 + [True] * 4)
    assert np.array_equal(retained, reference_tracks.smooth_binary(retained, min_run=4))


def test_synthetic_acoustic_tracks_are_valid():
    sample_rate = 16000
    time = np.arange(sample_rate * 2) / sample_rate
    waveform = np.where(
        (time > 0.3) & (time < 1.4),
        0.2 * np.sin(2 * np.pi * 160 * time),
        0,
    ).astype(np.float32)
    for function in (
        reference_tracks.detect_vuv_boundaries,
        reference_tracks.detect_acoustic_event_boundaries,
    ):
        times = function(waveform, sample_rate)
        assert all(np.isfinite(times))
        assert all(0 < value < 2 for value in times)
        assert times == sorted(set(times))


def test_nmse_before_quantization():
    features = np.array([[1.0, 0.0], [3.0, 0.0]])
    assert feature_nmse(features, [0]) == pytest.approx(0.2)
    assert feature_nmse(features, [0, 1]) == 0
    assert feature_nmse(np.zeros((3, 2)), [0]) == 0


def test_atome_and_tadpc_return_valid_partitions():
    features = np.random.default_rng(42).normal(size=(24, 4))
    for result in (atome_style_starts(features), tadpc_style_starts(features)):
        assert result.starts[0] == 0
        assert all(left < right for left, right in zip(result.starts, result.starts[1:]))
        assert result.starts[-1] < len(features)


def test_final_paper_systems_match_the_manuscript():
    assert len(PAPER_SYSTEMS) == 7
    assert PAPER_SYSTEMS[0] == "uniform"
    assert PAPER_SYSTEM_LABELS["tfc_style_entropy"] == "Entropy-Mass"
    assert set(PAPER_SYSTEM_LABELS) == set(PAPER_SYSTEMS)


def test_decoder_training_provenance_remains_explicit():
    assert len(DECODER_TRAINING_SYSTEMS) == 9
    assert "tfc_style_entropy" not in DECODER_TRAINING_SYSTEMS
    assert {"elastic_time_greedy", "elastic_time_dp", "dcdit_1d"}.issubset(
        DECODER_TRAINING_SYSTEMS
    )


def test_entropy_mass_exact_budget_partition():
    starts = equal_entropy_starts(np.ones(12), segments=3, max_span=4)
    assert starts == [0, 4, 8]
    lengths = np.diff([*starts, 12])
    assert len(starts) == 3
    assert max(lengths) <= 4


def test_waveform_frame_entropy_is_finite():
    sample_rate = 16000
    waveform = np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate)
    times = (np.arange(12, dtype=np.float64) + 0.5) / 12
    entropy = waveform_frame_entropy(waveform, sample_rate, times, 1.0)
    assert entropy.shape == (12,)
    assert np.isfinite(entropy).all()
    assert (entropy > 0).all()


def test_repository_excludes_generated_artifacts():
    assert not any((ROOT / name).exists() for name in ("data", "results", "tables", "figures"))
    names = [path.relative_to(ROOT).as_posix().lower() for path in ROOT.rglob("*")]
    assert all("/exp3" not in f"/{name}" and "/exp4" not in f"/{name}" for name in names)
    excluded = {".bin", ".ckpt", ".flac", ".jpeg", ".jpg", ".mp3", ".pdf", ".png", ".pt", ".pth", ".svg", ".wav"}
    assert not any(path.suffix.lower() in excluded for path in ROOT.rglob("*"))
