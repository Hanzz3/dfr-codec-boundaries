import ast
import json
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
from metrics import feature_nmse
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


def test_direct_q1_asr_defaults_cover_nine_algorithms():
    source = (ROOT / "src/exp2_reconstruction_utility/direct_q1_asr/core.py").read_text()
    module = ast.parse(source)
    assignment = next(
        node for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SYSTEMS" for target in node.targets)
    )
    systems = ast.literal_eval(assignment.value)
    assert len(systems) == 9
    assert systems[-2:] == ("atome_style", "tadpc_style")


def test_configs_follow_paper_experiment_layout():
    configs = sorted((ROOT / "configs").glob("*/*.json"))
    assert [path.parent.name for path in configs] == [
        "exp1_boundary_interpretability",
        "exp2_reconstruction_utility",
        "exp2_reconstruction_utility",
        "exp2_reconstruction_utility",
    ]
    payloads = [json.loads(path.read_text()) for path in configs]
    assert all(payload["paper_experiment"].startswith(("EXP1:", "EXP2:")) for payload in payloads)
    cross_rate = json.loads((ROOT / "configs/exp2_reconstruction_utility/cross_rate_evaluation.json").read_text())
    assert cross_rate["rates_hz"] == [6.25, 8.33, 10.0]


def test_release_is_code_only():
    assert not any((ROOT / name).exists() for name in ("data", "results", "tables", "figures"))
    names = [path.relative_to(ROOT).as_posix().lower() for path in ROOT.rglob("*")]
    assert all("/exp3" not in f"/{name}" and "/exp4" not in f"/{name}" for name in names)
    excluded = {".bin", ".ckpt", ".flac", ".jpeg", ".jpg", ".mp3", ".pdf", ".png", ".pt", ".pth", ".svg", ".wav"}
    assert not any(path.suffix.lower() in excluded for path in ROOT.rglob("*"))
