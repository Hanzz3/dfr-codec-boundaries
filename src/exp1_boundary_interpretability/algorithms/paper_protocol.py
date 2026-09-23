"""Canonical method and rate definitions for the final manuscript."""

from __future__ import annotations


PAPER_SYSTEMS = (
    "uniform",
    "flexicodec_threshold",
    "codecslime_dp",
    "ple",
    "tadpc_style",
    "tfc_style_entropy",
    "atome_style",
)

PAPER_SYSTEM_LABELS = {
    "uniform": "Uniform",
    "flexicodec_threshold": "Similarity",
    "codecslime_dp": "CodecSlime",
    "ple": "PLE",
    "tadpc_style": "TADPC",
    "tfc_style_entropy": "Entropy-Mass",
    "atome_style": "A-ToMe",
}

# The shared waveform decoder was adapted before the final seven-method
# evaluation set was fixed. Keep this historical mixture explicit so the
# released training code does not rewrite the experimental provenance.
DECODER_TRAINING_SYSTEMS = (
    "uniform",
    "flexicodec_threshold",
    "codecslime_dp",
    "ple",
    "elastic_time_greedy",
    "elastic_time_dp",
    "dcdit_1d",
    "atome_style",
    "tadpc_style",
)

DECODER_TRAINING_RATES_HZ = (3.0, 6.25, 8.33, 10.0)
PAPER_CORRELATION_RATES_HZ = (6.25, 8.33, 10.0)
PAPER_PRIMARY_RATE_HZ = 6.25

