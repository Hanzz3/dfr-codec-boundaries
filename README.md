# Interpreting and Evaluating Dynamic-Rate Speech Codec Boundaries

Implementation code for the boundary-interpretability and
reconstruction-utility experiments in the accompanying ICASSP manuscript.

## Final paper protocol

The paper compares seven methods:

| Paper name | Internal identifier |
|---|---|
| Uniform | `uniform` |
| Similarity | `flexicodec_threshold` |
| CodecSlime | `codecslime_dp` |
| PLE | `ple` |
| TADPC | `tadpc_style` |
| Entropy-Mass | `tfc_style_entropy` |
| A-ToMe | `atome_style` |

The canonical method and rate definitions are in
`src/exp1_boundary_interpretability/algorithms/paper_protocol.py`.
The shared waveform decoder was adapted earlier with a historical mixture of
nine allocation rules at 3, 6.25, 8.33, and 10 Hz. That training mixture is
preserved in `DECODER_TRAINING_SYSTEMS` for provenance, but it is not the final
seven-method comparison reported in the paper.

## Repository layout

```text
src/
  exp1_boundary_interpretability/   reference tracks and boundary evaluation
  exp2_reconstruction_utility/      decoder, ASR, cross-rate, matched-rate code
tests/                              CPU unit and release-integrity tests
environments/                       pinned CPU analysis environment
run_implementation.py              local PYTHONPATH wrapper
```

Generated data, cached features, model weights, reconstructed audio, and paper
result files are intentionally excluded. TIMIT, LibriTTS, pretrained models,
and checkpoints must be obtained separately under their respective licenses.

## CPU environment and tests

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/requirements-analysis.txt
python -m pytest -q
```

Full decoder and ASR runs additionally require PyTorch, Transformers, PEFT,
FlexiCodec, DAC/audiotools, the pretrained models named in the paper, and a
CUDA environment compatible with those packages.

## Paper result map

| Paper result | Main implementation |
|---|---|
| Table 1: six reference-boundary tracks | `evaluate_boundaries.py`, `evaluate_entropy_mass.py`, `build_paper_seven.py` |
| Table 2: q1/q1:8 reconstruction and direct q1 ASR | `reconstruction/evaluate.py`, `reconstruction/merge_shards.py`, `direct_q1_asr/` |
| Table 3: cross-rate absolute Spearman correlation | `cross_rate/merge_reconstruction.py`, `cross_rate/summarize_paper_srcc.py` |
| Table 4: matched-rate phoneme/syllable test | `matched_rate/`, `matched_rate/summarize_paper_table.py` |

Figure 2 uses the layer-wise output of EXP1. Figure 3 visualizes the 6.25-Hz
alignment and NMSE values summarized by the Table 1/2 pipeline.

## Entry points

Use the wrapper to expose each experiment's local modules:

```bash
python run_implementation.py --profile exp1 \
  src/exp1_boundary_interpretability/evaluate_boundaries.py --help

python run_implementation.py --profile exp1 \
  src/exp1_boundary_interpretability/evaluate_entropy_mass.py --help

python run_implementation.py --profile cross-rate \
  src/exp2_reconstruction_utility/cross_rate/build_paper_seven.py --help

python run_implementation.py --profile reconstruction \
  src/exp2_reconstruction_utility/reconstruction/evaluate.py --help

python run_implementation.py --profile direct-asr \
  src/exp2_reconstruction_utility/direct_q1_asr/evaluate.py --help

python run_implementation.py --profile cross-rate \
  src/exp2_reconstruction_utility/cross_rate/summarize_paper_srcc.py --help

python run_implementation.py --profile matched-rate \
  src/exp2_reconstruction_utility/matched_rate/summarize_paper_table.py --help
```

## Entropy-Mass and final seven-method bundle

Entropy-Mass computes a 64-bin soft-histogram entropy value for each native
codec frame, then uses exact-budget dynamic programming to partition the
utterance into spans with approximately equal entropy mass. The maximum span
is 640 ms. For each reported rate, first materialize Entropy-Mass and then
combine it with the six retained methods from the archived selector bundle:

```bash
python run_implementation.py --profile exp1 \
  src/exp1_boundary_interpretability/evaluate_entropy_mass.py \
  --feature-index FEATURE_INDEX.jsonl \
  --reference-tracks REFERENCE_TRACKS.jsonl \
  --target-rate-hz 6.25 --target-rate-hz 8.33 --target-rate-hz 10 \
  --output-dir OUTPUT/entropy_mass

python run_implementation.py --profile cross-rate \
  src/exp2_reconstruction_utility/cross_rate/build_paper_seven.py \
  --base-dir OUTPUT/archived_nine/rate_6p25 \
  --entropy-dir OUTPUT/entropy_mass/rate_6p25 \
  --target-rate-hz 6.25 \
  --output-dir OUTPUT/paper_seven/rate_6p25
```

Repeat the merge for 8.33 and 10 Hz. The merge validates the full Cartesian
product of 1,680 utterances, seven methods, and six reference tracks before
writing an accepted bundle.

## Notes on names and provenance

`Entropy-Mass` is the paper name for the released
`tfc_style_entropy` adapter. It is inspired by temporal-entropy allocation but
is not a reimplementation of the learned multi-resolution TFC codec. Likewise,
the other allocation rules are controlled adapters used under the common
FlexiCodec interface; see `THIRD_PARTY_NOTICES.md`.
