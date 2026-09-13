# Reproduction

## EXP1 implementation entry points

```bash
python run_implementation.py --profile exp1 \
  src/exp1_boundary_interpretability/build_reference_tracks.py --help

python run_implementation.py --profile exp1 \
  src/exp1_boundary_interpretability/algorithms/calibrate_atome_tadpc.py --help

python run_implementation.py --profile exp1 \
  src/exp1_boundary_interpretability/evaluate_boundaries.py --help
```

Full execution requires licensed TIMIT, MFA alignments, a cached GPT-2
tokenizer, SenseVoice/FlexiCodec features, and learned predictor artifacts.

## EXP2 implementation entry points

```bash
python run_implementation.py --profile decoder \
  src/exp2_reconstruction_utility/decoder_finetuning/train_shared_decoder.py --help

python run_implementation.py --profile reconstruction \
  src/exp2_reconstruction_utility/reconstruction/evaluate.py --help

python run_implementation.py --profile direct-asr \
  src/exp2_reconstruction_utility/direct_q1_asr/train.py --help

python run_implementation.py --profile matched-rate \
  src/exp2_reconstruction_utility/matched_rate/build_reference_partitions.py --help
```

These need separately obtained LibriTTS/TIMIT, pretrained FlexiCodec,
SenseVoice, Whisper Small, WavLM Base Plus SV, Qwen2.5-0.5B, accepted predictor
weights, decoder checkpoint, exact manifests, and a compatible GPU environment.

The historical GPU requirements are retained for reference but are not a full
lockfile. Full clean-machine training/inference was not rerun while packaging;
the archive provides the experiment implementation but does not distribute
aggregate results or claim one-command end-to-end GPU reproduction. The raw
outputs produced by these entry points can be assembled into the paper tables
using ordinary dataframe operations.
