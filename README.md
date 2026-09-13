# Interpreting and Evaluating Dynamic-Rate Speech Codec Boundaries

Implementation code for boundary interpretability and reconstruction-utility
experiments with dynamic-frame-rate speech codecs.

## Repository layout

```text
src/
  exp1_boundary_interpretability/
  exp2_reconstruction_utility/
tests/
environments/
run_implementation.py
```

The EXP1 modules construct reference tracks, evaluate boundary alignment, and
implement dynamic-rate selectors. The EXP2 modules fine-tune and evaluate the
decoder, run direct semantic-token ASR, evaluate multiple frame rates, and run
matched-rate boundary tests.

## CPU environment and tests

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/requirements-analysis.txt
python -m pytest -q
```

## Entry points

Use the wrapper to add the appropriate local modules to `PYTHONPATH`:

```bash
python run_implementation.py --profile exp1 \
  src/exp1_boundary_interpretability/evaluate_boundaries.py --help

python run_implementation.py --profile decoder \
  src/exp2_reconstruction_utility/decoder_finetuning/train_shared_decoder.py --help

python run_implementation.py --profile reconstruction \
  src/exp2_reconstruction_utility/reconstruction/evaluate.py --help

python run_implementation.py --profile direct-asr \
  src/exp2_reconstruction_utility/direct_q1_asr/train.py --help

python run_implementation.py --profile matched-rate \
  src/exp2_reconstruction_utility/matched_rate/build_reference_partitions.py --help

python run_implementation.py --profile matched-rate \
  src/exp2_reconstruction_utility/matched_rate/evaluate_decoder.py --help
```

Full training and inference require separately obtained datasets, pretrained
models, model checkpoints, and GPU dependencies. These assets are not included.
