# Interpreting and Evaluating Dynamic-Rate Speech Codec Boundaries

Public repository: <https://github.com/WangHanZJU/dfr-codec-boundaries>

Implementation code for the two experiments in the current manuscript:

- `src/exp1_boundary_interpretability/`: reference-track construction,
  boundary matching, DFR algorithms, and rate calibration.
- `src/exp2_reconstruction_utility/`: decoder fine-tuning, q1/q8 reconstruction,
  direct q1 ASR, cross-rate evaluation, and matched-rate tests.

The repository intentionally contains no `exp3` or `exp4` directories. Those
were server workflow names and do not correspond to the final paper. Aggregate
results, table builders, figure builders, and figure-only data are excluded.

## Code Check

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/requirements-analysis.txt
python -m pytest -q
```

## Repository Layout

```text
configs/
  exp1_boundary_interpretability/
  exp2_reconstruction_utility/
src/
  exp1_boundary_interpretability/
  exp2_reconstruction_utility/
tests/
docs/
environments/
```

The four configuration files are named by their role in the paper; dated
server run names have been removed. The 3-Hz setting remains only in decoder
fine-tuning because it was one of the decoder's training conditions. Table 3
uses 6.25, 8.33, and 10 Hz.

## Scope

The retained implementation modules and CPU tests are verified from a clean
extracted archive. Full model training and inference need separately obtained TIMIT/LibriTTS,
pretrained models, accepted checkpoints, and GPU dependencies. No speech,
transcripts, hypotheses, aggregate experiment results, model weights,
credentials, or private server paths are distributed.

See `docs/IMPLEMENTATION_DETAILS.md` and `docs/REPRODUCTION.md`. Model
checkpoints are not included in this repository.
