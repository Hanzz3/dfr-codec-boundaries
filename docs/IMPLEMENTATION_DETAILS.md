# Implementation details

## EXP1: Boundary interpretability

All 1,680 TIMIT TEST utterances are evaluated. MFA 3.0 phone and word intervals
are used as linguistic reference sources and cross-checked against native TIMIT
annotations. The release does not call these manually annotated ground truth;
they are derived reference tracks.

`reference_tracks.py` constructs six tracks. Phoneme boundaries include silence
transitions; word boundaries retain transitions between non-silence words.
Within each word, vowels define syllable nuclei. The final consonant between
successive nuclei starts the next syllable; without an intervening consonant,
the next vowel starts it. A word with at most one vowel forms one syllable.
GPT-2 token character offsets are linearly mapped to MFA word intervals without
phone snapping. Both syllable and BPE tracks retain unit onsets after time zero.

V/UV uses centered 25-ms windows and a 10-ms hop at 16 kHz. A frame is voiced
when `dB > max(-45, percentile(dB,30))` and
`ZCR < min(0.22, percentile(ZCR,75))`. Runs shorter than 40 ms are changed only
when both neighbors agree. Acoustic events use per-utterance standardized
log-RMS, ZCR, spectral centroid, and flatness; adjacent cosine-distance peaks
are selected with a nominal 5-Hz budget and 50-ms spacing, combined with V/UV
transitions, and deduplicated within 20 ms.

Boundary F1 uses maximum-cardinality monotonic one-to-one matching at an
inclusive 40-ms tolerance. TP, FP, and FN are pooled over utterances before
computing `2TP/(2TP+FP+FN)`. Each reference track is evaluated independently.

Nine algorithms are compared on frozen SenseVoice layer-49 features mapped to
the FlexiCodec grid: Uniform, Similarity, CodecSlime, PLE, Elastic-G,
Elastic-DP, DC-DiT, A-ToMe, and TADPC. Their controlled allocation adapters are
under `algorithms/`. Elastic predictors use three 128-wide GRUCell/SwiGLU layers
and an eight-frame horizon; DC-DiT uses a 1-D width-128 local predictor with
center context 2. Learned predictors use LibriTTS train-clean-100, 50 epochs,
AdamW, batch 64, learning rate 3e-4, seed 42, with dev-clean selection.

Table 1 uses nominal 6.25 Hz; realized algorithm rates are 6.17-6.28 Hz.
Configuration is in `configs/exp1_boundary_interpretability/table1_evaluation.json`.

## EXP2: Boundary reconstruction utility

For a given algorithm and utterance, q1 and q8 use identical selected spans.
The shared decoder fine-tuning updates the ConvNeXt decoder, bottleneck
transformer, and DAC waveform decoder while freezing the encoder and
quantizers. A new DAC discriminator is trained. The objective is adversarial
loss plus 2x feature matching, 15x multiscale log-mel loss, and 0.1x frozen
speaker-embedding cosine loss. Training balances nine algorithms across 3,
6.25, 8.33, and 10 Hz; global batch 4, seed 42, generator/discriminator AdamW
rates 1e-5/1e-4, betas (0.8,0.9). The accepted paper checkpoint is step 1,000
within the 10,000-step schedule. See `decoder_finetuning.json`.

Waveform WER uses Whisper Small on reconstructed audio. Corpus WER is total word
errors divided by total reference words: 14,552 words per algorithm on TIMIT
TEST. PESQ is wideband at 16 kHz on q8. SpkSim is cosine similarity from
`microsoft/wavlm-base-plus-sv` embeddings on q8.

NMSE is measured before quantization. Each selected segment mean is repeated
over its original frames; mean squared feature error is divided by the original
feature mean-square magnitude, floored at 1e-8, then averaged over utterances.

Direct q1 ASR performs a shared q8 encode, extracts the semantic component,
passes semantic latents and log-duration features through a learned projector,
and trains Qwen2.5-0.5B for transcription. Prompt/audio positions are masked
from language-model loss. The code uses LoRA; source defaults are rank 16,
alpha 32, dropout .05, projector LR 2e-4 and LoRA LR 1e-4. The accepted
checkpoint was selected on LibriTTS dev, never TIMIT TEST.

The cross-rate evaluation runs the nine algorithms separately at 6.25, 8.33,
and 10 Hz. The resulting boundary and reconstruction summaries support the
syllable/phoneme Spearman analysis reported in Table 3.

Tables 4 and 5 use the original frozen decoder, unlike Tables 2 and 3. Table 4
tests phoneme-derived partitions near 8.06 Hz. Table 5 tests syllable-derived
partitions near 4.47 Hz. Its CodecSlime result is 1,641/14,552 = 11.2768%,
displayed as 11.28%. CodecSlime has 23,215 segments versus 22,875 for the other
Table 5 rows, so it is nominal-rate controlled rather than exactly count matched.

These implementations adapt published boundary-allocation rules to a common
codec interface; they are not full end-to-end reproductions of each original
codec system.
