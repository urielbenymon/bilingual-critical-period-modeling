# Corrected German Probe Evaluation

## Purpose

This directory preserves corrected evaluation outputs for the
English L1 -> German/Finnish L2 representation experiment.

The original model checkpoints were reused. No model retraining
was performed during this correction.

## Original evaluation issue

The original hidden-state extraction used a maximum sequence
length of 128 tokens.

German probe sentence ID 19 contained 177 tokens and was
therefore truncated to 128 tokens.

The other 19 German probes were unaffected.

## Correction

All five German checkpoint representations were re-extracted
using `--max_length 256` and the original German probe file:

`data/class_probes/de_probes.txt`

The five corrected extraction files are:

- `de_l1_baseline_hidden_states.pt`
- `de_seq_epoch1_hidden_states.pt`
- `de_seq_final_hidden_states.pt`
- `de_ewc_epoch1_hidden_states.pt`
- `de_ewc_final_hidden_states.pt`

All corrected outputs were saved separately from the original
evaluation files.

## Reproduction

Run commands from the repository root in the project's
configured Python environment.

### 1. Extract corrected German hidden states

Run the following commands from the repository root:

```bash
python src/analysis/extract_hidden_states.py \
  --checkpoint checkpoints/gpt2-class_l1-en-sequential-42-0808-1-class-l1 \
  --probe_file data/class_probes/de_probes.txt \
  --output_file analysis_outputs/class/corrected_de_256/de_l1_baseline_hidden_states.pt \
  --max_length 256

python src/analysis/extract_hidden_states.py \
  --checkpoint checkpoints/gpt2-class_l2-ende-sequential-42-0808-2-class-de/checkpoint-85 \
  --probe_file data/class_probes/de_probes.txt \
  --output_file analysis_outputs/class/corrected_de_256/de_seq_epoch1_hidden_states.pt \
  --max_length 256

python src/analysis/extract_hidden_states.py \
  --checkpoint checkpoints/gpt2-class_l2-ende-sequential-42-0808-2-class-de \
  --probe_file data/class_probes/de_probes.txt \
  --output_file analysis_outputs/class/corrected_de_256/de_seq_final_hidden_states.pt \
  --max_length 256

python src/analysis/extract_hidden_states.py \
  --checkpoint checkpoints/gpt2-class_l2-ende-sequential-42-0808-2-ewc-class-de-ewc/checkpoint-85 \
  --probe_file data/class_probes/de_probes.txt \
  --output_file analysis_outputs/class/corrected_de_256/de_ewc_epoch1_hidden_states.pt \
  --max_length 256

python src/analysis/extract_hidden_states.py \
  --checkpoint checkpoints/gpt2-class_l2-ende-sequential-42-0808-2-ewc-class-de-ewc \
  --probe_file data/class_probes/de_probes.txt \
  --output_file analysis_outputs/class/corrected_de_256/de_ewc_final_hidden_states.pt \
  --max_length 256
```

These are the exact checkpoint paths recorded in the corrected
`.pt` extraction metadata.

### 2. Recompute L2 trajectory similarities

```bash
python src/analysis/compare_l2_trajectories.py \
  --de_baseline analysis_outputs/class/corrected_de_256/de_l1_baseline_hidden_states.pt \
  --de_seq_epoch1 analysis_outputs/class/corrected_de_256/de_seq_epoch1_hidden_states.pt \
  --de_seq_final analysis_outputs/class/corrected_de_256/de_seq_final_hidden_states.pt \
  --de_ewc_epoch1 analysis_outputs/class/corrected_de_256/de_ewc_epoch1_hidden_states.pt \
  --de_ewc_final analysis_outputs/class/corrected_de_256/de_ewc_final_hidden_states.pt \
  --fi_baseline analysis_outputs/class/fi_l1_baseline_hidden_states.pt \
  --fi_seq_epoch1 analysis_outputs/class/fi_seq_epoch1_hidden_states.pt \
  --fi_seq_final analysis_outputs/class/fi_seq_final_hidden_states.pt \
  --fi_ewc_epoch1 analysis_outputs/class/fi_ewc_epoch1_hidden_states.pt \
  --fi_ewc_final analysis_outputs/class/fi_ewc_final_hidden_states.pt \
  --output_sentence_csv analysis_outputs/class/corrected_de_256/l2_trajectory_sentence_results.csv \
  --output_summary_csv analysis_outputs/class/corrected_de_256/l2_trajectory_summary.csv
```

The Finnish extraction files are reused unchanged from the
original evaluation.

### 3. Recompute statistical contrasts

```bash
python src/analysis/analyze_trajectory_contrasts.py \
  --l1_csv analysis_outputs/class/en_l1_trajectory_sentence_results.csv \
  --l2_csv analysis_outputs/class/corrected_de_256/l2_trajectory_sentence_results.csv \
  --output_dir analysis_outputs/class/corrected_de_256/contrasts
```

### 4. Regenerate reporting tables

```bash
python src/analysis/make_results_table.py \
  --contrast_dir analysis_outputs/class/corrected_de_256/contrasts \
  --output_dir analysis_outputs/class/corrected_de_256/results_tables
```

## Validation

The correction was independently checked against the original
saved evaluation outputs.

- All five German extraction files contain 20 probe sentences.
- Only German sentence ID 19 changed.
- Its processed token count increased from 128 to 177.
- Hidden-state vectors for German sentence IDs 0-18 were identical.
- Finnish and English L1 evaluation outputs remained unchanged.
- Twelve of 480 sentence-level L2 similarity rows changed.
- Twenty-four of 72 statistical contrast rows changed numerically.
- No contrast changed whether its 95% bootstrap CI included zero.
- Sixteen of 48 compact reporting rows changed numerically.

These checks establish consistency of the corrected evaluation
with the saved checkpoints and original analysis pipeline.
They do not establish independent reproduction of model training.

## Interpretation

Confidence intervals quantify variation across probe sentences
within the fixed trained-model runs. They do not quantify
uncertainty across independently trained model seeds.

Cosine similarity to the English-only baseline measures
representational alignment, not language proficiency or
human critical-period effects.

## Preservation

The original evaluation outputs remain in their existing
locations. Files in this directory are corrected evaluation
artifacts, not replacements for the original records.
