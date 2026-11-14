# Modeling Bilingual Learning Through LLM Training

Code and analysis outputs for the manuscript **"Modeling Bilingual Learning Through LLM Training: A Critical-Period Exploration"** by Uriel Benymon, Tristan Whipps, and Armita Ghobadi.

- **Authors:** Uriel Benymon and Tristan Whipps (equal contribution)
- **Principal investigator:** Armita Ghobadi, Department of Psychology, Barnard College
- **Contact:** Uriel Benymon (Department of Computer Science, Columbia University)
- **Preprint:** forthcoming on PsyArXiv

## Overview

This repository extends the code released with Constantinescu et al. (2025), *Investigating Critical Period Effects in Language Acquisition through Neural Language Models* (Transactions of the Association for Computational Linguistics, 13, 96–120; original code: https://github.com/iconstantinescu/lm-critical-period).

The study trains a small two-layer GPT-2 model on English (L1) and then on German (a relatively related L2) or Finnish (a relatively distant L2), under either Sequential training or Elastic Weight Consolidation (EWC). Mean-pooled hidden states for 20 held-out probe sentences per language are compared with those of the English-only model using cosine similarity at the embedding layer and both transformer layers, after one and two L2 epochs. All runs use seed 42.

## Repository layout (study-specific files)

| Path | Contents |
| --- | --- |
| `src/data_processing/build_class_experiment.py` | Builds the matched English, German, and Finnish corpora (3,000 OPUS Books sentences per language; 2,490 train / 255 validation / 255 test) |
| `src/learn/configs/gpt2_class_l1.txt`, `gpt2_class_l2.txt` | Training configurations for the English L1 model and the German/Finnish L2 runs |
| `src/learn/train_gpt2_local.sh` | Single-process (CPU) variant of the original GPT-2 training launcher |
| `src/analysis/extract_hidden_states.py` | Extracts mean-pooled hidden states for a probe file from a saved checkpoint |
| `src/analysis/compare_l1_trajectories.py`, `compare_l2_trajectories.py` | Computes sentence-level cosine similarity to the pre-L2 baseline |
| `src/analysis/analyze_trajectory_contrasts.py` | Bootstrap confidence intervals and effect sizes for stage, training-condition, and language contrasts |
| `src/analysis/make_results_table.py` | Generates the compact reporting tables |
| `src/analysis/plot_trajectories.py` | Plots the L1 retention and L2 trajectories |
| `analysis_outputs/class/` | Saved hidden states, similarity results, contrasts, tables, and figures from the original evaluation |
| `analysis_outputs/class/corrected_de_256/` | Corrected German evaluation (maximum sequence length 256); see its `README.md` |

**The results reported in the manuscript use the corrected German outputs in `analysis_outputs/class/corrected_de_256/`.** The original evaluation truncated one German probe sentence (ID 19, 177 tokens) at 128 tokens. The corrected directory documents the re-extraction and the exact commands to reproduce it. The original outputs are kept for the record.

Training corpora, probe files, tokenizers, and model checkpoints are not tracked in this repository (`data/` and `checkpoints/` are excluded by `.gitignore`).

## Limitations

The data come from a single training run per condition. Confidence intervals describe variation across the 20 probe sentences within that run, not across independently trained models. Cosine similarity to the English-only baseline measures representational alignment, not language proficiency or human critical-period effects. See the manuscript for the full discussion.

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`) and the original work by Constantinescu et al. (2025).

---

## Upstream documentation

The sections below are carried over from the original repository and describe its general training pipeline.

### Language Models Critical Period


## Creating environment on Euler

Make sure you are in the root of your project.

```
module load eth_proxy gcc/8.2.0 python_gpu/3.9.9
python -m venv --system-site-packages venv
source ./venv/bin/activate

pip install --upgrade pip
```

Whenever you install a new package, make sure your venv is activated!

To install new packages:
```
pip install --no-cache-dir <package_names>
```

If you need a different version of an existing package, then you can also install it in your virtual environment. For instance for installing a newer numpy version:

```
OPENBLAS=$OPENBLAS_ROOT/lib/libopenblas.so pip install --ignore-installed --no-deps numpy==1.20.0
```

List of existing packages on Euler:  https://scicomp.ethz.ch/wiki/Python_on_Euler

To install required packages for this project:

```
OPENBLAS=$OPENBLAS_ROOT/lib/libopenblas.so pip install --ignore-installed --no-cache-dir -r requirements.txt
```

You might also want to login to your wandb account (only once): ```wandb login```

## Running the scripts on Euler

Before running the scripts, make sure you modify the *.euler files accordingly:
* Modify SBATCH options (e.g. resource requests) 
* Modify the venv source path: ```source <path_to_project>/venv/bin/activate```

Run the scripts from the project root.

```
### See help for all the available options ###
./scripts/train.sh -h

### Train a model with default configuration ###
./scripts/train.sh

### Train a model with custom configuration ###
./scripts/train.sh -n <model_name> -d <dataset> -1 <lang1> -s <seed> -p <project_name>
```

Run a sweep:

```
1. Create a new sweep yaml configuration in ./src/learn/configs folder
2. ./scripts/sweep.sh -n <config_name> -1 <lang1> -p <project_name>
3. ./scripts/train.sh -n <model_name> -1 <lang1> -w <sweep_id>

```

## Creating environment locally with Conda

First, create a conda environment with
```bash
$ conda env create -f util/environment.yml
```
Then activate the environment with `source util/activate.sh` and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ # conda install pytorch torchvision cpuonly -c pytorch
$ pip install transformers[torch]
$ pip install datasets
$ pip install evaluate
$ pip install wandb
```


## Running the scripts Locally

First, train the model in lang1 with command:
```bash
$ MODE=sequential MODEL=gpt2 CONFIG=config11 DATASET=unified_clean LANG1=fi SEED=7 source src/learn/train_model.sh
```

And then run to finetune the model in the lang2:
```bash
$ MODE=sequential MODEL=gpt2 CONFIG=config11 DATASET=unified_clean LANG1=fi LANG2=en SEED=7 CHECKPOINT=gpt2-config11-fi-sequential-7-1109-1 source src/learn/train_model.sh
```