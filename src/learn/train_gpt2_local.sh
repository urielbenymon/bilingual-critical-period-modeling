#!/bin/bash

# Purpose: Launches GPT-2 training/evaluation through run_clm.py using the dataset, tokenizer, model name, seed, and other settings supplied by the broader training pipeline.
# Local changes: Replaces the original 8-process GPU launcher with one Python process and removes FP16/NCCL/DDP options because the Legion Go is currently using CPU-only PyTorch.

# Collects any extra command-line arguments passed into this script so they can be forwarded to run_clm.py.
EXTRA_FLAGS="$@"

# Prints the extra arguments to the terminal so the active training options are visible.
echo $EXTRA_FLAGS

# CHANGED FROM ORIGINAL:
# - "torchrun --nproc_per_node 8"  →  "python" (single-process launcher for the Legion Go).
RUN_APPLICATION="python"

# Starts the main causal-language-model training program using the launcher defined above.
#
# The arguments below specify:
# - model architecture (GPT-2)
# - tokenizer and dataset locations
# - experiment identity (run name, seed)
# - checkpoint and logging behavior
# - training/evaluation hyperparameters
#
# CHANGED FROM ORIGINAL:
# - Removed "--fp16" because mixed-precision training requires CUDA GPUs.
# - Removed "--ddp_backend nccl" because NCCL is only used for distributed NVIDIA GPU training.
# - Removed "--ddp_timeout 7200" because Distributed Data Parallel (DDP) is no longer being used.
#
# ALL OTHER ARGUMENTS BELOW ARE CURRENTLY IDENTICAL TO THE ORIGINAL train_gpt2.sh.
${RUN_APPLICATION} ./src/learn/run_clm.py \
    --model_type gpt2 \
    --tokenizer_name "${DATA_DIR}/gpt2_tokenizer" \
    --train_file "${DATA_DIR}/raw/train.txt" \
    --validation_file "${DATA_DIR}/raw/validation.txt" \
    --cache_dir "${DATA_DIR}/cache" \
    --run_name ${MODEL_NAME} \
    --seed ${SEED} \
    --report_to wandb \
    --output_dir "./checkpoints/${MODEL_NAME}" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 128 \
    --eval_accumulation_steps 128 \
    --block_size 512 \
    --do_train \
    --logging_steps 50 \
    --do_eval \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --low_cpu_mem_usage \
    ${EXTRA_FLAGS}