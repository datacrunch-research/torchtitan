#!/bin/bash

# Training script with Float8 and compile enabled
# Matches the H200 benchmark configuration

export NUM_NODES=1
export MASTER_ADDR=localhost

torchrun \
  --nnodes $NUM_NODES \
  --nproc_per_node 8 \
  --rdzv_id 101 \
  --rdzv_backend c10d \
  --rdzv_endpoint "$MASTER_ADDR:29500" \
  torchtitan/train.py \
  --job.config-file torchtitan/models/llama3/train_configs/llama3_8b_float8.toml
