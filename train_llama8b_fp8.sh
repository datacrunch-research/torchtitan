#!/bin/bash

# Training script with Float8 and compile enabled
# Works for both single-node (docker) and multi-node (Kubernetes)
#
# Single-node: uses defaults (NUM_NODES=1, MASTER_ADDR=localhost)
# Multi-node:  uses PET_* env vars from Kubeflow Training Operator

export NUM_NODES=${PET_NNODES:-1}
export NODE_RANK=${PET_NODE_RANK:-0}
export MASTER_ADDR=${PET_MASTER_ADDR:-localhost}
export MASTER_PORT=${PET_MASTER_PORT:-29500}

# NCCL profiler settings
export NCCL_PROFILER_PLUGIN=/workspace/nccl/ext-profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
export NCCL_INSPECTOR_PROM_DUMP=1
export NCCL_INSPECTOR_DUMP_DIR=/workspace/prom_dump/

torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=8 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  torchtitan/train.py \
  --job.config-file torchtitan/models/llama3/train_configs/llama3_8b_float8.toml
