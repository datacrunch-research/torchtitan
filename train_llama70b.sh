#!/bin/bash

# Llama 3 70B training launcher (no fp8).
# Works for both single-node (docker run) and multi-node (Kubernetes / torchrun).
#
# Single-node: uses defaults (NUM_NODES=1, MASTER_ADDR=localhost)
# Multi-node:  uses PET_* env vars from the Kubeflow Training Operator

export NUM_NODES=${PET_NNODES:-1}
export NODE_RANK=${PET_NODE_RANK:-0}
export MASTER_ADDR=${PET_MASTER_ADDR:-localhost}
export MASTER_PORT=${PET_MASTER_PORT:-29500}

torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=8 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  -m torchtitan.train \
  --module llama3 \
  --config llama3_70b \
  "$@"
