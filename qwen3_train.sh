#!/bin/bash

export NUM_NODES=1
export MASTER_ADDR=localhost

# NCCL debug logging - real-time output to file
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=/home/riccardo/torchtitan/logs/nccl_debug_%h_%p.log

# Flight recorder - dumps on timeout/error
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/home/riccardo/torchtitan/logs/nccl_trace_
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

torchrun \
  --nnodes $NUM_NODES \
  --nproc_per_node 8 \
  --rdzv_id 101 \
  --rdzv_backend c10d \
  --rdzv_endpoint "$MASTER_ADDR:29500" \
  --local-ranks-filter=0 --role=rank --tee=3 \
  -m torchtitan.train --job.config-file torchtitan/models/qwen3/train_configs/qwen3_32b.toml
