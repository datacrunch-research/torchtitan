#!/bin/bash

export NUM_NODES=1
export MASTER_ADDR=localhost

# export TORCH_FR_DUMP_TEMP_FILE="/tmp/trace_qwen3_"
# export TORCH_NCCL_DUMP_ON_TIMEOUT=1
# export TORCH_FR_BUFFER_SIZE=2000
export TORCH_NCCL_DEBUG_INFO_PIPE_FILE=/tmp/nccl_trace_pipe

torchrun \
  --nnodes $NUM_NODES \
  --nproc_per_node 8 \
  --rdzv_id 101 \
  --rdzv_backend c10d \
  --rdzv_endpoint "$MASTER_ADDR:29500" \
  --local-ranks-filter=0 --role=rank --tee=3 \
  -m torchtitan.train --job.config-file torchtitan/models/qwen3/train_configs/qwen3_32b.toml
