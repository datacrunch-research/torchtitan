# Docker image for torchtitan (Llama 3 70B)

This image trains Llama 3 70B (no fp8) on a cluster. It is built on the NGC
PyTorch base image with a pinned nightly PyTorch stack, ships the Llama-3.1-70B
tokenizer baked in (cluster nodes have no shared storage), and launches training
via `train_llama70b.sh`.

## Building

The tokenizer download requires a Hugging Face token with access to
`meta-llama/Llama-3.1-70B`, passed as a build secret:

```bash
docker build -t torchtitan:llama3-70b \
  --secret id=hf_token,src=$HOME/.cache/huggingface/token .
```

Tag and push to the registry:

```bash
docker tag torchtitan:llama3-70b registry.datacrunch.io/verda-aa/torchtitan:llama3-70b
docker push registry.datacrunch.io/verda-aa/torchtitan:llama3-70b
```

## Running

Llama 3 70B uses 8-way tensor parallelism, so a node with 8 GPUs is required.

Single node, 8 GPUs:

```bash
docker run --gpus all -it --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  torchtitan:llama3-70b
```

Multi-node runs are driven by the `PET_*` environment variables set by the
Kubeflow Training Operator (`PET_NNODES`, `PET_NODE_RANK`, `PET_MASTER_ADDR`,
`PET_MASTER_PORT`); `train_llama70b.sh` picks them up automatically.

Extra arguments are forwarded to the trainer, e.g. to shorten a smoke test:

```bash
docker run --gpus all -it --rm torchtitan:llama3-70b --training.steps 10
```

## What runs

`train_llama70b.sh` invokes:

```bash
torchrun ... -m torchtitan.train --module llama3 --config llama3_70b
```

The `llama3_70b` config is registered in
`torchtitan/models/llama3/config_registry.py` (8-way TP, full activation
checkpointing, bf16 - no fp8).
