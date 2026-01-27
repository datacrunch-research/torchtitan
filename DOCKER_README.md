# Docker for torchtitan

## Building the Docker Image

```bash
docker build -t torchtitan:latest .
```

## Running Training

Single-node, 8 GPUs:
```bash
docker run --gpus all -it --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/workspace/torchtitan \
  -e MASTER_ADDR=localhost \
  -e MASTER_PORT=29500 \
  torchtitan:latest
```

```bash
wandb login
# or wandb init
hf auth login
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer
```

```bash
# to avoid all the debug logs
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF
./train_fp8.sh
```
