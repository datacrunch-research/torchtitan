<div align="center">

# Wan model in torchtitan

</div>

## Overview
This directory contains the implementation of Wan2.2 TI2V-5B model in torchtitan. In torchtitan, we showcase the pre-training of the model. The Wan2.2 TI2V-5B model is a transformer-based video generation model that uses flow matching for training.

## Prerequisites
Create a `uv` venv by running:
```bash
uv venv --python 3.12
uv sync
```
This will install all dependencies including PyTorch nightly (pinned to tested versions in `pyproject.toml`).

<details>
<summary>Legacy setup</summary>

The old manual setup is no longer needed since all dependencies are now managed in `pyproject.toml`:
```bash
cd path/to/torchtitan
uv venv --python 3.12
uv pip install --pre torch torchvision torchdata
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
uv pip install -r torchtitan/experiments/wan/requirements-wan.txt
```
</details>

## Download the Wan2.2 TI2V-5B weights
Download the Wan2.2 TI2V-5B weights from HF:
```bash
python scripts/download_hf_assets.py --repo_id Wan-AI/Wan2.2-TI2V-5B  --all --hf_token <HF_TOKENn>
python scripts/download_hf_assets.py --repo_id Wan-AI/Wan2.2-I2V-A14B --all --hf_token <HF_TOKEN>
# or
# hf auth login
# python scripts/download_hf_assets.py --repo_id Wan-AI/Wan2.2-TI2V-5B --all
```

Download the 1X World Model dataset with:
```bash
# Remember to login to hf first
hf auth login --token $HF_TOKEN
hf download 1x-technologies/worldmodel_raw_data --repo-type dataset --local-dir $DATA_DIR/world_model_raw_data --token $HF_TOKEN
```

## Usage
Run the following command to train the model:
```bash
./torchtitan/experiments/wan/run_train.sh
```

If you want to train with other model args, run the following command:
```bash
CONFIG_FILE="./torchtitan/experiments/wan/train_configs/wan_1xwm.toml" ./torchtitan/experiments/wan/run_train.sh
```

Or run torchrun directly with timestamped logging:
```bash
torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter=0 --role=rank --tee=3 \
    --log-dir="./logs/$(date +%Y%m%d_%H%M%S)" \
    -m torchtitan.experiments.wan.train \
    --job.config_file=./torchtitan/experiments/wan/train_configs/wan_1xwm_latents.toml
```


## Supported Features
- Parallelism: The model supports FSDP, HSDP, CP for training on multiple GPUs.
- Activation checkpointing: The model uses activation checkpointing to reduce memory usage during training.
- Distributed checkpointing and loading.
    - Notes on the current checkpointing implementation: To keep the model weights are sharded the same way as checkpointing, we need to shard the model weights before saving the checkpoint. This is done by checking each module at the end of evaluation, and sharding the weights of the module if it is a FSDPModule.
- Video generation: The model supports text-to-video generation with flow matching.
- Multi-modal encoding: Supports T5 and CLIP encoders for text conditioning.


## TODO
- [ ] More parallelism support (Tensor Parallelism, Pipeline Parallelism, etc)
- [ ] Implement the num_flops_per_token calculation in get_nparams_and_flops() function
- [ ] Add `torch.compile` support
