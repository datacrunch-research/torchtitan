<div align="center">

# Wan model in torchtitan

</div>

## Overview
This directory contains the implementation of Wan2.2 TI2V-5B model in torchtitan. In torchtitan, we showcase the pre-training of the model. The Wan2.2 TI2V-5B model is a transformer-based video generation model that uses flow matching for training.

## Prerequisites
Install the required dependencies:
```bash
pip install -r requirements-wan.txt
```

## Download the Wan2.2 TI2V-5B weights
Download the Wan2.2 TI2V-5B weights from HF:
```bash
python scripts/download_hf_assets.py --repo_id Wan2.2-TI2V-5B --all --hf_token <your_access_token>
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
