# Custom TorchTitan Wan 2.2 TI2V-5B model

1. Select dataset -> In our case this is the 1x World Model dataset (+ other NVIDIA stuff)
check what they did for FLUX
    1. So far we are relying on `decord` but there is also this [PyNvVideoCodec](https://developer.nvidia.com/pynvvideocodec)
2. Model and Training component definition -> add both TrainSpec and config files
3. Training Config and loops -> requires the custom version of the Trainer class
4. Parallelization Function -> FSDP2 and considerations on how to handle the other components of the pipeline
5. Training, Performance Analysis, and Debugging

What TorchTitan people had for FLUX.1:
- Parallelism: The model supports FSDP, HSDP for training on multiple GPUs. The FLUX.1 models are using TorchTitan compatible architecture, ready to integrate more parallelism.
- Activation checkpointing: The model uses activation checkpointing to reduce memory usage during training.
- Checkpointing: save and load DCP format checkpoint.
- Classifier-free diffusion guidance support: We support classifier-free guidance for FLUX.1 models.


- [x] add `wan_dataset` unit test
- [ ] get the fwd pass
    - currently is possible to run `torchrun --nproc_per_node=4 torchtitan/experiments/wan/train.py  --job.config_file torchtitan/experiments/wan/train_configs/wan_cc1xm.toml  2>&1 | tee debug.err`
- [ ] update [`README.md`](./README.md)
- [ ] update configs [`train_configs/wan_cc1xm.toml`](./train_configs/wan_cc1xm.toml) and [`train_configs/wan_1xwm.toml`](./train_configs/wan_1xwm.toml)
- [ ] [`train.py`](./train.py:135) has the precomputed embeddings as class attributes, but it should be handled better
- [ ] [`wan_vae.py`](./model/wan_vae.py:1495) Refactor `WanVideoVAE.encode()` to accept tensor (B, C, T, H, W) directly instead of list
    - Currently accepts both list of (C, T, H, W) tensors or single (B, C, T, H, W) tensor
    - Should standardize on tensor input for better performance and cleaner API
- [ ] [`wan_vae.py`](./model/wan_vae.py:1527) Refactor `WanVideoVAE.decode()` to accept tensor (B, C, T, H, W) directly instead of list
    - Currently accepts both list of (C, T, H, W) tensors or single (B, C, T, H, W) tensor
    - Should standardize on tensor input for better performance and cleaner API

