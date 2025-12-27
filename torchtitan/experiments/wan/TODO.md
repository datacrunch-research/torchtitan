# Custom TorchTitan Wan 2.2 TI2V-5B model

Run the train w/:
```bash
torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --local-ranks-filter=0 --role=rank --tee=3 --log-dir=./logs/debug -m torchtitan.experiments.wan.train --job.config_file=./torchtitan/experiments/wan/train_configs/wan_1xwm.toml
```


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


- VAE and FSDP:
    - While implementing the VAE I found applying FSDP breaks the cache mechanism so it needs some fixes (WIP)
    - Consider doing a reimplementation of that part (it also fails with `torch.compile`)
- [x] add `wan_dataset` unit test
- [x] get the fwd pass
    - currently is possible to run `torchrun --nproc_per_node=4 torchtitan/experiments/wan/train.py  --job.config_file torchtitan/experiments/wan/train_configs/wan_cc1xm.toml  2>&1 | tee debug.err`
- [ ] update [`README.md`](./README.md)
- [ ] update configs [`train_configs/wan_cc1xm.toml`](./train_configs/wan_cc1xm.toml) and [`train_configs/wan_1xwm.toml`](./train_configs/wan_1xwm.toml)
- [ ] [`train.py`](./train.py:135) has the precomputed embeddings as class attributes, but it should be handled better
- [x] [`wan_vae.py`](./model/wan_vae.py:1509) Refactor `WanVideoVAE.encode()` to accept tensor (B, C, T, H, W) directly instead of list
    - Now accepts `videos: torch.Tensor` with shape (B, C, T, H, W) directly
    - Validates input is a 5D tensor and processes batches efficiently
- [x] [`wan_vae.py`](./model/wan_vae.py:1570) Refactor `WanVideoVAE.decode()` to accept tensor (B, C, T, H, W) directly instead of list
    - Now accepts `hidden_states: torch.Tensor` with shape (B, z_dim, T', H', W') directly
    - Validates input is a 5D tensor and processes batches efficiently
- [ ] [`wan_datasets.py`](./wan_datasets.py:535-537) Fix dataloader configuration for testing
    - Currently has a workaround: when `num_workers == 0` and `prefetch_factor != 1.`, it sets `prefetch_factor = None` to prevent test failures
    - Need to add proper code to pass the correct `prefetch_factor`/`num_workers`/`persistent_workers` values when testing
    - This workaround is needed because the `dataset_wan` test fails without it

--- 
`WanVAE2.2` or `WanVAE38`
The "38" suffix refers to the **Wan 2.2 VAE** variant, which is different from the original Wan VAE:

| Feature | `WanVideoVAE` (original) | `WanVideoVAE38` (Wan 2.2) |
|---------|--------------------------|---------------------------|
| Latent dim (`z_dim`) | 16 | **48** |
| Encoder dim | 96 | **160** |
| Decoder dim | 96 | **256** |
| Uses patchify | No | **Yes** (2x2) |
| Upsampling factor | 8 | **16** |
| Classes used | `VideoVAE_`, `Encoder3d`, `Decoder3d` | `VideoVAE38_`, `Encoder3d_38`, `Decoder3d_38` |

The "38" naming likely comes from an internal Alibaba version number. The key architectural difference is that the **Wan 2.2 VAE** (the 38 version) has:
- Higher latent dimensionality (48 vs 16 channels)
- Uses 2x2 spatial patchification before encoding
- Uses `Resample38` which is designed to match the original Wan 2.2 temporal handling

You're using `WanVideoVAE38` which is correct for the **Wan 2.2 TI2V-5B** model you're working with.

The original code you pasted earlier (from Alibaba's Wan repo) corresponds to the `Wan2_2_VAE` class which uses `WanVAE_` - this is **equivalent** to your `VideoVAE38_` in torchtitan. The naming is just different.
