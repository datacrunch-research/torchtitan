# Custom TorchTitan Wan 2.2 TI2V-5B model

Run the debug script for validation: 
```bash
torchrun --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --local-ranks-filter=0 --role=rank --tee=3 --log-dir=./logs/debug -m torchtitan.experiments.wan.validate --job.config_file=./torchtitan/experiments/wan/train_configs/validate.toml
```

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
    - torchao?
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

---

## Comprehensive TODO List for `add_validation` Branch

### 1. Validation Implementation (Priority: High)

- [ ] Complete the validation loop in [`validate.py:199`](./validate.py) - remove `NotImplementedError`
- [ ] Add pipeline parallelism support for validation (currently not supported, see `validate.py:115`)
- [ ] Implement validation loss computation for Wan model (sampling code for Wan)
- [ ] Wire up `metrics_processor.log_validation()` properly
- [ ] Test validation with different batch sizes and step counts

### 2. Classifier-Free Guidance (Priority: Medium)

- [ ] Implement CFG support in `generate_video()` function ([`sampling.py:96`](./inference/sampling.py))
- [ ] Add empty T5 encoding handling for unconditional generation ([`sampling.py:112`](./inference/sampling.py))
- [ ] Test CFG with different guidance scales (config: `classifier_free_guidance_scale`)

### 3. VAE + FSDP Compatibility (Priority: High)

- [ ] Fix cache mechanism that breaks with FSDP wrapping ([`parallelize.py:220`](./infra/parallelize.py))
    - FSDP wrapping of `wan_video_vae` breaks the internal caching logic
- [ ] Consider reimplementing the caching layer to be FSDP-compatible
- [ ] Explore `torchao` as an alternative optimization approach
- [ ] Test VAE encoding/decoding with and without FSDP

### 4. torch.compile Support (Priority: Medium)

- [ ] Fix VAE compatibility with `torch.compile` (currently breaks)
- [ ] Test model compilation with `inductor` backend
- [ ] Profile performance improvements with compilation enabled
- [ ] Update `[compile]` config section documentation

### 5. Code Cleanup

- [ ] Handle precomputed embeddings better in [`train.py:131`](./train.py)
    - Currently stored as class attributes, should be handled more cleanly
- [ ] Remove deprecated `save_image` function from [`sampling.py:234`](./inference/sampling.py)
- [ ] Fix tokenizer `_n_words` hardcoded value in [`tokenizer.py:93`](./tokenizer.py) - needs verification
- [ ] Add validation code in [`train.py:196`](./train.py) for Wan model
- [ ] Clean up DiT-related TODO in [`train.py:263`](./train.py)

### 6. Data Pipeline

- [ ] Fix dataloader `prefetch_factor`/`num_workers`/`persistent_workers` workaround ([`wan_datasets.py:503-504`](./wan_datasets.py))
    - Current workaround sets `prefetch_factor = None` when `num_workers == 0`
    - Need proper handling for test vs production configurations
- [ ] Verify zero-copy behavior for video frames ([`wan_datasets.py:447`](./wan_datasets.py))

### 7. Documentation

- [ ] Update [`README.md`](./README.md) with validation usage and examples
- [ ] Update config file documentation for `wan_cc1xm.toml` and `wan_1xwm.toml`
- [ ] Add FLOPS calculation for Wan model ([`args.py:66`](./model/args.py))
- [ ] Document the differences between `WanVideoVAE` and `WanVideoVAE38`

### 8. Additional Parallelism

- [ ] Add Tensor Parallelism support
- [ ] Add Pipeline Parallelism support
- [ ] Implement `num_flops_per_token` calculation in `get_nparams_and_flops()` function

### 9. Inference Pipeline

- [ ] Verify image resolution handling in [`sampling.py:91`](./inference/sampling.py)
- [ ] Verify latent unpacking logic in [`sampling.py:226`](./inference/sampling.py)
- [ ] Add more sampling schedulers (beyond flow matching)
