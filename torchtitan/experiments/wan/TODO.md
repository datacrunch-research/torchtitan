# Custom TorchTitan Wan 2.2 TI2V-5B model

Run the debug script for validation:
```bash
torchrun --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --local-ranks-filter=0 --role=rank --tee=3  -m torchtitan.experiments.wan.validate --job.config_file=./torchtitan/experiments/wan/train_configs/validate.toml
```

Run the train w/:
```bash
torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --local-ranks-filter=0 --role=rank --tee=3 -m torchtitan.experiments.wan.train --job.config_file=./torchtitan/experiments/wan/train_configs/wan_1xwm.toml
```

Run training with pre-encoded latents:
```bash
# First, encode latents (one-time):
# python -m torchtitan.experiments.wan.encode_latents \
#     --dataset_path ./dataset/world_model_raw_data/train_v2.0_raw \
#     --output_dir ./dataset/world_model_raw_data/train_v2.0_latents \
#     --vae_path assets/hf/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
#     --num_samples 100000
torchrun --nproc_per_node=8 -m torchtitan.experiments.wan.encode_latents \
      --dataset_path ./dataset/world_model_raw_data/train_v2.0_raw \
      --output_dir ./dataset/world_model_raw_data/train_v2.0_latents \
      --vae_path assets/hf/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
      --downsampled 4 \
      --clip_length 77 \
      --batch_size 64 \
      --compile

# Then train with pre-encoded latents:
torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --local-ranks-filter=0 --role=rank --tee=3 --log-dir="./logs/$(date +%Y%m%d_%H%M%S)"  -m torchtitan.experiments.wan.train
 --job.config_file=./torchtitan/experiments/wan/train_configs/wan_1xwm_latents.toml
```

---
## Comprehensive TODO List for `add_validation` Branch

1. Select dataset -> In our case this is the 1x World Model dataset (+ other NVIDIA stuff)
check what they did for FLUX
    1. So far we are relying on `decord` but there is also this [PyNvVideoCodec](https://developer.nvidia.com/pynvvideocodec)

### 1. Validation Implementation (Priority: High)

- [x] Complete the validation loop in [`validate.py:199`](./validate.py) - remove `NotImplementedError`
- [ ] Add pipeline parallelism support for validation (currently not supported, see `validate.py:115`)
- [x] Implement validation loss computation for Wan model (sampling code for Wan)
- [x] Wire up `metrics_processor.log_validation()` properly
- [x] Test validation with different batch sizes and step counts
- [ ] Update the loss to be compatible with the latest loss (they put mse default to be sum, I want to go to how it's done for FLUX now)

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

- [x] Fix VAE compatibility with `torch.compile` (currently breaks)
- [x] Test model compilation with `inductor` backend
- [ ] Profile performance improvements with compilation enabled
- [ ] Update `[compile]` config section documentation

### 5. Code Cleanup

- [x] Handle precomputed embeddings better in [`train.py:131`](./train.py)
    - Currently stored as class attributes, should be handled more cleanly
- [ ] Remove deprecated `save_image` function from [`sampling.py:234`](./inference/sampling.py)
- [x] Fix tokenizer `_n_words` hardcoded value in [`tokenizer.py:93`](./tokenizer.py) - needs verification
- [x] Add validation code in [`train.py:196`](./train.py) for Wan model
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
