# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Iterable, Optional
import time


import torch

from torchtitan.config import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools import utils
from torchtitan.components.dataloader import DataloaderExhaustedError



from torchtitan.experiments.wan.infra.parallelize import parallelize_encoders
from torchtitan.experiments.wan.model.hf_embedder import WanEmbedder
from torchtitan.experiments.wan.utils import (
    create_position_encoding_for_latents,
    pack_latents,
    unpack_latents,
    preprocess_data,
)
from torchtitan.experiments.wan.tokenizer import build_wan_tokenizer

from torchtitan.experiments.wan.model.wan_vae import load_wan_vae

from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer



class WanTrainer(Trainer):
    # Note: Wan model now uses the standard initialization flow in train.py.
    # Weight loading into FSDP-wrapped models is supported via set_model_state_dict(),
    # so weights are loaded after parallelization like all other models.

    def __init__(self, job_config: JobConfig):
        logger.info("=" * 80)
        logger.info("Initializing WanTrainer...")
        logger.info("=" * 80)
        super().__init__(job_config)
        logger.info("Base Trainer initialization completed, continuing with Wan-specific setup...")

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        # For Wan model, we need distinct seed across FSDP ranks to ensure we randomly dropout prompts info in dataloader
        logger.info("Setting determinism for Wan model (distinct seeds across FSDP ranks)...")
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            job_config.debug,
            # distinct_seed_mesh_dims=["dp_shard", "dp_replicate"],
            distinct_seed_mesh_dims=["dp_replicate", "fsdp"],
        )

        # NOTE: self._dtype is the data type used for encoders (image encoder, T5 text encoder).
        # We cast the encoders and it's input/output to this dtype.  If FSDP with mixed precision training is not used,
        # the dtype for encoders is torch.float32 (default dtype for Wan Model).
        # Otherwise, we use the same dtype as mixed precision training process.
        self._dtype = (
            TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
            if self.parallel_dims.dp_shard_enabled
            else torch.float32
        )
        logger.info(f"Encoder dtype set to: {self._dtype}")

        # load components
        logger.info("=" * 80)
        logger.info("STEP 5: Loading Wan model components (VAE, encoders)")
        logger.info("=" * 80)
        model_args = self.train_spec.model_args[job_config.model.flavor]
        logger.info(f"  - Model args: {model_args}")
        
        logger.info(f"  - Loading Wan VAE from: {job_config.encoder.wan_vae_path}")
        logger.info(f"    Device: {self.device}, Dtype: {self._dtype}")
        self.wan_video_vae = load_wan_vae(
            job_config.encoder.wan_vae_path,
            model_args.wan_video_vae_params,
            device=self.device,
            dtype=self._dtype,
            random_init=job_config.training.test_mode,
        )
        logger.info("  ✓ Wan VAE loaded successfully")
        # TODO: add here the loading of the WanVideoVAE38
        
        logger.info(f"  - Loading T5 encoder: {job_config.encoder.t5_encoder}")
        logger.info(f"    Device: {self.device}, Dtype: {self._dtype}")
        self.t5_encoder = WanEmbedder(
            version=job_config.encoder.t5_encoder,
            random_init=job_config.training.test_mode,
        ).to(device=self.device, dtype=self._dtype)
        logger.info("  ✓ T5 encoder loaded successfully")

        # Apply FSDP to the T5 model / VAE
        logger.info("=" * 80)
        logger.info("STEP 6: Applying FSDP parallelization to encoders (T5, VAE)")
        logger.info("=" * 80)
        self.t5_encoder, self.wan_video_vae = parallelize_encoders(
            t5_model=self.t5_encoder,
            wan_video_vae=self.wan_video_vae,
            parallel_dims=self.parallel_dims,
            job_config=job_config,
        )
        logger.info("  ✓ FSDP parallelization applied to encoders")

        # TODO: this part should be handled better
        # Precompute empty string embeddings once to save memory and time
        # This allows T5 encoder to be offloaded since we only encode ""
        logger.info("=" * 80)
        logger.info("STEP 7: Precomputing empty string embeddings for T5")
        logger.info("=" * 80)
        logger.info("  - This allows encoder to be offloaded since we only encode empty strings (\"\")")
        logger.info("  - T5 uses last_hidden_state for per-token embeddings")
        logger.info("  - Precomputed embeddings will be reused during training to avoid encoder forward passes")
        with torch.no_grad():
            # Get empty string tokens
            logger.info("  - Building tokenizer...")
            t5_tokenizer = build_wan_tokenizer(job_config)
            logger.info("  - Encoding empty string...")
            empty_t5_tokens_tensor = t5_tokenizer.encode("").to(device=self.device)
            logger.info(f"  - T5 tokens shape: {empty_t5_tokens_tensor.shape}")
            
            logger.info("  - Computing embeddings using encoder...")
            # Compute embeddings using the encoder (after FSDP wrapping)
            self._precomputed_t5_embedding = self.t5_encoder(empty_t5_tokens_tensor).to(dtype=self._dtype)
            
            logger.info(f"  - T5 embedding shape (before squeeze): {self._precomputed_t5_embedding.shape}")
            
            # Remove batch dimension for single sample
            logger.info("  - Removing batch dimension...")
            self._precomputed_t5_embedding = self._precomputed_t5_embedding.squeeze(0)  # [seq_len, hidden_dim]
            
            logger.info(f"  - T5 embedding shape (after squeeze): {self._precomputed_t5_embedding.shape}")
            
        logger.info("  ✓ Empty string embeddings precomputed successfully")
        
        # Delete T5 encoder after precomputation to free memory
        # It's no longer needed since we use precomputed embeddings
        # if not job_config.validation.enable:
        logger.info("  - Deleting T5 encoder (no longer needed, using precomputed embeddings)...")
        del self.t5_encoder
        self.t5_encoder = None
        # Also delete tokenizer as it's no longer needed
        del t5_tokenizer
        
        logger.info("  ✓ Encoder and tokenizer deleted to free memory")
        # else:
            # logger.info("  - Keeping encoder (validation is enabled and may need it)")

        if job_config.validation.enable:
            logger.info("Initializing Wan validator...")
            logger.info(f"t5_encoder is None {self.t5_encoder is None}")
            self.validator.wan_init(
                device=self.device,
                _dtype=self._dtype,
                wan_video_vae=self.wan_video_vae,
                t5_encoder=self.t5_encoder,
                precomputed_t5_embedding=self._precomputed_t5_embedding
            )
            logger.info("Wan validator initialized")
            # TODO: also here add the validation code for wan
        
        logger.info("=" * 80)
        logger.info("WanTrainer initialization completed")
        logger.info("=" * 80)

    
    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator."""
        device_type = utils.device_type
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            # logger.info(f"labels.shape: {labels.shape}")
            ntokens_batch = 1 + (labels.shape[1] - 1) // 4 *  labels.shape[2] // 16 * labels.shape[3] // 16
            
            # logger.info(f"ntokens_batch: {ntokens_batch}")
            # logger.info(f"self.ntokens_seen: {self.ntokens_seen}")
            self.ntokens_seen += ntokens_batch
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Move tensors to the appropriate device
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(device_type)
            labels = labels.to(device_type)

            yield input_dict, labels

    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        # Preprocess data: generate t5 embeddings, encode video with VAE
        # Encoder may be None if it was deleted after precomputation
        input_dict = preprocess_data(
            device=self.device,
            dtype=self._dtype,
            wan_video_vae=self.wan_video_vae,
            t5_encoder=getattr(self, 't5_encoder', None),
            batch=input_dict,
            precomputed_t5_embedding=self._precomputed_t5_embedding,
        )

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        # explicitly convert flux model to be Bfloat16 no matter FSDP is applied or not
        model = self.model_parts[0]
        # TODO: same for Wan's DiT?

        t5_encodings = input_dict["t5_encodings"]
        labels = input_dict["latents"]


        bsz = labels.shape[0]

        # Get number of conditioning frames to keep clean (no noise)
        # The model expects first num_latent_cond frames to be clean for conditioning
        num_latent_cond = model.num_latent_cond if hasattr(model, 'num_latent_cond') else 2

        # Generate noise and timesteps for diffusion training
        with torch.no_grad(), torch.device(self.device):
            noise = torch.randn_like(labels, dtype=torch.bfloat16)
            timesteps = torch.rand((bsz,), dtype=torch.bfloat16)
            sigmas = timesteps.view(-1, 1, 1, 1, 1)
            # Mix clean latents with noise based on timesteps
            # Shape: (B, C, T, H, W) - we mask along temporal dimension (dim=2)
            latents = (1 - sigmas) * labels + sigmas * noise
            
            # Masking: Keep first num_latent_cond frames clean (no noise) for conditioning
            # These frames serve as conditioning context for the model
            if num_latent_cond > 0 and labels.shape[2] > num_latent_cond:
                # Keep first num_latent_cond frames as clean latents (no noise)
                latents[:, :, :num_latent_cond, :, :] = labels[:, :, :num_latent_cond, :, :]
        # logger.info(f"latents shape: {latents.shape}")
        # logger.info(f"latents device: {latents.device}")
        # assert latents.dtype == torch.bfloat16, "Latents must be bfloat16"
        # logger.info(f"latents dtype: {latents.dtype}")

        bsz, _, _, latent_height, latent_width = latents.shape

        # Prepare positional encodings and patchify latents for model input
        POSITION_DIM = 3  # constant for Flux flow model
        with torch.no_grad(), torch.device(self.device):
            # Create positional encodings for text
            text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM)

            # Patchify: Convert latent into a sequence of patches
            latents_p = pack_latents(latents)
            
            # Compute target: noise - labels for frames that need prediction
            # For conditioning frames (first num_latent_cond), target should be zero
            # since we're not predicting noise for clean conditioning frames
            target_noise_diff = noise - labels
            if num_latent_cond > 0 and labels.shape[2] > num_latent_cond:
                # Set target to zero for conditioning frames (no noise prediction needed)
                target_noise_diff[:, :, :num_latent_cond, :, :] = 0.0
            target = pack_latents(target_noise_diff)

        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=self.parallel_dims.world_mesh["cp"],
                cp_buffers=[
                    latents_p,
                    # latent_pos_enc,
                    t5_encodings,
                    text_pos_enc,
                    target,
                ],
                cp_seq_dims=[1, 1, 1, 1, 1],
                cp_no_restore_buffers={
                    latents_p,
                    # latent_pos_enc,
                    t5_encodings,
                    text_pos_enc,
                    target,
                },
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if self.parallel_dims.cp_enabled
            else None
        )
        # Forward pass through the model
        with self.train_context(optional_context_parallel_ctx):
            with self.maybe_enable_amp:
                # Model forward: predict noise in latents
                latent_noise_pred = model(
                    x=latents,
                    timesteps=timesteps,
                    context=t5_encodings,
                    robot_states=input_dict["robot_states"],
                )  

                # Pack the model output to match the target format
                # Model outputs (B, C, T, H, W), target is (B, T*H/2*W/2, C*4)
                latent_noise_pred = pack_latents(latent_noise_pred)

                # Compute loss between predicted noise and target
                loss = self.loss_fn(latent_noise_pred, target)

            # Free intermediate tensors before backward to avoid memory peak
            del (latent_noise_pred, noise, target)
            # Backward pass: compute gradients
            loss.backward()

        return loss


if __name__ == "__main__":
    init_logger()
    logger.info("=" * 80)
    logger.info("Starting Wan training script")
    logger.info("=" * 80)
    
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    logger.info(f"Configuration loaded: {config.job.config_file if hasattr(config.job, 'config_file') else 'N/A'}")
    trainer: Optional[WanTrainer] = None

    try:
        logger.info("Creating WanTrainer instance...")
        trainer = WanTrainer(config)
        logger.info("WanTrainer created successfully")
        
        if config.checkpoint.create_seed_checkpoint:
            logger.info("Creating seed checkpoint...")
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Seed checkpoint created successfully")
        else:
            logger.info("Starting training loop...")
            trainer.train()
            logger.info("Training loop completed")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        if trainer:
            trainer.close()
        raise
    else:
        logger.info("Cleaning up resources...")
        trainer.close()
        torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")
        logger.info("=" * 80)
        logger.info("Wan training execution completed successfully")
        logger.info("=" * 80)
