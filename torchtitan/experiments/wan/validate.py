# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Generator, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torchmetrics.functional.image import peak_signal_noise_ratio


from torchtitan.tools.logging import init_logger
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import Validator
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.wan.wan_datasets import build_wan_validation_dataloader
from torchtitan.experiments.wan.inference.sampling import generate_video, save_video
from torchtitan.experiments.wan.model.hf_embedder import WanEmbedder

from torchtitan.experiments.wan.model.wan_vae import WanVideoVAE
from torchtitan.experiments.wan.tokenizer import build_wan_tokenizer
from torchtitan.experiments.wan.utils import (
    create_position_encoding_for_latents,
    pack_latents,
    preprocess_data,
)
from torchtitan.tools.logging import logger


class WanValidator(Validator):
    """
    Simple validator focused on correctness and integration.

    Args:
        job_config: Job configuration
        validation_dataloader: The validation dataloader
        loss_fn: Loss function to use for validation
        model: The model to validate (single model, no parallelism)
    """

    validation_dataloader: BaseDataLoader

    def __init__(
        self,
        job_config: JobConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: LossFunction,
        validation_context: Generator[None, None, None],
        maybe_enable_amp: Generator[None, None, None],
        metrics_processor: MetricsProcessor | None = None,
        pp_schedule: _PipelineSchedule | None = None,
        pp_has_first_stage: bool | None = None,
        pp_has_last_stage: bool | None = None,
    ):
        self.job_config = job_config
        self.parallel_dims = parallel_dims
        self.loss_fn = loss_fn
        self.all_timesteps = self.job_config.validation.all_timesteps
        self.validation_dataloader = build_wan_validation_dataloader(
            job_config=job_config,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            generate_timestamps=not self.all_timesteps,
            infinite=self.job_config.validation.steps != -1,
        )
        self.validation_context = validation_context
        self.maybe_enable_amp = maybe_enable_amp
        self.metrics_processor = metrics_processor
        
        self.t5_tokenizer = build_wan_tokenizer(self.job_config)
        self.precomputed_t5_embedding = None
        # with torch.no_grad():
        #     empty_t5_tokens_tensor = self.t5_tokenizer.encode("").to(device=self.device)
        #     self._precomputed_t5_embedding = self.t5_encoder(empty_t5_tokens_tensor).to(dtype=self._dtype)
        #     self._precomputed_t5_embedding = self._precomputed_t5_embedding.squeeze(0)  # [seq_len, hidden_dim]


        if self.job_config.validation.steps == -1:
            logger.warning(
                "Setting validation steps to -1 might cause hangs because of "
                "unequal sample counts across ranks when dataset is exhausted."
            )

    def wan_init(
        self,
        device: torch.device,
        _dtype: torch.dtype,
        wan_video_vae: WanVideoVAE,
        t5_encoder: Optional[WanEmbedder] = None,
        precomputed_t5_embedding: Optional[Tensor] = None,
    ):
        self.device = device
        self._dtype = _dtype
        self.wan_video_vae = wan_video_vae
        self.t5_encoder = t5_encoder
        self.precomputed_t5_embedding = precomputed_t5_embedding

    @torch.no_grad()
    def validate(
        self,
        model_parts: list[nn.Module],
        step: int,
    ) -> None:
        # Set model to eval mode
        # TODO: currently does not support pipeline parallelism
        model = model_parts[0]
        model.eval()

        # Disable cfg dropout during validation
        training_cfg_prob = self.job_config.training.classifier_free_guidance_prob
        self.job_config.training.classifier_free_guidance_prob = 0.0

        save_img_count = self.job_config.validation.save_img_count

        parallel_dims = self.parallel_dims

        accumulated_losses = []
        device_type = dist_utils.device_type
        num_steps = 0

        for input_dict, labels in self.validation_dataloader:
            if (
                self.job_config.validation.steps != -1
                and num_steps >= self.job_config.validation.steps
            ):
                break

            prompt = input_dict.pop("prompt")
            if not isinstance(prompt, list):
                prompt = [prompt]
            for p in prompt:
                if save_img_count != -1 and save_img_count <= 0:
                    break
                
                input_dict["num_cond_frames"] = self.job_config.validation.num_cond_frames
                logger.info(input_dict.keys())
                video = generate_video(
                    device=self.device,
                    dtype=self._dtype,
                    job_config=self.job_config,
                    model=model,
                    input_dict=input_dict,
                    wan_video_vae=self.wan_video_vae,
                    t5_tokenizer=self.t5_tokenizer,
                    t5_encoder=self.t5_encoder,
                    precomputed_t5_embedding=self.precomputed_t5_embedding,
                )
                logger.info(f"Video shape: {video.shape}")

                save_video(
                    name=f"video_rank{str(torch.distributed.get_rank())}_{step}.mp4",
                    output_dir=os.path.join(
                        self.job_config.job.dump_folder,
                        self.job_config.validation.save_img_folder,
                    ),
                    video=video,
                    add_sampling_metadata=True,
                )
                # save_image(
                #     name=f"image_rank{str(torch.distributed.get_rank())}_{step}.png",
                #     output_dir=os.path.join(
                #         self.job_config.job.dump_folder,
                #         self.job_config.validation.save_img_folder,
                #     ),
                #     x=image,
                #     add_sampling_metadata=True,
                #     prompt=p,
                # )
                # save_img_count -= 1

            # generate t5 embeddings
            input_dict["image"] = labels
            input_dict = preprocess_data(
                device=self.device,
                dtype=self._dtype,
                wan_video_vae=self.wan_video_vae,
                t5_encoder=self.t5_encoder,
                batch=input_dict,
                precomputed_t5_embedding=self.precomputed_t5_embedding,
            )
            labels = input_dict["latents"]
            # labels = input_dict["img_encodings"].to(device_type)
            t5_encodings = input_dict["t5_encodings"]
            logger.info(f"T5 encodings shape: {t5_encodings.shape}")
            logger.info(f"Labels shape: {labels.shape}")

            bsz = labels.shape[0]
            # TODO: Add here the sampling code for Wan model
            # If using all_timesteps we generate all 8 timesteps and expand our batch inputs here
            if self.all_timesteps:
                stratified_timesteps = torch.tensor(
                    [1 / 8 * (i + 0.5) for i in range(8)],
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(bsz)
                t5_encodings = t5_encodings.repeat_interleave(8, dim=0)
                labels = labels.repeat_interleave(8, dim=0)
            else:
                stratified_timesteps = input_dict.pop("timestep")

            # Note the tps may be inaccurate due to the generating image step not being counted
            self.metrics_processor.ntokens_since_last_log += labels.numel()

            # Apply timesteps here and update our bsz to efficiently compute all timesteps and samples in a single forward pass
            with torch.no_grad(), torch.device(self.device):
                noise = torch.randn_like(labels)
                timesteps = stratified_timesteps.to(labels)
                sigmas = timesteps.view(-1, 1, 1, 1)
                latents = (1 - sigmas) * labels + sigmas * noise

            bsz, _, latent_height, latent_width = latents.shape

            POSITION_DIM = 3  # constant for Wan flow model
            with torch.no_grad(), torch.device(self.device):
                # Create positional encodings
                latent_pos_enc = create_position_encoding_for_latents(
                    bsz, latent_height, latent_width, POSITION_DIM
                )
                text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM)

                # Patchify: Convert latent into a sequence of patches
                latents = pack_latents(latents)
                target = pack_latents(noise - labels)

                optional_context_parallel_ctx = (
                    dist_utils.create_context_parallel_ctx(
                        cp_mesh=parallel_dims.world_mesh["cp"],
                        cp_buffers=[
                            latents,
                            latent_pos_enc,
                            t5_encodings,
                            text_pos_enc,
                            target,
                        ],
                        cp_seq_dims=[1, 1, 1, 1, 1],
                        cp_no_restore_buffers={
                            latents,
                            latent_pos_enc,
                            t5_encodings,
                            text_pos_enc,
                            target,
                        },
                        cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
                    )
                    if parallel_dims.cp_enabled
                    else None
                )

                with self.validation_context(optional_context_parallel_ctx):
                    with self.maybe_enable_amp:
                        latent_noise_pred = model(
                            img=latents,
                            img_ids=latent_pos_enc,
                            txt=t5_encodings,
                            txt_ids=text_pos_enc,
                            timesteps=timesteps,
                        )

                    loss = self.loss_fn(latent_noise_pred, target)

            del noise, target, latent_noise_pred, latents

            accumulated_losses.append(loss.detach())

            num_steps += 1

        # Compute average loss
        loss = torch.sum(torch.stack(accumulated_losses))
        loss /= num_steps
        if parallel_dims.dp_cp_enabled:
            global_avg_loss = dist_utils.dist_mean(
                loss, parallel_dims.world_mesh["dp_cp"]
            )
        else:
            global_avg_loss = loss.item()

        self.metrics_processor.log_validation(loss=global_avg_loss, step=step)

        # Set model back to train mode
        model.train()

        # re-enable cfg dropout for training
        self.job_config.training.classifier_free_guidance_prob = training_cfg_prob


def build_wan_validator(
    job_config: JobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    parallel_dims: ParallelDims,
    loss_fn: LossFunction,
    validation_context: Generator[None, None, None],
    maybe_enable_amp: Generator[None, None, None],
    metrics_processor: MetricsProcessor | None = None,
    pp_schedule: _PipelineSchedule | None = None,
    pp_has_first_stage: bool | None = None,
    pp_has_last_stage: bool | None = None,
) -> WanValidator:
    """Build a simple validator focused on correctness."""
    return WanValidator(
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=parallel_dims,
        loss_fn=loss_fn,
        validation_context=validation_context,
        maybe_enable_amp=maybe_enable_amp,
        metrics_processor=metrics_processor,
        pp_schedule=pp_schedule,
        pp_has_first_stage=pp_has_first_stage,
        pp_has_last_stage=pp_has_last_stage,
    )


if __name__ == "__main__":
    """
    Test script for the WanValidator.
    
    This script tests the validation dataloader and basic setup without requiring
    distributed training. It initializes components with random weights for testing.
    
    Usage:
        python -m torchtitan.experiments.wan.validate
    """
    from torchtitan.config.manager import ConfigManager
    from torchtitan.experiments.wan.model.wan_vae import load_wan_vae
    from torchtitan.experiments.wan import wan_configs
    from icecream import ic

    # Initialize logger for standalone execution
    init_logger()
    logger.info("=" * 80)
    logger.info("Starting WanValidator test script")
    logger.info("=" * 80)
    
    # Parse config from CLI arguments (passed via torchrun or command line)
    # Usage: torchrun ... -m torchtitan.experiments.wan.validate --job.config_file=<path>
    config_manager = ConfigManager()
    job_config = config_manager.parse_args()  # Uses sys.argv automatically
    logger.info(f"Config loaded from: {job_config.job.config_file}")
    logger.info("Config loaded successfully")
    ic(job_config)
    
    # Get device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Using device: {device}, dtype: {dtype}")
    
    # Build tokenizer
    logger.info("Building tokenizer...")
    tokenizer = build_wan_tokenizer(job_config)
    logger.info("Tokenizer built successfully")
    
    # Build validator (this will also create the dataloader)
    logger.info("Building validator...")
    validator = build_wan_validator(
        job_config=job_config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        parallel_dims=ParallelDims(
            dp_replicate=1,
            dp_shard=1,  # No sharding for single-device test
            cp=1,
            tp=1,
            pp=1,
            ep=1,  # Expert parallelism
            etp=1,  # Expert tensor parallelism
            world_size=1,
        ),
        loss_fn=None,  # Not needed for dataloader test
        validation_context=None,
        maybe_enable_amp=None,
    )
    logger.info("Validator built successfully")
    ic(validator)

    # Get model args for VAE params
    model_args = wan_configs[job_config.model.flavor]
    ic(model_args)
    
    
    logger.info(f"Loading VAE from: {job_config.encoder.wan_vae_path}")
    wan_video_vae = load_wan_vae(
        chkpt_path=job_config.encoder.wan_vae_path,
        wan_vae_params=model_args.wan_video_vae_params,
        device=device,
        dtype=dtype,
    )
    logger.info("VAE loaded successfully")
    
    logger.info(f"Loading T5 encoder: {job_config.encoder.t5_encoder}")
    t5_encoder = WanEmbedder(
        version=job_config.encoder.t5_encoder,
    ).to(device=device, dtype=dtype)
    logger.info("T5 encoder loaded successfully")
    
    # Initialize validator with encoders
    validator.wan_init(
        device=device,
        _dtype=dtype,
        wan_video_vae=wan_video_vae,
        t5_encoder=t5_encoder,
        precomputed_t5_embedding=None,
    )
    logger.info("Validator initialized with encoders")
    
    # Test the dataloader by iterating through a few batches
    logger.info("=" * 80)
    logger.info("Testing validation dataloader...")
    logger.info("=" * 80)
    
    num_batches_to_test = 3
    for batch_idx, (input_dict, labels) in enumerate(validator.validation_dataloader):
        if batch_idx >= num_batches_to_test:
            break
        ic(input_dict.keys())
        ic(input_dict["video_frames"].shape)
        ic(labels.shape)
        assert torch.all(input_dict["video_frames"] == labels)
        video_frames = input_dict["video_frames"]
        video_frames = video_frames.to(device=device, dtype=dtype).permute(0, 4, 1, 2, 3)
        ic(video_frames.type(), video_frames.device)
        ic(video_frames.shape)
        with torch.inference_mode():
            latents = wan_video_vae.encode(videos=video_frames, device=device, tiled=True)
            ic(latents.shape)
            latentes = latents.permute(0, 2, 1, 3, 4)
            ic(latentes.shape)
            # TODO: fix the tiled execution path 
            reconstructed_video = wan_video_vae.decode(hidden_states=latents, device=device, tiled=False)
        reconstructed_video = reconstructed_video.clamp(-1.0, 1.0)
        
        # Convert back to same dtype as original for PSNR calculation
        # Both should be float32 for accurate PSNR calculation
        video_frames_cpu =  video_frames.cpu().float()
        reconstructed_cpu = reconstructed_video.cpu().float()
        
        # Calculate PSNR per frame
        # Both videos are in (B, C, T, H, W) format
        B, C, T, H, W = video_frames_cpu.shape
        psnr_values = []
        
        logger.info(f"\nCalculating PSNR for {T} frames...")
        for t in range(T):
            # Extract frame t: (B, C, H, W)
            original_frame = video_frames_cpu[:, :, t, :, :]  # (B, C, H, W)
            reconstructed_frame = reconstructed_cpu[:, :, t, :, :]  # (B, C, H, W)
            
            # Compute PSNR for this frame across batch
            # data_range=2.0 because values are in [-1, 1] range (range = 2.0)
            # reduction="none" to get per-sample PSNR, dim=(1,2,3) to reduce over C, H, W
            psnr_frame = peak_signal_noise_ratio(
                reconstructed_frame,
                original_frame,
                data_range=2.0,  # Range is [-1, 1] = 2.0
                reduction="none",
                dim=(1, 2, 3),  # Reduce over C, H, W, keep batch dimension
            )
            
            # Ensure it's 1D tensor: (B,)
            if psnr_frame.dim() == 0:
                psnr_frame = psnr_frame.unsqueeze(0)
            psnr_values.append(psnr_frame)
        
        # Stack to get (T, B) then transpose to (B, T)
        if len(psnr_values) > 0:
            psnr_values = torch.stack(psnr_values, dim=0)  # (T, B)
            if psnr_values.dim() == 2:
                psnr_values = psnr_values.transpose(0, 1)  # (B, T)
        
        # psnr_values shape: (B, T) - PSNR for each batch and frame
        logger.info("\nPSNR Results:")
        logger.info(f"  - PSNR shape: {psnr_values.shape}")
        logger.info(f"  - PSNR per frame (mean across batch): {psnr_values.mean(dim=0)}")
        logger.info(f"  - PSNR per batch (mean across frames): {psnr_values.mean(dim=1)}")
        logger.info(f"  - Overall PSNR (mean): {psnr_values.mean().item():.4f} dB")
        logger.info(f"  - PSNR min: {psnr_values.min().item():.4f} dB")
        logger.info(f"  - PSNR max: {psnr_values.max().item():.4f} dB")
        
        # This PSNR represents the upper bound - best possible reconstruction
        # Any model-generated video should have PSNR <= this value
        logger.info("\n" + "=" * 80)
        logger.info("VAE PSNR Test Summary:")
        logger.info(f"  - This PSNR ({psnr_values.mean().item():.4f} dB) is the UPPER BOUND")
        logger.info("  - Model-generated videos should have PSNR <= this value")
        logger.info("  - Higher PSNR = better reconstruction quality")
        logger.info("=" * 80)
        # Save frames with highest and lowest PSNR for each video in batch
        logger.info("\n" + "=" * 80)
        logger.info("Saving frames with highest and lowest PSNR...")
        logger.info("=" * 80)
        
        from PIL import Image
        from datetime import datetime
        import secrets
        
        # Generate timestamp and random string for folder name
        # Format: YYYYMMDD_HHMMSS_randomstring
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = secrets.token_hex(4)  # 8 character random hex string
        folder_name = f"{timestamp}_{random_str}_vae_psnr_frames"
        
        # Get logs directory from job config or use default
        logs_dir = os.path.join(
            job_config.job.dump_folder,
            folder_name
        )
        os.makedirs(logs_dir, exist_ok=True)
        logger.info(f"Saving frames to: {logs_dir}")
        
        # psnr_values shape: (B, T) - PSNR for each batch and frame
        for b in range(B):
            # Get PSNR values for this video: (T,)
            video_psnr = psnr_values[b]  # Shape: (T,)
            
            # Find indices of highest and lowest PSNR frames
            max_psnr_idx = video_psnr.argmax().item()
            min_psnr_idx = video_psnr.argmin().item()
            
            max_psnr_value = video_psnr[max_psnr_idx].item()
            min_psnr_value = video_psnr[min_psnr_idx].item()
            
            logger.info(f"\nVideo {b}:")
            logger.info(f"  - Highest PSNR: frame {max_psnr_idx}, value: {max_psnr_value:.4f} dB")
            logger.info(f"  - Lowest PSNR: frame {min_psnr_idx}, value: {min_psnr_value:.4f} dB")
            
            # Extract frames from original and reconstructed videos
            # Both are in (B, C, T, H, W) format
            original_max_frame = video_frames_cpu[b, :, max_psnr_idx, :, :]  # (C, H, W)
            reconstructed_max_frame = reconstructed_cpu[b, :, max_psnr_idx, :, :]  # (C, H, W)
            
            original_min_frame = video_frames_cpu[b, :, min_psnr_idx, :, :]  # (C, H, W)
            reconstructed_min_frame = reconstructed_cpu[b, :, min_psnr_idx, :, :]  # (C, H, W)
            
            # Convert from [-1, 1] float to [0, 255] uint8 for saving
            def frame_to_uint8(frame):
                """Convert frame from [-1, 1] float to [0, 255] uint8."""
                frame = frame.clamp(-1.0, 1.0)
                frame = (frame + 1.0) * 127.5  # Map [-1, 1] to [0, 255]
                frame = frame.clamp(0, 255)
                return frame.byte()
            
            # Convert frames to uint8
            original_max_frame_uint8 = frame_to_uint8(original_max_frame)
            reconstructed_max_frame_uint8 = frame_to_uint8(reconstructed_max_frame)
            original_min_frame_uint8 = frame_to_uint8(original_min_frame)
            reconstructed_min_frame_uint8 = frame_to_uint8(reconstructed_min_frame)
            
            # Rearrange from (C, H, W) to (H, W, C) for PIL
            def chw_to_hwc(frame):
                """Convert from (C, H, W) to (H, W, C)."""
                return frame.permute(1, 2, 0)
            
            original_max_frame_hwc = chw_to_hwc(original_max_frame_uint8)
            reconstructed_max_frame_hwc = chw_to_hwc(reconstructed_max_frame_uint8)
            original_min_frame_hwc = chw_to_hwc(original_min_frame_uint8)
            reconstructed_min_frame_hwc = chw_to_hwc(reconstructed_min_frame_uint8)
            
            # Convert to numpy and then to PIL Image
            def tensor_to_pil(tensor):
                """Convert tensor (H, W, C) to PIL Image."""
                numpy_array = tensor.cpu().numpy()
                return Image.fromarray(numpy_array)
            
            # Save highest PSNR frames
            original_max_img = tensor_to_pil(original_max_frame_hwc)
            reconstructed_max_img = tensor_to_pil(reconstructed_max_frame_hwc)
            
            original_max_path = os.path.join(
                logs_dir,
                f"batch{b}_frame{max_psnr_idx}_original_max_psnr_{max_psnr_value:.2f}dB.png"
            )
            reconstructed_max_path = os.path.join(
                logs_dir,
                f"batch{b}_frame{max_psnr_idx}_reconstructed_max_psnr_{max_psnr_value:.2f}dB.png"
            )
            
            original_max_img.save(original_max_path)
            reconstructed_max_img.save(reconstructed_max_path)
            logger.info(f"  - Saved max PSNR frames: {original_max_path}, {reconstructed_max_path}")
            
            # Save lowest PSNR frames
            original_min_img = tensor_to_pil(original_min_frame_hwc)
            reconstructed_min_img = tensor_to_pil(reconstructed_min_frame_hwc)
            
            original_min_path = os.path.join(
                logs_dir,
                f"batch{b}_frame{min_psnr_idx}_original_min_psnr_{min_psnr_value:.2f}dB.png"
            )
            reconstructed_min_path = os.path.join(
                logs_dir,
                f"batch{b}_frame{min_psnr_idx}_reconstructed_min_psnr_{min_psnr_value:.2f}dB.png"
            )
            
            original_min_img.save(original_min_path)
            reconstructed_min_img.save(reconstructed_min_path)
            logger.info(f"  - Saved min PSNR frames: {original_min_path}, {reconstructed_min_path}")
        
        logger.info(f"\n✓ All frames saved to: {logs_dir}")
        logger.info("=" * 80)
        
    # Optionally test preprocessing
    logger.info("\nTesting data preprocessing...")
    for batch_idx, (input_dict, labels) in enumerate(validator.validation_dataloader):
        if batch_idx >= 1:  # Just test one batch
            break
        
        # Add labels to input_dict as expected by preprocess_data
        input_dict["image"] = labels
        assert torch.all(input_dict["video_frames"] == labels)

        
        # Test preprocessing (VAE encoding + T5 encoding)
        processed = preprocess_data(
            device=device,
            dtype=dtype,
            wan_video_vae=wan_video_vae,
            t5_encoder=t5_encoder,
            batch=input_dict,
            precomputed_t5_embedding=None,
        )
        ic(processed.keys())
        ic(processed["latents"].shape)
        ic(processed["t5_encodings"].shape)
        
