import torch
import os
import json
import math
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
import argparse
from torchmetrics.functional.image import peak_signal_noise_ratio


from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger
# from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

from wan_video_1x import Wan1xVideoPipeline as WanVideoPipeline
from wan_video_vae_1x import WanVideoVAE38
from wan_video_1x_dit import WanModel1x
from utils import load_training_state, save_training_state, set_seed
from dataset import RawVideoDataset

import logging
from icecream import ic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_validation(model, args, accelerator, torch_dtype, log_prefix="val"):
    """Run validation on the entire validation dataset"""
    validation_dataset = RawVideoDataset(
        args.validation_dataset_base_path,
        downsampled=args.downsampled,
        window_size=77,
        robot_temporal_mode=args.robot_temporal_mode,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=1,
        drop_last=False,
        num_workers=args.dataset_num_workers,
    )
    validation_dataloader = accelerator.prepare(validation_dataloader)

    all_val_psnr_values = []

    with torch.no_grad():
        pbar = tqdm(
            validation_dataloader,
            ncols=100,
            disable=not accelerator.is_local_main_process,
        )
        for data in pbar:
            video_frames, robot_states = data
            video_frames = video_frames.to(dtype=torch_dtype)
            robot_states = robot_states.to(dtype=torch_dtype)
            max_value = 1.0
            min_value = -1.0
            video_frames = video_frames * ((max_value - min_value) / 255.0) + min_value
            video_frames = video_frames.permute(0, 1, 4, 2, 3)

            generated_video = model.module.validation_forward(
                video_frames=video_frames,
                robot_states=robot_states,
                height=512,
                width=512,
                num_inference_steps=args.num_inference_steps,
            )

            last_frames_gt = video_frames[:, -1, :, :, :]
            generated_video = generated_video.to(
                dtype=torch.bfloat16, device=accelerator.device
            )
            generated_video = generated_video.permute(0, 2, 1, 3, 4)
            last_frames_pred = generated_video[:, -1, :, :, :]

            psnr_values = peak_signal_noise_ratio(
                last_frames_pred, last_frames_gt, data_range=2.0, reduction="none"
            )
            all_val_psnr_values.append(psnr_values.flatten())

            # Display current batch average
            batch_psnr = psnr_values.mean()
            pbar.set_description(f"Val PSNR: {batch_psnr.item():.2f}")

    # Compute final validation PSNR
    if len(all_val_psnr_values) > 0:
        all_val_psnr_tensor = torch.cat(all_val_psnr_values, dim=0)
        all_val_psnr_gathered = accelerator.gather_for_metrics(all_val_psnr_tensor)
        final_val_psnr = all_val_psnr_gathered.mean()

        accelerator.print(f"Final Validation PSNR: {final_val_psnr.item():.4f} dB")
        accelerator.log({f"{log_prefix}/psnr": final_val_psnr.item()})

        return final_val_psnr.item()

    return None


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_base_path",
        type=str,
        default="/root/riccardo/dataset/world_model_raw_data/val_v2.0_raw/",
        help="Base path of the dataset.",
    )
    parser.add_argument(
        "--validation_dataset_base_path",
        type=str,
        default="/root/riccardo/dataset/world_model_raw_data/val_v2.0_raw/",
        help="Base path of the validation dataset.",
    )
    parser.add_argument(
        "--dataset_metadata_path",
        type=str,
        default=None,
        help="Path to the metadata file of the dataset.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=77,
        help="Number of frames per video. Frames are sampled from the video prefix.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducibility."
    )

    parser.add_argument(
        "--data_file_keys",
        type=str,
        default="image,video",
        help="Data file keys in the metadata. Comma-separated.",
    )
    parser.add_argument(
        "--dataset_repeat",
        type=int,
        default=1,
        help="Number of times to repeat the dataset per epoch.",
    )
    parser.add_argument(
        "--downsampled",
        type=int,
        default=4,
        help="Rate downsample the dataset.",
        choices=[1, 2, 4],
    )
    parser.add_argument(
        "--window_size", type=int, default=8, help="Window size for the dataset."
    )

    parser.add_argument(
        "--num_inference_steps", type=int, default=10, help="Number of inference steps."
    )
    parser.add_argument(
        "--train_psnr_step",
        type=int,
        default=100,
        help="Number of steps to compute teh psnr on the train batch.",
    )

    parser.add_argument(
        "--robot_temporal_mode",
        type=str,
        default="downsample",
        help="Robot temporal mode.",
        choices=[
            "downsample",
            "full"
        ],
    )

    parser.add_argument(
        "--model_paths",
        type=str,
        default=None,
        help="Paths to load models. In JSON format.",
    )
    parser.add_argument(
        "--model_id_with_origin_paths",
        type=str,
        default=None,
        help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate."
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument(
        "--output_path", type=str, default="./models", help="Output save path."
    )
    parser.add_argument(
        "--remove_prefix_in_ckpt",
        type=str,
        default="pipe.dit.",
        help="Remove prefix in ckpt.",
    )
    parser.add_argument(
        "--trainable_models",
        type=str,
        default=None,
        help="Models to train, e.g., dit, vae, text_encoder.",
    )
    parser.add_argument(
        "--lora_base_model",
        type=str,
        default=None,
        help="Which model LoRA is added to.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Which layers LoRA is added to.",
    )
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.",
    )
    parser.add_argument(
        "--extra_inputs", default=None, help="Additional model inputs, comma-separated."
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to offload gradient checkpointing to CPU memory.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--max_timestep_boundary",
        type=float,
        default=1.0,
        help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).",
    )
    parser.add_argument(
        "--min_timestep_boundary",
        type=float,
        default=0.0,
        help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).",
    )
    parser.add_argument(
        "--find_unused_parameters",
        default=False,
        action="store_true",
        help="Whether to find unused parameters in DDP.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.",
    )
    parser.add_argument(
        "--dataset_num_workers",
        type=int,
        default=14,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay."
    )
    parser.add_argument(
        "--last_latents_only",
        default=False,
        action="store_true",
        help="Whether to only use the last 4 frames for loss computation.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps for learning rate scheduler.",
    )
    parser.add_argument(
        "--use_cosine_schedule",
        action="store_true",
        help="Use cosine annealing with linear warmup instead of constant LR.",
    )

    parser.add_argument(
        "--no_validation",
        default=False,
        action="store_true",
        help="Whether to skip validation.",
    )
    parser.add_argument(
        "--no_robot_conditioning",
        default=False,
        action="store_true",
        help="Whether to use robot conditioning.",
    )
    parser.add_argument(
        "--adaln_mode",
        type=str,
        default="additive",
        choices=["additive", "multiplicative"],
        help="AdaLN mode for robot conditioning: 'additive' or 'multiplicative'.",
    )

    # CFG training parameters
    parser.add_argument(
        "--cfg_training",
        default=False,
        action="store_true",
        help="Whether to use CFG training.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="CFG scale for training (should be 1.0, CFG scaling happens during inference).",
    )
    parser.add_argument(
        "--cfg_robot_prob",
        type=float,
        default=0.1,
        help="Probability of dropping robot conditioning during training (enables CFG at inference).",
    )

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint dir to resume the run from",
    )
    parser.add_argument(
        "--no_save_training_state",
        action="store_true",
        default=False,
        help="Save training state (optimizer/scheduler) for resuming",
    )
    args = parser.parse_args()

    if args.downsampled == 1:
        args.frames_per_clip = 77
        args.num_cond_frames = 17
    elif args.downsampled == 2:
        args.frames_per_clip = 41
        args.num_cond_frames = 9
    elif args.downsampled == 4:
        args.frames_per_clip = 21
        args.num_cond_frames = 5

    return args


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        last_latents_only: bool = False,
        use_robot_conditioning: bool = False,
        adaln_mode: str = "additive",
        robot_temporal_mode: str = "downsample",
        cfg_scale: float = 1.0,
        cfg_robot_prob: float = 0.1,
        downsampled: int = 1,
        cfg_training: bool = False,
    ):
        super().__init__()

        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [
                ModelConfig(
                    model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]
                )
                for i in model_id_with_origin_paths
            ]

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            robot_temporal_mode=robot_temporal_mode,
        )

        self.use_robot_conditioning = use_robot_conditioning
        self.adaln_mode = adaln_mode
        self.robot_temporal_mode = robot_temporal_mode
        self.cfg_training = cfg_training

        # Load WanVideoVAE38 for convenience (removes a bunch of prints)
        vae = WanVideoVAE38()
        vae.load_state_dict(self.pipe.vae.state_dict())
        self.pipe.vae = vae.to(dtype=self.pipe.torch_dtype)
        dit_orignal_params_dict = {
            "has_image_input": False,
            "patch_size": [1, 2, 2],
            "in_dim": 48,
            "dim": 3072,
            "ffn_dim": 14336,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 48,
            "num_heads": 24,
            "num_layers": 30,
            "eps": 1e-06,
            "seperated_timestep": True,
            "require_clip_embedding": False,
            "require_vae_embedding": False,
            "fuse_vae_embedding_in_latents": True,
        }
        dit = WanModel1x(
            **dit_orignal_params_dict,
            use_robot_conditioning=self.use_robot_conditioning,
            robot_state_dim=25,
            adaln_mode=self.adaln_mode,
            robot_temporal_mode=self.robot_temporal_mode,
            cfg_training=self.cfg_training,
        )
        dit.load_state_dict(self.pipe.dit.state_dict(), strict=False)
        self.pipe.dit = dit.to(dtype=self.pipe.torch_dtype)

        ## Switching pipe to training mode
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Freeze untrainable models
        self.pipe.freeze_except(
            [] if trainable_models is None else trainable_models.split(",")
        )
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(
                    f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys"
                )
                if len(load_result[1]) > 0:
                    print(
                        f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}"
                    )
                if self.use_robot_conditioning:
                    print(f"Loading robot conditioning weights from LoRA checkpoint")
                    state_dict = load_state_dict(lora_checkpoint)
                    self.safe_load_robot_conditioning_weights(state_dict)
            setattr(self.pipe, lora_base_model, model)

        if self.use_robot_conditioning:
            self.pipe.dit.robot_state_compression.train()
            self.pipe.dit.robot_state_compression.requires_grad_(True)
            self.pipe.dit.robot_temporal_compression.train()
            self.pipe.dit.robot_temporal_compression.requires_grad_(True)
            self.pipe.dit.r_embedding.train()
            self.pipe.dit.r_embedding.requires_grad_(True)
            self.pipe.dit.r_projection.train()
            self.pipe.dit.r_projection.requires_grad_(True)
            if self.cfg_training:
                self.pipe.dit.no_states_token.requires_grad_(True)

            for block in self.pipe.dit.blocks:
                block.robot_modulation.requires_grad_(True)

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.last_latents_only = last_latents_only
        self.cfg_scale = cfg_scale
        self.cfg_robot_prob = cfg_robot_prob

        if downsampled == 4:
            self.num_cond_frames = 5
            self.clip_length = 21
        elif downsampled == 2:
            self.num_cond_frames = 9
            self.clip_length = 41
        else:
            self.num_cond_frames = 17
            self.clip_length = 77

    def safe_load_robot_conditioning_weights(self, state_dict):
        """
        Safely load robot conditioning weights from checkpoint, filtering out incompatible
        robot_temporal_compression layers when switching between different temporal modes.
        """
        # Extract robot conditioning keys based on current mode
        if self.robot_temporal_mode == "hierarchical":
            robot_keys = [
                k
                for k in state_dict.keys()
                if any(
                    prefix in k for prefix in ["robot_conditioning", "robot_modulation"]
                )
            ]
        else:
            robot_keys = [
                k
                for k in state_dict.keys()
                if any(
                    prefix in k
                    for prefix in [
                        "robot_state_compression",
                        "robot_temporal_compression",
                        "r_embedding",
                        "r_projection",
                        "robot_modulation",
                    ]
                )
            ]

        if not robot_keys:
            print("No robot conditioning keys found in checkpoint")
            return

        # Create filtered state dict with only robot conditioning weights
        robot_state_dict = {k: v for k, v in state_dict.items() if k in robot_keys}

        # Check for temporal compression layer compatibility (skip for hierarchical mode)
        incompatible_keys = []
        if self.robot_temporal_mode != "hierarchical":
            temporal_keys = [k for k in robot_keys if "robot_temporal_compression" in k]

            for key in temporal_keys:
                if key in robot_state_dict:
                    checkpoint_shape = robot_state_dict[key].shape
                    try:
                        model_param = getattr(self.pipe.dit, key.split(".")[0])
                        for part in key.split(".")[1:]:
                            if part.isdigit():
                                model_param = model_param[int(part)]
                            else:
                                model_param = getattr(model_param, part)
                        model_shape = model_param.shape

                        if checkpoint_shape != model_shape:
                            print(
                                f"Shape mismatch for {key}: checkpoint {checkpoint_shape} vs model {model_shape}"
                            )
                            incompatible_keys.append(key)
                    except (AttributeError, IndexError) as e:
                        print(f"Could not check shape for {key}: {e}")
                        incompatible_keys.append(key)

        # Remove incompatible temporal compression keys
        for key in incompatible_keys:
            # print(f"Skipping incompatible key: {key}")
            robot_state_dict.pop(key, None)

        # Load the filtered state dict
        if robot_state_dict:
            load_result = self.pipe.dit.load_state_dict(robot_state_dict, strict=False)
            print(f"Loaded {len(robot_state_dict)} robot conditioning parameters")

            if load_result.missing_keys:
                # print(f"Missing keys (will be randomly initialized): {load_result.missing_keys}")
                print(
                    f"Missing keys (will be randomly initialized): {len(load_result.missing_keys)}"
                )
            if load_result.unexpected_keys:
                # print(f"Unexpected keys (ignored): {load_result.unexpected_keys}")
                print(f"Unexpected keys (ignored): {len(load_result.unexpected_keys)}")
        else:
            print("No compatible robot conditioning weights found to load")

    def forward(
        self,
        batch,
    ):
        input_video, robot_states = batch
        input_video = input_video.to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )
        # Normalize video frames between -1 and 1
        max_value = 1.0
        min_value = -1.0
        input_video = input_video * ((max_value - min_value) / 255.0) + min_value

        # B, T, H, W, C -> B, T, C, H, W
        input_video = input_video.permute(0, 1, 4, 2, 3)

        robot_states = robot_states.to(
            device=self.pipe.device, dtype=self.pipe.torch_dtype
        )

        loss = self.pipe.training_loss(
            input_video=input_video,
            robot_states=robot_states,
            max_timestep_boundary=self.max_timestep_boundary,
            min_timestep_boundary=self.min_timestep_boundary,
            tiled=False,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            num_cond_frames=self.num_cond_frames,
            last_latents_only=self.last_latents_only,
            cfg_scale=self.cfg_scale,
            cfg_robot_prob=self.cfg_robot_prob,
            cfg_training=self.cfg_training,
        )
        return loss

    def validation_forward(
        self,
        video_frames,
        prompt="",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        height=512,
        width=512,
        num_inference_steps=10,
        robot_states=None,
    ):
        return self.pipe(
            input=video_frames,
            num_cond_frames=self.num_cond_frames,
            num_frames=self.clip_length,
            robot_states=robot_states,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=None,
            height=height,
            width=width,
            tiled=True,
            num_inference_steps=num_inference_steps,
            validation=True,
        )


def main():
    args = wan_parser()
    logger.info(f"Start Training")
    logger.info(f"args: {args}")

    set_seed(args.seed)

    dataset = RawVideoDataset(
        args.dataset_base_path,
        repeat=args.dataset_repeat,
        downsampled=args.downsampled,
        window_size=args.window_size,
        robot_temporal_mode=args.robot_temporal_mode,
    )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        use_robot_conditioning=not args.no_robot_conditioning,
        downsampled=args.downsampled,
        adaln_mode=args.adaln_mode,
        robot_temporal_mode=args.robot_temporal_mode,
        last_latents_only=args.last_latents_only,
        cfg_scale=args.cfg_scale,
        cfg_robot_prob=args.cfg_robot_prob,
        cfg_training=args.cfg_training,
    )
    torch_dtype = model.pipe.torch_dtype

    model_logger = ModelLogger(
        args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataset_num_workers,
        drop_last=True,
        persistent_workers=True,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
        log_with="wandb",
    )
    # TODO: add run name
    accelerator.init_trackers("1x-challenge", config=vars(args))
    # Create learning rate scheduler
    if args.use_cosine_schedule:
        # Calculate total training steps
        total_steps = len(dataset) * args.num_epochs

        # Create warmup + cosine scheduler
        def lr_lambda(step):
            if step < args.warmup_steps:
                # Linear warmup from 1e-6 to target LR
                min_lr_ratio = 1e-6 / args.learning_rate
                warmup_ratio = step / args.warmup_steps
                return min_lr_ratio + warmup_ratio * (1.0 - min_lr_ratio)
            else:
                # Cosine annealing
                progress = (step - args.warmup_steps) / (
                    total_steps - args.warmup_steps
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    if args.use_cosine_schedule:
        accelerator.print(
            f"Using cosine schedule with {args.warmup_steps} warmup steps and {total_steps} total steps"
        )
    else:
        accelerator.print("Using constant learning rate")

    start_epoch = 0
    start_step = 0
    if args.resume_from is not None:
        # if accelerator.is_local_main_process: import ipdb; ipdb.set_trace()

        start_epoch, start_step = load_training_state(
            args.resume_from, optimizer, scheduler, accelerator
        )
        accelerator.print(f"Resuming from epoch {start_epoch}, step {start_step}")

    accelerator.print(f"Start Training Loop")
    global_step = start_step
    for epoch_id in range(start_epoch, args.num_epochs):
        pbar = tqdm(
            dataloader, ncols=100, disable=not accelerator.is_local_main_process
        )
        for step, data in enumerate(pbar):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, args.save_steps)
                scheduler.step(global_step)
                global_step += 1
            train_loss = accelerator.gather_for_metrics(loss).mean()
            current_lr = scheduler.get_last_lr()[0]
            accelerator.log({"train/loss": train_loss.item(), "train/lr": current_lr})
            pbar.set_description(
                f"Epoch {epoch_id}, Loss: {train_loss.item():.4f}, LR: {current_lr:.2e}"
            )
            # break

            if args.train_psnr_step is not None:
                if step % args.train_psnr_step == 0 and step != 0:
                    # First do quick PSNR on current training batch
                    video_frames, robot_states = data
                    video_frames = video_frames.to(dtype=torch_dtype)
                    robot_states = robot_states.to(dtype=torch_dtype)
                    max_value = 1.0
                    min_value = -1.0
                    video_frames = (
                        video_frames * ((max_value - min_value) / 255.0) + min_value
                    )
                    video_frames = video_frames.permute(0, 1, 4, 2, 3)

                    generated_video = model.module.validation_forward(
                        # generated_video = model.validation_forward(
                        video_frames=video_frames,
                        robot_states=robot_states,
                        height=512,
                        width=512,
                        num_inference_steps=args.num_inference_steps,
                    )
                    last_frames_gt = video_frames[:, -1, :, :, :]
                    generated_video = generated_video.to(
                        dtype=torch.bfloat16, device=accelerator.device
                    )
                    generated_video = generated_video.permute(0, 2, 1, 3, 4)
                    last_frames_pred = generated_video[:, -1, :, :, :]

                    psnr_values = peak_signal_noise_ratio(
                        last_frames_pred,
                        last_frames_gt,
                        data_range=2.0,
                        reduction="none",
                    )
                    batch_mean_psnr = psnr_values.mean()
                    avg_psnr_batch = accelerator.gather_for_metrics(
                        batch_mean_psnr
                    ).mean()
                    accelerator.log({"train/psnr": avg_psnr_batch.item()})
                    accelerator.print(f"train/psnr {avg_psnr_batch.item()}")

                    # Then do full validation dataset
                    psnr_value = run_validation(
                        model, args, accelerator, torch_dtype, log_prefix="val"
                    )

                    # Save checkpoint with PSNR value
                    if psnr_value is not None:
                        model_logger.on_step_end(accelerator, model, save_steps=1)
                        # Save training state as well
                        if not args.no_save_training_state:
                            save_training_state(
                                accelerator=accelerator,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch_id,
                                output_path=args.output_path,
                                psnr_value=psnr_value,
                                step=global_step,
                            )

        if not args.no_save_training_state:
            save_training_state(
                accelerator=accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_id,
                output_path=args.output_path,
                step=global_step,
            )
        # if args.save_steps is None:
        model_logger.on_epoch_end(accelerator, model, epoch_id)

        if not args.no_validation:
            run_validation(model, args, accelerator, torch_dtype, log_prefix="val")

    model_logger.on_training_end(accelerator, model, args.save_steps)
    accelerator.end_training()


if __name__ == "__main__":
    main()
