# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Disable CUDA graphs before any torch imports (for torch.compile compatibility with VAE cache)
import os

os.environ["TORCHINDUCTOR_CUDAGRAPH_TREES"] = "0"

"""
Offline script to pre-encode videos with VAE and save latents to disk.

Usage:
    python -m torchtitan.experiments.wan.encode_latents \
        --dataset_path ./dataset/world_model_raw_data/train_v2.0_raw \
        --output_dir ./dataset/world_model_raw_data/train_v2.0_latents \
        --vae_path assets/hf/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
        --num_samples 100000 \
        --batch_size 128

This creates latent files that can be loaded during training instead of
encoding videos on-the-fly, providing significant speedup.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from icecream import ic
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ic.configureOutput(includeContext=True)


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


from torchtitan.experiments.wan.model.dataset import RawVideoDataset
from torchtitan.experiments.wan.model.wan_vae import load_wan_vae, WanVAEParams


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-encode videos to latents")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the raw video dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save encoded latents",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="assets/hf/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        help="Path to VAE checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100000,
        help="Number of samples to encode (0 = all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for VAE encoding",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting sample index (for resuming or parallel encoding)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for encoding",
    )
    parser.add_argument(
        "--downsampled",
        type=int,
        default=1,
        help="Downsampling factor for video frames",
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        default=77,
        help="Number of frames per clip",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="pt",
        choices=["pt", "safetensors"],
        help="Format to save latents",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers for parallel video loading",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=1,
        help="Number of batches to prefetch per worker",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for VAE encoder (faster but slower startup)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    # Setup device - use local_rank for multi-GPU
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Create output directory (only main process)
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()  # Wait for directory creation

    if is_main:
        print("=" * 80)
        print("Pre-encoding videos to latents")
        print("=" * 80)
        print(f"  Dataset path: {args.dataset_path}")
        print(f"  Output dir: {args.output_dir}")
        print(f"  VAE path: {args.vae_path}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Num workers: {args.num_workers}")
        print(f"  Prefetch factor: {args.prefetch_factor}")
        print(f"  Dtype: {args.dtype}")
        print(f"  Compile: {args.compile}")
        print(f"  World size: {world_size}")
    ic(rank, world_size, local_rank, device)

    # Load VAE
    print("\nLoading VAE...")
    vae_params = WanVAEParams(vae_type="38", z_dim=48, dim=160)
    ic(vae_params)
    vae = load_wan_vae(
        args.vae_path,
        vae_params,
        device=device,
        dtype=dtype,
    )
    print("  ✓ VAE loaded")

    # Compile VAE encoder for faster inference
    if args.compile:
        print("  Compiling VAE encoder with torch.compile...")
        vae.model.encoder = torch.compile(
            vae.model.encoder,
            mode="max-autotune-no-cudagraphs",
            backend="inductor",
            fullgraph=False,
            dynamic=False,
        )
        print("  ✓ VAE encoder compiled")

    # Load dataset
    print("\nLoading dataset...")
    ic(args.dataset_path, args.downsampled, args.clip_length)
    dataset = RawVideoDataset(
        data_dir=args.dataset_path,
        downsampled=args.downsampled,
        clip_length=args.clip_length,
    )
    dataset_size = len(dataset)
    ic(dataset_size)
    print(f"  ✓ Dataset loaded: {dataset_size} samples")

    # Debug: check first sample
    first_sample = dataset[0]
    ic(first_sample.keys() if isinstance(first_sample, dict) else type(first_sample))

    # Determine range to encode
    start_idx = args.start_idx
    end_idx = (
        min(start_idx + args.num_samples, dataset_size)
        if args.num_samples > 0
        else dataset_size
    )
    total_to_encode = end_idx - start_idx

    # Distribute work across ranks - each rank handles indices where idx % world_size == rank
    all_indices = list(range(start_idx, end_idx))
    my_indices = [idx for idx in all_indices if idx % world_size == rank]
    num_to_encode = len(my_indices)

    ic(rank, total_to_encode, num_to_encode, my_indices[:5] if my_indices else [])

    if is_main:
        print(f"\nEncoding samples {start_idx} to {end_idx} ({total_to_encode} total)")
        print(f"  Per-rank samples: ~{num_to_encode}")
        print(f"  Estimated output size: ~{total_to_encode * 0.5:.1f} MB")

    # Create a subset dataset for this rank's indices
    subset_dataset = Subset(dataset, my_indices)

    # Custom collate function to return indices along with data
    def collate_with_indices(batch):
        # batch is a list of (video_frames, robot_states) tuples
        videos = torch.stack(
            [
                b[0] if isinstance(b[0], torch.Tensor) else torch.from_numpy(b[0])
                for b in batch
            ],
            dim=0,
        )
        return videos

    # Create DataLoader with parallel workers
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collate_with_indices,
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True

    dataloader = DataLoader(subset_dataset, **loader_kwargs)

    if is_main:
        print(
            f"  DataLoader created with {args.num_workers} workers, prefetch={args.prefetch_factor}"
        )

    # Encoding loop
    encode_start = time.perf_counter()
    encoded_count = 0
    skipped_count = 0
    batch_idx = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Rank {rank}", disable=not is_main)
        for videos in pbar:
            # Calculate which original indices this batch corresponds to
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, num_to_encode)
            batch_indices = my_indices[start:end]
            batch_idx += 1

            # Check which samples need encoding (skip existing)
            indices_to_encode = []
            videos_to_encode = []
            for i, idx in enumerate(batch_indices):
                output_path = output_dir / f"latent_{idx:08d}.pt"
                if not output_path.exists():
                    indices_to_encode.append(idx)
                    videos_to_encode.append(videos[i])
                else:
                    skipped_count += 1

            if not indices_to_encode:
                continue

            # Stack only videos that need encoding
            videos_batch = torch.stack(videos_to_encode, dim=0)  # (B, T, H, W, C)
            videos_batch = videos_batch.to(
                device=device, dtype=dtype, non_blocking=True
            )
            videos_batch = videos_batch.permute(
                0, 1, 4, 2, 3
            )  # (B, T, H, W, C) -> (B, T, C, H, W)

            # Normalize from [0, 255] to [-1, 1]
            videos_batch = videos_batch * (2.0 / 255.0) - 1.0
            videos_batch = videos_batch.transpose(
                1, 2
            )  # (B, T, C, H, W) -> (B, C, T, H, W)

            # Encode with VAE
            latents = vae.encode(videos_batch, device=device, tiled=False)

            # Save each latent
            for i, idx in enumerate(indices_to_encode):
                latent = latents[i].cpu()
                output_path = output_dir / f"latent_{idx:08d}.pt"

                if args.save_format == "pt":
                    torch.save(latent, output_path)
                else:
                    # safetensors format
                    from safetensors.torch import save_file

                    save_file(
                        {"latent": latent}, output_path.with_suffix(".safetensors")
                    )

                encoded_count += 1

            # Update progress bar
            pbar.set_postfix(encoded=encoded_count, skipped=skipped_count)

    # Sync all ranks before summary
    if world_size > 1:
        dist.barrier()

    # Summary
    encode_time = time.perf_counter() - encode_start
    ic(rank, encoded_count, skipped_count, encode_time)

    if is_main:
        print("\n" + "=" * 80)
        print("Encoding complete!")
        print("=" * 80)
        print(f"  Encoded by this rank: {encoded_count} samples")
        print(f"  Skipped (existing): {skipped_count} samples")
        print(
            f"  Time: {encode_time:.1f}s ({encoded_count / max(encode_time, 0.1):.1f} samples/sec)"
        )
        print(f"  Output: {args.output_dir}")

    # Save metadata (only main rank)
    if is_main:
        metadata = {
            "dataset_path": args.dataset_path,
            "vae_path": args.vae_path,
            "dtype": args.dtype,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "total_samples": total_to_encode,
            "downsampled": args.downsampled,
            "clip_length": args.clip_length,
            "world_size": world_size,
        }
        torch.save(metadata, output_dir / "metadata.pt")
        print(f"  Metadata saved to {output_dir / 'metadata.pt'}")

    # Cleanup distributed
    cleanup_distributed()


if __name__ == "__main__":
    main()
