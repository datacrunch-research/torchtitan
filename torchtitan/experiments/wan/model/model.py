# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, Tensor
import math
import hashlib
from einops import rearrange
from typing import Tuple, Optional
from torchtitan.tools.logging import logger

def hash_state_dict_keys(state_dict):
    """Hash the keys of a state dict to identify model variants."""
    keys = sorted(state_dict.keys())
    key_string = "|".join(keys)
    return hashlib.md5(key_string.encode()).hexdigest()

from torchtitan.experiments.wan.model.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
    precompute_freqs_cis_3d,
    Head,
    DiTBlock,

)

from torchtitan.protocols import ModelProtocol

from .args import WanModelArgs

class WanModel(torch.nn.Module):
    def __init__(self, model_args: WanModelArgs):
        super().__init__()
        self.model_args = model_args
        self.in_dim = model_args.in_dim
        self.out_dim = model_args.out_dim
        self.text_dim = model_args.text_dim
        self.ffn_dim = model_args.ffn_dim
        self.freq_dim = model_args.freq_dim
        self.hidden_dim = model_args.hidden_dim
        self.patch_size = model_args.patch_size
        self.num_layers = model_args.num_layers
        logger.info(f"num_layers: {self.num_layers}")
        self.eps = model_args.eps
        self.context_in_dim = model_args.context_in_dim
        self.hidden_size = model_args.hidden_size
        self.mlp_ratio = model_args.mlp_ratio
        self.num_heads = model_args.num_heads
        
        self.patch_embedding = nn.Conv3d(
            self.in_dim, 
            self.hidden_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim), 
            nn.GELU(approximate="tanh"), 
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim), 
            nn.SiLU(), 
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(self.hidden_dim, self.hidden_dim * 6)
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    self.hidden_dim, 
                    self.num_heads, 
                    self.ffn_dim, 
                    self.eps
                )
                for _ in range(self.num_layers)
            ]
        )
        self.head = Head(self.hidden_dim, self.out_dim, self.patch_size, self.eps)
        head_dim = self.hidden_dim // self.num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)
        # downsampled := 4  (from 30fps to ~8fps)
        self.num_cond_frames = 5
        self.clip_length = 21
        self.num_latent_cond = 1 + self.num_cond_frames // 4
    
    
    def init_weights(self, buffer_device=None):
        # # Adapted from DiT weight initialization: https://github.com/facebookresearch/DiT/blob/main/models.py#L189
        # # initialize Linear Layers: img_in, txt_in
        # nn.init.xavier_uniform_(self.img_in.weight)
        # nn.init.constant_(self.img_in.bias, 0)
        # nn.init.xavier_uniform_(self.txt_in.weight)
        # nn.init.constant_(self.txt_in.bias, 0)

        # # Initialize time_in, vector_in (MLPEmbedder)
        # self.time_in.init_weights(init_std=0.02)
        # self.vector_in.init_weights(init_std=0.02)

        # # Initialize transformer blocks:
        # for block in self.single_blocks:
        #     block.init_weights()
        # for block in self.double_blocks:
        #     block.init_weights()

        # # Zero-out output layers:
        # self.final_layer.init_weights()
        pass 
        # TODO: place the pretrained weights loading here
    
    @staticmethod
    def sinusoidal_embedding_1d_batched(dim, position):
        """
        Args:
            dim: embedding dimension
            position: tensor of shape (B, T) where B is batch size, T is sequence length
        Returns:
            tensor of shape (B, T, dim)
        """
        B, T = position.shape

        # Reshape to (B*T,) to process all elements at once
        position_flat = position.view(-1)

        # Apply the original function
        sinusoid = torch.outer(
            position_flat.type(torch.float64),
            torch.pow(
                10000,
                -torch.arange(
                    dim // 2, dtype=torch.float64, device=position.device
                ).div(dim // 2),
            ),
        )
        x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        x = x.to(position.dtype)

        # Reshape back to (B, T, dim)
        return x.view(B, T, dim)
    
    def patchify(
        self,
        x: torch.Tensor,
    ):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )
    
    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Tensor,
        robot_states: Optional[Tensor] = None,
    ) -> Tensor:
        zeroes = torch.zeros(
            (x.shape[0], self.num_latent_cond * x.shape[3] * x.shape[4] // 4),
            dtype=x.dtype,
            device=x.device,
        )
        broadcast_timesteps = timesteps[:, None].repeat(
            1,
            (x.shape[2] - self.num_latent_cond) * x.shape[3] * x.shape[4] // 4,
        )
        timesteps_combined = torch.cat([zeroes, broadcast_timesteps], dim=1).to(
            dtype=x.dtype
        )
        ts = self.time_embedding(
            self.sinusoidal_embedding_1d_batched(self.freq_dim, timesteps_combined)
        )
        t_mod = self.time_projection(ts).unflatten(2, (6, self.hidden_dim))
        context = self.text_embedding(context)
        # context = torch.cat([context] * x.shape[0], dim=0)

        x, (f, h, w) = self.patchify(x)
        # Handle meta device: if freqs are on meta, recompute them on the correct device
        # This happens when FSDP wraps the model and buffers end up on meta device
        if self.freqs[0].device.type == "meta":
            # Recompute freqs on the correct device
            head_dim = self.hidden_dim // self.num_heads
            freqs_tuple = precompute_freqs_cis_3d(head_dim)
            # Move to correct device (freqs are complex, so we don't change dtype)
            freqs_tuple = (
                freqs_tuple[0].to(device=x.device),
                freqs_tuple[1].to(device=x.device),
                freqs_tuple[2].to(device=x.device),
            )
        else:
            freqs_tuple = self.freqs
        
        freqs = (
            torch.cat(
                [
                    freqs_tuple[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs_tuple[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs_tuple[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(f * h * w, 1, -1)
        )
        # Ensure freqs is on the same device as x (important for FSDP/meta device)
        # Note: freqs are complex tensors (for RoPE), so we only move device, not dtype
        if freqs.device != x.device:
            freqs = freqs.to(device=x.device)

        for block in self.blocks:
            x = block(x, context, t_mod, freqs, robot_states)
        
        # Head expects ts (time embedding) not t_mod (projected modulation)
        # Reference: wan_video_1x.py line 560 and wan_video_1x_dit.py line 884
        # ts has shape (B, L, dim) which matches Head.forward() expectation for 3D input
        x = self.head(x, ts)
        x = self.unpatchify(x, (f, h, w))
        return x
        

class _WanModel(nn.Module, ModelProtocol):
    """
    Transformer model for flow matching on sequences.

    Args:
        model_args: WanModelArgs.

    Attributes:
        model_args (TransformerModelArgs): Model configuration arguments.
    """

    def __init__(self, model_args: WanModelArgs):
        super().__init__()

        self.model_args = model_args

        self.in_dim = model_args.in_dim
        self.out_dim = model_args.out_dim
        if model_args.hidden_dim % model_args.num_heads != 0:
            raise ValueError(
                f"Hidden size {model_args.hidden_dim} must be divisible by num_heads {model_args.num_heads}"
            )
        # pe_dim = model_args.hidden_dim // model_args.num_heads
        # if sum(model_args.axes_dim) != pe_dim:
        #     raise ValueError(
        #         f"Got {model_args.axes_dim} but expected positional dim {pe_dim}"
        #     )
        self.hidden_dim = model_args.hidden_dim
        self.num_heads = model_args.num_heads
        # self.pe_embedder = EmbedND(
        #     dim=pe_dim, theta=model_args.theta, axes_dim=model_args.axes_dim
        # )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(model_args.vec_in_dim, self.hidden_size)
        self.txt_in = nn.Linear(model_args.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=model_args.mlp_ratio,
                    qkv_bias=model_args.qkv_bias,
                )
                for _ in range(model_args.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=model_args.mlp_ratio
                )
                for _ in range(model_args.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def init_weights(self, buffer_device=None):
        # Adapted from DiT weight initialization: https://github.com/facebookresearch/DiT/blob/main/models.py#L189
        # initialize Linear Layers: img_in, txt_in
        nn.init.xavier_uniform_(self.img_in.weight)
        nn.init.constant_(self.img_in.bias, 0)
        nn.init.xavier_uniform_(self.txt_in.weight)
        nn.init.constant_(self.txt_in.bias, 0)

        # Initialize time_in, vector_in (MLPEmbedder)
        self.time_in.init_weights(init_std=0.02)
        self.vector_in.init_weights(init_std=0.02)

        # Initialize transformer blocks:
        for block in self.single_blocks:
            block.init_weights()
        for block in self.double_blocks:
            block.init_weights()

        # Zero-out output layers:
        self.final_layer.init_weights()

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


class WanModel1x_(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        use_robot_conditioning: bool = True,
        robot_state_dim: int = 25,
        adaln_mode: str = "additive",  # "additive" or "multiplicative"
        robot_temporal_mode: str = "downsample",  # "downsample", "full", "adaptive"
        r_dim: int = 256,
        cfg_training: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads  # Store num_heads for recomputing freqs on meta device
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.r_dim = r_dim
        self.cfg_training = cfg_training

        # Validate adaln_mode parameter
        assert adaln_mode in ["additive", "multiplicative"], (
            f"adaln_mode must be 'additive' or 'multiplicative', got {adaln_mode}"
        )
        self.adaln_mode = adaln_mode

        assert robot_temporal_mode in [
            "downsample",
            "full"
        ], (
            f"robot_temporal_mode must be 'downsample' or 'full', got {robot_temporal_mode}"
        )
        self.robot_temporal_mode = robot_temporal_mode

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    has_image_input,
                    dim,
                    num_heads,
                    ffn_dim,
                    eps,
                    use_robot_conditioning,
                    adaln_mode,
                )
                for _ in range(num_layers)
            ]
        )

        self.use_robot_conditioning = use_robot_conditioning
        if self.use_robot_conditioning:
            if self.cfg_training:
                self.no_states_token = nn.Parameter(
                    torch.randn(1, 1, self.r_dim) * 0.02
                )
            else:
                self.no_states_token = None

            # B,t,25 -> B,t,r_dim
            # Robot state processing with sinusoidal embeddings
            # Indices 0-20: joint angles, 21-22: binary hand states, 23-24: velocities
            self.robot_state_compression = nn.Sequential(
                nn.Linear(
                    71, self.r_dim
                ),  # 23 original + 23 sin + 23 cos + 2 binary = 71
                nn.SiLU(),
                nn.Linear(self.r_dim, self.r_dim),
            )
            # Temporal compression based on the robot_temporal_mode
            if self.robot_temporal_mode == "downsample":
                self.robot_temporal_compression = nn.Sequential(
                    nn.Conv1d(
                        self.r_dim, self.r_dim, kernel_size=3, stride=2, padding=1
                    ),
                    nn.SiLU(),
                    nn.GroupNorm(8, self.r_dim),
                    nn.Conv1d(
                        self.r_dim, self.r_dim, kernel_size=3, stride=2, padding=1
                    ),
                    nn.SiLU(),
                    nn.GroupNorm(8, self.r_dim),
                )
            elif self.robot_temporal_mode == "full":
                self.robot_temporal_compression = nn.Sequential(
                    nn.Conv1d(
                        self.r_dim, self.r_dim, kernel_size=5, stride=4, padding=2
                    ),
                    nn.SiLU(),
                    nn.GroupNorm(8, self.r_dim),
                    nn.Conv1d(
                        self.r_dim, self.r_dim, kernel_size=5, stride=3, padding=1
                    ),
                    nn.SiLU(),
                    nn.GroupNorm(8, self.r_dim),
                )
            # (B, t_l, r_dim) -> (B, t_l, dim)
            self.r_embedding = nn.Sequential(
                nn.Linear(self.r_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
            )
            self.r_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)



    @staticmethod
    def _add_sinusoids_global(x):
        # x: (B, T, 8) - all continuous (legs + neck + velocities)
        sin = torch.sin(x)
        cos = torch.cos(x)
        return torch.cat([x, sin, cos], dim=-1)  # (B, T, 24)

    @staticmethod
    def _add_sinusoids_fine(x):
        # x: (B, T, 17) where 0-14 are continuous joints, 15-16 are binary hands
        continuous = x[:, :, :15]  # joints 6-20
        binary = x[:, :, 15:]  # hand states 21-22
        sin = torch.sin(continuous)
        cos = torch.cos(continuous)
        return torch.cat([continuous, sin, cos, binary], dim=-1)  # (B, T, 47)

    @staticmethod
    def _add_sinusoids_unified(x):
        # x: (B, T, 25) - all robot state features
        # indices 0-20: joint angles (continuous), 21-22: binary hand states, 23-24: velocities (continuous)
        continuous = torch.cat([x[:, :, :21], x[:, :, 23:25]], dim=-1)  # (B, T, 23)
        binary = x[:, :, 21:23]  # (B, T, 2)
        sin = torch.sin(continuous)
        cos = torch.cos(continuous)
        return torch.cat([continuous, sin, cos, binary], dim=-1)  # (B, T, 71)

    @staticmethod
    def sinusoidal_embedding_1d_batched(dim, position):
        """
        Args:
            dim: embedding dimension
            position: tensor of shape (B, T) where B is batch size, T is sequence length
        Returns:
            tensor of shape (B, T, dim)
        """
        B, T = position.shape

        # Reshape to (B*T,) to process all elements at once
        position_flat = position.view(-1)

        # Apply the original function
        sinusoid = torch.outer(
            position_flat.type(torch.float64),
            torch.pow(
                10000,
                -torch.arange(
                    dim // 2, dtype=torch.float64, device=position.device
                ).div(dim // 2),
            ),
        )
        x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        x = x.to(position.dtype)

        # Reshape back to (B, T, dim)
        return x.view(B, T, dim)

    def patchify(
        self,
        x: torch.Tensor,
    ):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        t_mod: torch.Tensor,
        context: torch.Tensor,
        robot_states: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        num_cond_frames: int = 17,
        **kwargs,
    ):
        # Simple timestep processing - pipeline handles the complex separated timestep logic
        # t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timesteps))
        # t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        # context = self.text_embedding(context)
        # context = torch.cat([context] * x.shape[0] , dim=0)

        # if self.has_image_input:
        #     x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        #     clip_embdding = self.img_emb(clip_feature)
        #     context = torch.cat([clip_embdding, context], dim=1)

        # Robot states are now pre-processed by the pipeline
        r_states = robot_states

        x, (f, h, w) = self.patchify(x)
        # Handle meta device: if freqs are on meta, recompute them on the correct device
        # This happens when FSDP wraps the model and buffers end up on meta device
        if self.freqs[0].device.type == "meta":
            # Recompute freqs on the correct device
            head_dim = self.dim // self.num_heads
            freqs_tuple = precompute_freqs_cis_3d(head_dim)
            # Move to correct device (freqs are complex, so we don't change dtype)
            freqs_tuple = (
                freqs_tuple[0].to(device=x.device),
                freqs_tuple[1].to(device=x.device),
                freqs_tuple[2].to(device=x.device),
            )
        else:
            freqs_tuple = self.freqs

        freqs = (
            torch.cat(
                [
                    freqs_tuple[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs_tuple[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs_tuple[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
        )
        # Ensure freqs is on the same device as x (important for FSDP/meta device)
        # Note: freqs are complex tensors (for RoPE), so we only move device, not dtype
        if freqs.device != x.device:
            freqs = freqs.to(device=x.device)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            context,
                            t_mod,
                            freqs,
                            r_states,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        r_states,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs, r_states)

        x = self.head(x, timesteps)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()


class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(
                        name_.split(".")[:1]
                        + [name.split(".")[1]]
                        + name_.split(".")[2:]
                    )
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config

    def from_civitai(self, state_dict):
        state_dict = {
            name: param
            for name, param in state_dict.items()
            if not name.startswith("vace")
        }
        if hash_state_dict_keys(state_dict) == "1f5ab7703c6fc803fdded85ff040c316":
            # Wan-AI/Wan2.2-TI2V-5B
            config = {
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
                "eps": 1e-6,
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
            }
        else:
            config = {}
        return state_dict, config
