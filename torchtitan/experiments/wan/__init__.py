# from .wan_video_1x import WanVideoPipeline
# from .wan_video_vae_1x import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
# from .utils import set_seed, load_training_state, save_training_state


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_mse_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers

from torchtitan.experiments.wan.wan_datasets import build_wan_dataloader
from torchtitan.protocols.train_spec import TrainSpec
from .infra.parallelize import parallelize_wan
from .model.args import WanModelArgs
from .model.autoencoder import AutoEncoderParams
from .model.model import WanModel
from .model.state_dict_adapter import WanStateDictAdapter
from .model.wan_vae import WanVAEParams
from .validate import build_wan_validator

__all__ = [
    "WanModelArgs",
    "WanModel",
    "flux_configs",
    "parallelize_wan",
]


wan_args = {
    # "flux-dev": FluxModelArgs(
    #     in_channels=64,
    #     out_channels=64,
    #     vec_in_dim=768,
    #     context_in_dim=4096,
    #     hidden_size=3072,
    #     mlp_ratio=4.0,
    #     num_heads=24,
    #     depth=19,
    #     depth_single_blocks=38,
    #     axes_dim=(16, 56, 56),
    #     theta=10_000,
    #     qkv_bias=True,
    #     autoencoder_params=AutoEncoderParams(
    #         resolution=256,
    #         in_channels=3,
    #         ch=128,
    #         out_ch=3,
    #         ch_mult=(1, 2, 4, 4),
    #         num_res_blocks=2,
    #         z_channels=16,
    #         scale_factor=0.3611,
    #         shift_factor=0.1159,
    #     ),
    # ),
    # "flux-schnell": FluxModelArgs(
    #     in_channels=64,
    #     out_channels=64,
    #     vec_in_dim=768,
    #     context_in_dim=4096,
    #     hidden_size=3072,
    #     mlp_ratio=4.0,
    #     num_heads=24,
    #     depth=19,
    #     depth_single_blocks=38,
    #     axes_dim=(16, 56, 56),
    #     theta=10_000,
    #     qkv_bias=True,
    #     autoencoder_params=AutoEncoderParams(
    #         resolution=256,
    #         in_channels=3,
    #         ch=128,
    #         out_ch=3,
    #         ch_mult=(1, 2, 4, 4),
    #         num_res_blocks=2,
    #         z_channels=16,
    #         scale_factor=0.3611,
    #         shift_factor=0.1159,
    #     ),
    # ),
    # "flux-debug": FluxModelArgs(
    #     in_channels=64,
    #     out_channels=64,
    #     vec_in_dim=768,
    #     context_in_dim=4096,
    #     hidden_size=3072,
    #     mlp_ratio=4.0,
    #     num_heads=24,
    #     depth=19,
    #     depth_single_blocks=38,
    #     axes_dim=(16, 56, 56),
    #     theta=10_000,
    #     qkv_bias=True,
    #     autoencoder_params=AutoEncoderParams(
    #         resolution=256,
    #         in_channels=3,
    #         ch=128,
    #         out_ch=3,
    #         ch_mult=(1, 2, 4, 4),
    #         num_res_blocks=2,
    #         z_channels=16,
    #         scale_factor=0.3611,
    #         shift_factor=0.1159,
    #     ),
    # ),
    "wan-video": WanModelArgs(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=True,
        autoencoder_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
        wan_video_vae_params=WanVAEParams(vae_type="38", z_dim=48, dim=160),
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=WanModel,
        model_args=wan_args,
        parallelize_fn=parallelize_wan,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_wan_dataloader,
        build_tokenizer_fn=None,
        build_loss_fn=build_mse_loss,
        build_validator_fn=build_wan_validator,
        state_dict_adapter=WanStateDictAdapter,
    )
