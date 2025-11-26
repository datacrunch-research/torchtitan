import torch 
from torch import nn
import torch.nn.functional as F
import torch.amp as amp

from typing import Optional, List

from einops import rearrange
from icecream import ic

import logging
logger = logging.getLogger(__name__)

__all__ = ["Wan2_2_VAE"]

CACHE_T = 2


def count_conv3d(model: nn.Module):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count

class Resample(nn.Module):
    def __init__(
        self,
        dim: int, 
        mode: str,
    ):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d"
        )
        super().__init__()
        self.dim = dim 
        self.mode = mode 

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(), # TODO
                nn.Conv2d(
                    in_channels=dim, 
                    out_channels=dim,
                    kernel_size=3,
                    padding=1
                )
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(), #TODO
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    padding=1
                )
            )
            self.time_conv = CausalConv3d(
                in_channels=dim, 
                out_channels=dim*2,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0)
            )
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    stride=(2, 2)
                )
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    stride=(2, 2)
                )
            )
            self.time_conv = CausalConv3d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1), 
                padding=(0, 0, 0)
            )
        else:
            self.resample = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        feat_cache=None,
        feat_idx: List[int] = [0]
    ) -> torch.Tensor:
        b, c, t, h, w = x.shape

        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    # creating the cache of the current -2 elements in the temporal dimension
                    if (cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep"):
                        # cache the last frame of last two chunks
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if feat_cache[idx] == "Rep":
                        ic(x.shape)
                        x = self.time_conv(x)
                        ic(x.shape, "after time_conv")
                    else:
                        ic(x.shape)
                        ic(feat_cache[idx].shape)
                        x = self.time_conv(x, feat_cache[idx])
                        ic(x.shape, "after time_conv w/ feat_cache[idx]")
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack(
                        (x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                        dim=3,
                    )
                    x = x.reshape(b, c, t * 2, h, w)
                    ic(x.shape)
        
        t = x.shape[2]
        ic(x.shape)
        x = rearrange(x, "b c t h w -> (b t) c h w")
        ic(x.shape)
        x = self.resample(x) # TODO: what does this do?
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    input = torch.cat([feat_cache[idx][:, :, -1:, :, :], x], dim=2)
                    x = self.time_conv(input)
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x 

class Upsample(nn.Upsample):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return

class Resample_(nn.Module):

    def __init__(self, dim, mode):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
                # nn.Conv2d(dim, dim//2, 3, padding=1)
            )
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if (cache_x.shape[2] < 2 and feat_cache[idx] is not None and
                            feat_cache[idx] != "Rep"):
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                    cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (cache_x.shape[2] < 2 and feat_cache[idx] is not None and
                            feat_cache[idx] == "Rep"):
                        cache_x = torch.cat(
                            [
                                torch.zeros_like(cache_x).to(cache_x.device),
                                cache_x
                            ],
                            dim=2,
                        )
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight.detach().clone()
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight = nn.Parameter(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data.detach().clone()
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight = nn.Parameter(conv_weight)
        nn.init.zeros_(conv.bias.data)




class CausalConv3d(nn.Conv3d):
    """
    Causal 3D convolution that extends nn.Conv3d.
    
    Key difference from regular Conv3d:
    - Regular Conv3d: Can look at past AND future frames (non-causal)
    - CausalConv3d: Only looks at past frames (causal) - ensures temporal causality
    
    How it works:
    1. For temporal dimension (dim=2): Only pads on the left (past), never on the right (future)
    2. For spatial dimensions: Pads symmetrically (left and right) like normal conv
    3. Supports cache_x: Can use cached features from previous frames for efficiency
    
    Example:
        Input shape: [B, C, T, H, W] where T is temporal (frames)
        With kernel_size=3, padding=1:
        - Regular Conv3d: Frame t uses frames [t-1, t, t+1] (non-causal)
        - CausalConv3d: Frame t only uses frames [t-2, t-1, t] (causal)
    """

    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Store original padding for temporal dimension
        # Convert (T, H, W) padding to (W_left, W_right, H_left, H_right, T_left, T_right)
        self._padding = (
            self.padding[2],  # W left
            self.padding[2],  # W right
            self.padding[1],  # H left
            self.padding[1],  # H right
            2 * self.padding[0],  # T left (causal: only pad past frames)
            0,  # T right (never pad future frames - this is the key difference!)
        )
        # Set padding to zero so Conv3d doesn't pad, we'll do it manually
        self.padding = (0, 0, 0)
    
    def forward(self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, T, H, W]
            cache_x: Optional cached features from previous frames [B, C, T_cache, H, W]
                    Used to avoid recomputing features for frames we've already seen
        """
        padding = list(self._padding)
        # If we have cached features, concatenate them before padding
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)  # Concatenate along temporal dimension
            padding[4] -= cache_x.shape[2]  # Reduce padding needed since we added cache
        # Apply padding: (W_left, W_right, H_left, H_right, T_left, T_right)
        x = F.pad(x, padding)
        logger.info(f"x.shape: {x.shape}")
        x = super().forward(x)
        logger.info(f"x.shape: {x.shape}")
        return x


def patchify(
    x: torch.Tensor,
    patch_size: int = 2,
) -> torch.Tensor:
    """
    Patchify input tensor by grouping spatial dimensions into patches.
    
    For 4D input [B, C, H, W]: 
        Output: [B, C*patch_size*patch_size, H//patch_size, W//patch_size]
    For 5D input [B, C, T, H, W]:
        Output: [B, C*patch_size*patch_size, T, H//patch_size, W//patch_size]
    
    Uses pure torch operations (unfold + reshape) for better performance than einops.
    """
    ic(x.dim())
    if patch_size == 1:
        return x
    
    if x.dim() == 4:
        # Input x.shape: [B, C, H, W]
        assert x.shape[2] % patch_size == 0 and x.shape[3] % patch_size == 0, "H and W must be divisible by patch_size"
        # Output x.shape: [B, C*patch_size*patch_size, H//patch_size, W//patch_size]
        b, c, h, w = x.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        x = x.unfold(2, patch_size, patch_size) 
        logger.info(f"x.shape: {x.shape}")
        x = x.unfold(3, patch_size, patch_size)
        logger.info(f"x.shape: {x.shape}")

        # x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5)
        logger.info(f"x.shape: {x.shape}")
        x = x.contiguous()
        logger.info(f"x.is_contiguous: {x.is_contiguous()}")
        x = x.reshape(b, c * patch_size * patch_size, h_patches, w_patches)
        logger.info(f"x.shape: {x.shape}")
        return x
    elif x.dim() == 5:
        # Original code
        # x = rearrange(x, "b c f (h q) (w r) -> b (c r q) f h w", q=patch_size, r=patch_size)
        assert x.shape[3] % patch_size == 0 and x.shape[4] % patch_size == 0
        b, c, t, h, w = x.shape
        h_patches = h // patch_size
        w_patches = w // patch_size 
        ic(x.shape)
        ic(h_patches)
        ic(w_patches)

        x = x.unfold(dimension=3, size=patch_size, step=patch_size)
        ic(x.shape)
        x = x.unfold(dimension=4, size=patch_size, step=patch_size)
        ic(x.shape)

        x = x.permute(0, 2, 3, 4, 1, 5, 6)
        ic(x.shape)
        ic(x.is_contiguous())
        x = x.contiguous()
        ic(x.is_contiguous())
        
        x = x.reshape(b, c * patch_size * patch_size, t, h_patches, w_patches)
        ic(x.shape)
    else:
        raise NotImplementedError(f"Invalid input shape: {x.shape}, expected 4D tensor")
    return x 
    if x.dim() == 4:
        # Input: [B, C, H, W] where H and W are divisible by patch_size
        # Output: [B, C*patch_size*patch_size, H//patch_size, W//patch_size]
        b, c, h, w = x.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        
        # Use unfold to extract patches: [B, C, H, W] -> [B, C, H//patch_size, W//patch_size, patch_size, patch_size]
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # x is now [B, C, h_patches, w_patches, patch_size, patch_size]
        
        # Rearrange: [B, C, h_patches, w_patches, patch_size, patch_size] 
        #         -> [B, h_patches, w_patches, C, patch_size, patch_size]
        #         -> [B, C*patch_size*patch_size, h_patches, w_patches]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.reshape(b, c * patch_size * patch_size, h_patches, w_patches)
        
    elif x.dim() == 5:
        # Input: [B, C, F, H, W] where H and W are divisible by patch_size
        # Output: [B, C*patch_size*patch_size, F, H//patch_size, W//patch_size]
        b, c, f, h, w = x.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        
        # Use unfold on spatial dimensions: [B, C, F, H, W] -> [B, C, F, H//patch_size, W//patch_size, patch_size, patch_size]
        x = x.unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
        # x is now [B, C, F, h_patches, w_patches, patch_size, patch_size]
        
        # Rearrange: [B, C, F, h_patches, w_patches, patch_size, patch_size]
        #         -> [B, F, h_patches, w_patches, C, patch_size, patch_size]
        #         -> [B, C*patch_size*patch_size, F, h_patches, w_patches]
        x = x.permute(0, 2, 3, 4, 1, 5, 6).contiguous()
        x = x.reshape(b, c * patch_size * patch_size, f, h_patches, w_patches)
        
    else:
        raise ValueError(f"Invalid input shape: {x.shape}, expected 4D or 5D tensor")
    
    return x


def unpatchify(
    x: torch.Tensor,
    patch_size: int = 2,
) -> torch.Tensor:
    """
    Unpatchify input tensor by restoring spatial dimensions from patches.
    
    For 4D input [B, C*patch_size*patch_size, H, W]:
        Output: [B, C, H*patch_size, W*patch_size]
    For 5D input [B, C*patch_size*patch_size, F, H, W]:
        Output: [B, C, F, H*patch_size, W*patch_size]
    
    Uses pure torch operations (reshape + permute) for better performance than einops.
    """
    if patch_size == 1:
        return x
    
    if x.dim() == 4:
        # Input: [B, C*patch_size*patch_size, h_patches, w_patches]
        # Output: [B, C, h_patches*patch_size, w_patches*patch_size]
        b, c_patches, h_patches, w_patches = x.shape
        c = c_patches // (patch_size * patch_size)
        h = h_patches * patch_size
        w = w_patches * patch_size
        
        # Reshape: [B, C*patch_size*patch_size, h_patches, w_patches]
        #       -> [B, h_patches, w_patches, C, patch_size, patch_size]
        x = x.reshape(b, c, patch_size, patch_size, h_patches, w_patches)
        
        # Rearrange: [B, C, patch_size, patch_size, h_patches, w_patches]
        #         -> [B, C, h_patches, patch_size, w_patches, patch_size]
        #         -> [B, C, h_patches*patch_size, w_patches*patch_size]
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.reshape(b, c, h, w)
        
    elif x.dim() == 5:
        # Input: [B, C*patch_size*patch_size, F, h_patches, w_patches]
        # Output: [B, C, F, h_patches*patch_size, w_patches*patch_size]
        b, c_patches, f, h_patches, w_patches = x.shape
        c = c_patches // (patch_size * patch_size)
        h = h_patches * patch_size
        w = w_patches * patch_size
        
        # Reshape: [B, C*patch_size*patch_size, F, h_patches, w_patches]
        #       -> [B, F, h_patches, w_patches, C, patch_size, patch_size]
        x = x.reshape(b, c, patch_size, patch_size, f, h_patches, w_patches)
        
        # Rearrange: [B, C, patch_size, patch_size, F, h_patches, w_patches]
        #         -> [B, C, F, h_patches, patch_size, w_patches, patch_size]
        #         -> [B, C, F, h_patches*patch_size, w_patches*patch_size]
        x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
        x = x.reshape(b, c, f, h, w)
        
    else:
        raise ValueError(f"Invalid input shape: {x.shape}, expected 4D or 5D tensor")
    
    return x


class AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels
        logger.info(f"group_size: {self.group_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        ic(x.shape, "before pad")
        x = F.pad(x, pad)
        ic(x.shape, "after pad")

        b, c, t, h, w = x.shape
        t_ = t // self.factor_t
        h_ = h // self.factor_s
        w_ = w // self.factor_s

        # b, c, t, h, w -> b, c, t_, ft, h_, fs, w_, fs -> b, c, ft, fs, fs, t_, h_, w_ 
        x = x.view(b, c, t_, self.factor_t, h_, self.factor_s, w_, self.factor_s)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(b, c * self.factor, t_, h_, w_)
        x = x.view(b, self.out_channels, self.group_size, t_, h_, w_)
        x = x.mean(dim=2)
        return x
        
    

class Down_ResidualBlock(nn.Module):
    """
    Residual Downsample Block for the VAE Encoder
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temporal_downsample: bool,
        down_flag: bool,
    ):
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_channels=in_dim,
            out_channels=out_dim,
            factor_t=2 if temporal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(
                ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                )
            )
            in_dim = out_dim
        
        if down_flag:
            mode = "downsample3d" if temporal_downsample else "downsample2d"
            logger.info(f"mode final downsample: {mode}")
            downsamples.append(Resample(out_dim, mode=mode))
        
        self.downsamples = nn.Sequential(*downsamples)
        logger.info(f"downsamples: {self.downsamples}")


    
    def forward(self, x: torch.Tensor, feat_cache=None, feat_idx=[0]) -> torch.Tensor:
        x_copy = x.clone()
        for module in self.downsamples:
            ic(module)
            ic(x.shape)
            x = module(x, feat_cache, feat_idx)
            ic(x.shape)
        avg_shortcut = self.avg_shortcut(x_copy)
        ic(x.shape, avg_shortcut.shape)
        x = x + avg_shortcut
        return x

class RMS_norm(nn.Module):
    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias
        )

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),  # Normalize input features
            nn.SiLU(), 
            CausalConv3d(in_dim, out_dim, 3, padding=1),  # First conv (causal in time)
            RMS_norm(out_dim, images=False),  # Normalize after first conv
            nn.SiLU(),  # Second activation
            nn.Dropout(dropout),  # Regularization
            CausalConv3d(out_dim, out_dim, 3, padding=1),  # Second conv
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1) 
            if in_dim != out_dim else nn.Identity()
        )

    def forward(
        self, 
        x: torch.Tensor,
        feat_cache = None, 
        feat_idx = [0]
    ) -> torch.Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device)
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                ic(x.shape)
                ic(feat_cache[idx].shape)
                ic(feat_idx[0])
            else:
                x = layer(x)
        ic(x.shape, h.shape)
        x = x + h
        ic(x.shape)
        return x 

class AttentionBlock(nn.Module):
    """
    Single-head Causal Self-Attention
    """
    def __init__(self, dim:int):
        super().__init__()
        self.dim = dim 

        # layers of the attention block
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim*3,
            kernel_size=1
        )
        self.proj = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1 
        )

        nn.init.zeros_(self.proj.weight)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        identity = x
        ic(x.shape,  "Causal Attn FWD")
        b, c, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)

        # compute query, key, value
        q, k, v = (
            self.to_qkv(x)
                .reshape(b * t, 1, c * 3, -1)
                .permute(0, 1, 3, 2)
                .contiguous()
                .chunk(3, dim=-1)
        )

        # TODO: why not enable the causal here in the scaled_dot_product_attention??
        x = F.scaled_dot_product_attention(query=q, key=k, value=v)
        ic(x.shape)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b*t, c, h, w)

        # output
        x = self.proj(x)
        ic(f"x after proj {x.shape}")
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        ic(x.shape, identity.shape)
        x = x + identity
        return x

class Encoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[int] = [],
        temporal_downsample: List[bool] = [True, True, False],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_downsample = temporal_downsample
        self.dropout = dropout

        dims = [dim * u for u in [1] + dim_mult]
        logger.info(f"dims: {dims}")

        scale = 1.0
        self.conv1 = CausalConv3d(
            in_channels=12,
            out_channels=dims[0],
            kernel_size=3,
            padding=1,
        )

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            ic(f"i: {i}, in_dim: {in_dim}, out_dim: {out_dim}")
            t_down_flag = (
                temporal_downsample[i]
                if i < len(temporal_downsample) else False
            )
            logger.info(f"t_down_flag: {t_down_flag}")
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temporal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(
                in_dim=out_dim,
                out_dim=out_dim,
                dropout=dropout
            ),
            AttentionBlock(dim=out_dim),
            ResidualBlock(
                in_dim=out_dim,
                out_dim=out_dim,
                dropout=dropout
            )
        )

        self.head = nn.Sequential(
            RMS_norm(dim=out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(
                in_channels=out_dim,
                out_channels=z_dim,
                kernel_size=3,
                padding=1
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache = None,
        feat_idx = [0]
    ) -> torch.Tensor:

        ic(x.shape, "Encoder3d")
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:,:,-CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                ic(cache_x.shape[2])
                # There are previous elements in the cache
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x
                    ],
                    dim=2,
                )
            ic(x.shape)
            x = self.conv1(x, feat_cache[idx])
            ic(x.shape, "after conv1")
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        ic(x.shape)
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        
        ic(x.shape)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        ic(x.shape)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :,  :].unsqueeze(2).to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                ic(x.shape, feat_cache[idx].shape, feat_idx[0])
            else:
                x = layer(x)
        ic(x.shape)
        return x




class Decoder3d(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self,):
        pass


class WanVAE_(nn.Module):
    def __init__(
        self,
        dim: int = 160, 
        dec_dim: int = 256,
        z_dim: int = 16,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[int] = [],
        temporal_downsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.dec_dim = dec_dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_downsample = temporal_downsample
        self.temporal_upsample = temporal_downsample[::-1]

        self.encoder = Encoder3d(
            dim=dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temporal_downsample=self.temporal_downsample,
            dropout=dropout,
        )
        self.conv1 = CausalConv3d(
            in_channels=z_dim * 2, 
            out_channels=z_dim * 2, 
            kernel_size=1
        )
        self.conv2 = CausalConv3d(
            in_channels=z_dim, 
            out_channels=z_dim, 
            kernel_size=1
        )
        # self.decoder = Decoder3d(
        #     dim=dec_dim,
        #     z_dim=z_dim,
        #     dim_mult=dim_mult,
        #     num_res_blocks=num_res_blocks,
        #     attn_scales=attn_scales,
        #     temporal_upsample=self.temporal_upsample,
        #     dropout=dropout,
        # )
    
    def forward(
        self,
        x: torch.Tensor,
        scale: List[float] = [0., 1.],
    ):
        mu = self.encode(x, scale)
        x_recon = self.decode(mu, scale)
        return x_recon, mu
    
    def encode(
        self,
        x: torch.Tensor,
        scale: List[float] = [0., 1.],
    ) -> torch.Tensor:
        self.clear_cache() # TODO
        ic(x.shape)
        x = patchify(x, patch_size=2)
        ic(x.shape)
        t = x.shape[2]
        iter_  = 1 + (t - 1) // 4
        ic(iter_)
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0: 
                ic("First frame")
                ic(x[:,:,:1,:,:].shape)
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], dim=2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu 

        
    
    def clear_cache(self):
        # TODO
        # This method gets called before each encode/decode call
        # self._conv_num = count_conv3d(self.decoder)
        # self._conv_idx = [0]
        # self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(
    pretrained_path: Optional[str] = None,
    z_dim: int = 16,
    dim: int = 160,
    dim_mult: List[int] = [1, 2, 4, 4],
    temporal_downsample: List[bool] = [False, True, True],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_scales: List[int] = [],
    num_res_blocks: int = 2,
    dropout: float = 0.0,
):
    # cfg = dict(
    #     dim=dim,
    #     z_dim=z_dim,
    #     dim_mult=dim_mult,
    #     temporal_downsample=temporal_downsample,
    #     attn_scales=attn_scales,
    #     num_res_blocks=num_res_blocks,
    #     dropout=dropout,
    # )
    # init the model using meta device
    with torch.device("meta"):
        model = WanVAE_(
            dim=dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            temporal_downsample=temporal_downsample,
            attn_scales=attn_scales,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )
    
    # Load the checkpoint from pretrained_path
    logger.info(f"Loading checkpoint from {pretrained_path}")
    chckpt_state_dict = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(chckpt_state_dict, assign=True, strict=False)
    return model


class Wan2_2_VAE:
    def __init__(
        self,
        z_dim=48,
        c_dim=160,
        vae_pth=None,
        dim_mult=[1, 2, 4, 4],
        temporal_downsample=[False, True, True],
        dtype=torch.float, # TODO: bfloat16 maybe?
        device="cuda",
    ):
        self.dtype = dtype
        self.device = device 

        mean = torch.tensor(
            [
                -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557, 
                -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825, 
                -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502, 
                -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230, 
                -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
                0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
            ],
            device=self.device,
            dtype=self.dtype
        )
        std = torch.tensor(
            [
                0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
                0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
                0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
                0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 
                0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887, 
                0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
            ],
            device=self.device,
            dtype=self.dtype
        )

        self.scale = [mean, 1.0 / std]

        self.model = (
            _video_vae(
                pretrained_path=vae_pth,
                z_dim=z_dim,
                dim=c_dim,
                dim_mult=dim_mult,
                temporal_downsample=temporal_downsample,
            ).eval().requires_grad_(False).to(device)
        )
    
    # Original implementation
    def encode(self, videos: List[torch.Tensor]) -> List[torch.Tensor]:
        ic(len(videos))
        try: 
            if not isinstance(videos, list):
                raise TypeError("videos should be a list")
            with amp.autocast('cuda', dtype=self.dtype):
                return [
                    self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                    for u in videos
                ]
        except TypeError as e:
            logger.error(e)
            return None
    
    def decode(self, zs: List[torch.Tensor]) -> List[torch.Tensor]:
        try:
            if not isinstance(zs, list):
                raise TypeError("zs shoulf be a list")
            with amp.autocast('cuda', dtype=self.dtype):
                return [
                    self.model.decode(
                        u.unsqueeze(0), self.scale
                    ).float().clamp_(-1., 1.).squeeze(0)
                    for u in zs
                ]
        except TypeError as e:
            logger.error(e)
            return None

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype_ = torch.bfloat16
    vae = Wan2_2_VAE(
        vae_pth="assets/hf/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        dtype=torch.bfloat16, device=device
    )
    from torchtitan.experiments.wan.model.dataset import RawVideoDataset
    from torch.utils.data import DataLoader
    dataset = RawVideoDataset(
        data_dir="./dataset/world_model_raw_data/val_v2.0_raw/",
        downsampled=1,
    )
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    batch = next(iter(dataloader))
    video = batch[0].permute(0, 4, 1, 2, 3)
    ic(video.shape)
    ic(video.dtype)
    ic(video.device)
    ic(video.min(), video.max())
    video = ((video / 255.0) * 2.) - 1.  
    video = video.to(device=device, dtype=dtype_)
    ic(video.min(), video.max()) 
    # ic(batch[0].permute(0,1,4,2,3).shape)
    # exit()

    img = torch.rand((1, 3, 77, 512, 512)).to(device=device, dtype=torch.bfloat16)
    ic(vae)
    ic(img.shape)
    ic(video.shape)
    latent = vae.encode(list(video))
    ic(latent[0].shape)
    assert latent[0].shape == torch.Size((48, 20, 32, 32))

    
    # ic(latent.shape)