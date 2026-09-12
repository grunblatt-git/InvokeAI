"""LightTAE/TAEHV VAE used by Wan 2.1 and Krea-2 workflows."""

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

_WAN21_LATENTS_MEAN = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
_WAN21_LATENTS_STD = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
]


def _conv(in_channels: int, out_channels: int, **kwargs: Any) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, **kwargs)


class _MemBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: nn.Module):
        nn.Module.__init__(self)  # pyright: ignore [reportUnknownMemberType]
        self.conv = nn.Sequential(
            _conv(in_channels * 2, out_channels),
            activation,
            _conv(out_channels, out_channels),
            activation,
            _conv(out_channels, out_channels),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.activation = activation

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(torch.cat([x, memory], dim=1)) + self.skip(x))


class _Clamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class _TemporalPool(nn.Module):
    def __init__(self, channels: int, stride: int):
        nn.Module.__init__(self)  # pyright: ignore [reportUnknownMemberType]
        self.stride = stride
        self.conv = nn.Conv2d(channels * stride, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = x.shape
        return self.conv(x.reshape(-1, self.stride * channels, height, width))


class _TemporalGrow(nn.Module):
    def __init__(self, channels: int, stride: int):
        nn.Module.__init__(self)  # pyright: ignore [reportUnknownMemberType]
        self.stride = stride
        self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = x.shape
        return self.conv(x).reshape(-1, channels, height, width)


def _run_parallel(model: nn.Sequential, x: torch.Tensor, *, decode: bool, patch_size: int) -> torch.Tensor:
    batch, frames, channels, height, width = x.shape
    x = x.reshape(batch * frames, channels, height, width)
    if not decode and patch_size > 1:
        x = F.pixel_unshuffle(x, patch_size)

    for block in model:
        if isinstance(block, _MemBlock):
            frame_count = x.shape[0] // batch
            current = x.reshape(batch, frame_count, x.shape[1], x.shape[2], x.shape[3])
            memory = F.pad(current, (0, 0, 0, 0, 0, 0, 1, 0))[:, :frame_count]
            x = block(x, memory.reshape(x.shape))
        else:
            x = block(x)

    if decode and patch_size > 1:
        x = F.pixel_shuffle(x, patch_size)
    return x.reshape(batch, -1, x.shape[1], x.shape[2], x.shape[3])


class LightTAE(nn.Module):
    """A small Wan 2.1-compatible temporal autoencoder.

    LightTAE uses the same 16-channel, 8x spatial latent geometry as the Krea-2 VAE,
    but has its own sequential encoder/decoder layout and no latent normalization.
    """

    def __init__(self, latent_channels: int = 16):
        nn.Module.__init__(self)  # pyright: ignore [reportUnknownMemberType]
        activation = nn.ReLU(inplace=True)
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            _conv(3, 64),
            activation,
            _TemporalPool(64, 2),
            _conv(64, 64, stride=2, bias=False),
            _MemBlock(64, 64, activation),
            _MemBlock(64, 64, activation),
            _MemBlock(64, 64, activation),
            _TemporalPool(64, 2),
            _conv(64, 64, stride=2, bias=False),
            _MemBlock(64, 64, activation),
            _MemBlock(64, 64, activation),
            _MemBlock(64, 64, activation),
            _TemporalPool(64, 1),
            _conv(64, 64, stride=2, bias=False),
            _MemBlock(64, 64, activation),
            _MemBlock(64, 64, activation),
            _MemBlock(64, 64, activation),
            _conv(64, latent_channels),
        )
        feature_channels = (256, 128, 64, 64)
        self.decoder = nn.Sequential(
            _Clamp(),
            _conv(latent_channels, feature_channels[0]),
            activation,
            _MemBlock(feature_channels[0], feature_channels[0], activation),
            _MemBlock(feature_channels[0], feature_channels[0], activation),
            _MemBlock(feature_channels[0], feature_channels[0], activation),
            nn.Upsample(scale_factor=2),
            _TemporalGrow(feature_channels[0], 1),
            _conv(feature_channels[0], feature_channels[1], bias=False),
            _MemBlock(feature_channels[1], feature_channels[1], activation),
            _MemBlock(feature_channels[1], feature_channels[1], activation),
            _MemBlock(feature_channels[1], feature_channels[1], activation),
            nn.Upsample(scale_factor=2),
            _TemporalGrow(feature_channels[1], 2),
            _conv(feature_channels[1], feature_channels[2], bias=False),
            _MemBlock(feature_channels[2], feature_channels[2], activation),
            _MemBlock(feature_channels[2], feature_channels[2], activation),
            _MemBlock(feature_channels[2], feature_channels[2], activation),
            nn.Upsample(scale_factor=2),
            _TemporalGrow(feature_channels[2], 2),
            _conv(feature_channels[3], feature_channels[3], bias=False),
            activation,
            _conv(feature_channels[3], 3),
        )
        self.config = SimpleNamespace(
            z_dim=latent_channels,
            latents_mean=_WAN21_LATENTS_MEAN,
            latents_std=_WAN21_LATENTS_STD,
        )
        self.frames_to_trim = 3

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def encode(self, sample: torch.Tensor, **_: object) -> SimpleNamespace:
        sample = sample.movedim(2, 1)
        if sample.shape[1] % 4:
            sample = torch.cat([sample, sample[:, -1:].repeat_interleave(4 - sample.shape[1] % 4, dim=1)], dim=1)
        latent = _run_parallel(self.encoder, sample, decode=False, patch_size=1).movedim(2, 1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(self, latent: torch.Tensor, **_: object) -> tuple[torch.Tensor]:
        latent = latent.unsqueeze(0) if latent.ndim == 4 else latent
        if latent.shape[1] != self.latent_channels:
            latent = latent.movedim(1, 2)
        latent = _run_parallel(self.decoder, latent.movedim(2, 1), decode=True, patch_size=1)
        return (latent[:, self.frames_to_trim :].movedim(2, 1),)