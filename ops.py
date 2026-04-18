"""
ops.py — DCGAN layer primitives (PyTorch)

Faithful port of the original TF1.x ops.py.
Design decisions preserved from the blueprint:
  - LeakyReLU(0.2) used in BOTH Generator and Discriminator hidden layers
  - Batch norm with epsilon=1e-5, momentum=0.1 (PyTorch convention; TF momentum=0.9 = 1-0.1)
  - Weights init: truncated_normal(stddev=0.02) for conv, normal(stddev=0.02) for deconv/linear
  - No BN on Discriminator's first layer (DCGAN rule)
  - No BN on Generator's final layer
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Weight initialisation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _truncated_normal_init(tensor: torch.Tensor, stddev: float = 0.02) -> torch.Tensor:
    """Mimic TF truncated_normal_initializer: clamp at ±2σ."""
    nn.init.normal_(tensor, 0.0, stddev)
    with torch.no_grad():
        tensor.clamp_(-2 * stddev, 2 * stddev)
    return tensor


def _normal_init(tensor: torch.Tensor, stddev: float = 0.02) -> torch.Tensor:
    nn.init.normal_(tensor, 0.0, stddev)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Leaky ReLU (slope=0.2, used everywhere)
# ─────────────────────────────────────────────────────────────────────────────


def lrelu(leak: float = 0.2) -> nn.LeakyReLU:
    """Leaky ReLU with slope 0.2 — used in BOTH G and D hidden layers."""
    return nn.LeakyReLU(leak, inplace=True)


# ─────────────────────────────────────────────────────────────────────────────
# Batch normalisation (epsilon=1e-5, momentum matches TF decay=0.9)
# ─────────────────────────────────────────────────────────────────────────────


def batch_norm(num_features: int) -> nn.BatchNorm2d:
    """
    BatchNorm2d with epsilon=1e-5, momentum=0.1 (PyTorch uses 1-TF_momentum).
    TF: decay=0.9 → PyTorch: momentum=0.1
    affine=True → learnable γ/β (equivalent to scale=True in TF).
    """
    return nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True)


# ─────────────────────────────────────────────────────────────────────────────
# Convolution (Discriminator)
# ─────────────────────────────────────────────────────────────────────────────


def conv2d(
    in_channels: int,
    out_channels: int,
    k: int = 5,
    stride: int = 2,
    stddev: float = 0.02,
) -> nn.Conv2d:
    """
    5×5 stride-2 SAME-padded convolution used in Discriminator.
    padding=2 gives SAME behaviour for kernel=5, stride=2.
    init: truncated_normal(stddev=0.02), bias=0.
    """
    layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=k,
        stride=stride,
        padding=k // 2,
        bias=True,
    )
    _truncated_normal_init(layer.weight, stddev)
    nn.init.zeros_(layer.bias)
    return layer


# ─────────────────────────────────────────────────────────────────────────────
# Transposed convolution (Generator)
# ─────────────────────────────────────────────────────────────────────────────


def deconv2d(
    in_channels: int,
    out_channels: int,
    k: int = 5,
    stride: int = 2,
    stddev: float = 0.02,
) -> nn.ConvTranspose2d:
    """
    5×5 stride-2 transposed convolution used in Generator for upsampling.
    padding=2, output_padding=1 replicates TF conv2d_transpose SAME behaviour.
    init: random_normal(stddev=0.02) (not truncated — same as blueprint).
    """
    layer = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=k,
        stride=stride,
        padding=k // 2,
        output_padding=stride - 1,
        bias=True,
    )
    _normal_init(layer.weight, stddev)
    nn.init.zeros_(layer.bias)
    return layer


# ─────────────────────────────────────────────────────────────────────────────
# Linear layer
# ─────────────────────────────────────────────────────────────────────────────


def linear(in_features: int, out_features: int, stddev: float = 0.02) -> nn.Linear:
    """
    Fully connected layer.
    Used in: Generator first layer (z → 8192), Discriminator last layer (8192 → 1).
    init: random_normal(stddev=0.02), bias=0.
    """
    layer = nn.Linear(in_features, out_features, bias=True)
    _normal_init(layer.weight, stddev)
    nn.init.zeros_(layer.bias)
    return layer


# ─────────────────────────────────────────────────────────────────────────────
# Helper: spatial size after stride-2 SAME conv (ceil division)
# ─────────────────────────────────────────────────────────────────────────────


def conv_out_size_same(size: int, stride: int = 2) -> int:
    return math.ceil(size / stride)
