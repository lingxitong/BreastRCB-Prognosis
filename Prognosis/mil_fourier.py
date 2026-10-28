"""FOURIER_MIL feature encoder adapted from MIL_BASELINE@e1c5d9.

Copyright (c) 2025 Toby. Source modifications: remove classification head,
support configurable hidden width and dtype-safe FFT temporary tensors.
MIT notice: ../third_party/MIL_BASELINE_LICENSE.txt. No positional coordinates
are inferred; this model uses the stored patch order like its reference.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class APFF(nn.Module):
    def __init__(
        self,
        dim,
        num_blocks=8,
        mode="AP",
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        hidden_size_factor=1,
        drop=0.0,
    ):
        super().__init__()
        if dim % num_blocks != 0:
            raise ValueError(f"hidden_dim={dim} must be divisible by num_blocks={num_blocks}")

        self.mode = mode
        self.hidden_size = dim
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        hidden = self.block_size * self.hidden_size_factor
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, hidden))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, hidden))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, hidden, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        dtype = x.dtype
        x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        b, n, c = x.shape

        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(b, n // 2 + 1, self.num_blocks, self.block_size)

        hidden = self.block_size * self.hidden_size_factor
        o1_real = x.real.new_zeros(b, n // 2 + 1, self.num_blocks, hidden)
        o1_imag = torch.zeros_like(o1_real)
        o2_real = torch.zeros_like(x.real)
        o2_imag = torch.zeros_like(x.real)

        total_modes = n // 2 + 1
        kept_modes = max(1, int(total_modes * self.hard_thresholding_fraction))

        o1_real[:, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:, :kept_modes].real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", x[:, :kept_modes].imag, self.w1[1])
            + self.b1[0]
        )
        o1_imag[:, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:, :kept_modes].imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", x[:, :kept_modes].real, self.w1[1])
            + self.b1[1]
        )

        o1_real = self.drop(o1_real)
        o1_imag = self.drop(o1_imag)

        o2_real[:, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_real[:, :kept_modes], self.w2[0])
            - torch.einsum("...bi,bio->...bo", o1_imag[:, :kept_modes], self.w2[1])
            + self.b2[0]
        )
        o2_imag[:, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_imag[:, :kept_modes], self.w2[0])
            + torch.einsum("...bi,bio->...bo", o1_real[:, :kept_modes], self.w2[1])
            + self.b2[1]
        )

        x = torch.stack([self.drop(o2_real), self.drop(o2_imag)], dim=-1)
        if self.mode == "LP":
            x = F.softshrink(x, lambd=self.sparsity_threshold)
        elif self.mode == "HP":
            x = x - F.softshrink(x, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(x)
        x = x.reshape(b, n // 2 + 1, c)
        x = torch.fft.irfft(x, n=n, dim=1, norm="ortho")
        return x.type(dtype)


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=1.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class FourierBlock(nn.Module):
    def __init__(
        self,
        dim=512,
        num_blocks=8,
        mode="AP",
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        hidden_size_factor=1,
        mlp_ratio=1.0,
        drop=0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = APFF(
            dim=dim,
            num_blocks=num_blocks,
            mode=mode,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
            hidden_size_factor=hidden_size_factor,
            drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.mlp(self.norm2(x))
        return x



class FourierEncoder(nn.Module):
    """Adaptive Fourier filtering blocks followed by normalized CLS pooling."""

    def __init__(self, in_dim, hidden_dim=512, dropout=0.25, heads=4, depth=2):
        super().__init__()
        if depth < 1 or heads < 1 or hidden_dim % heads:
            raise ValueError("Fourier: depth/heads must be positive and heads must divide hidden_dim")
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 1e-6)
        self.layers = nn.ModuleList([
            FourierBlock(dim=hidden_dim, num_blocks=heads, drop=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.apply(initialize_weights)

    def forward(self, x):
        if x.ndim != 2 or x.shape[0] < 1:
            raise ValueError("Fourier expects a non-empty [N, D] bag")
        h = self.projection(x).unsqueeze(0)
        side = math.ceil(math.sqrt(x.shape[0]))
        padding = side * side - x.shape[0]
        if padding:
            h = torch.cat((h, h[:, :padding]), dim=1)
        h = torch.cat((self.cls_token.expand(1, -1, -1), h), dim=1)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)[:, 0]
