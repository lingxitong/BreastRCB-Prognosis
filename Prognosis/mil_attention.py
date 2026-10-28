"""Pure-PyTorch context, low-rank and linear-attention MIL encoders.

Adapted from lingxitong/MIL_BASELINE at commit
e1c5d9fcd771f30f94437e993ecf24c0535df991 (MIT, Copyright (c) 2025 Toby):
  modules/CA_MIL/ca_mil.py
  modules/ILRA_MIL/ilra_mil.py
  modules/LIN_MIL/lin_mil.py

The LIN source additionally credits charlotterchtr/Lin-MIL (MIT).
Classification heads are omitted: each encoder exposes [N, D] -> [1, H]
for the shared prognosis scorer and optional clinical fusion. LIN uses native
reshape/expand instead of einops; stochastic depth is disabled (Identity).
See THIRD_PARTY_NOTICES.md for the retained license and adaptation scope.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_dimensions(in_dim, hidden_dim):
    if in_dim < 1 or hidden_dim < 1:
        raise ValueError("in_dim 和 hidden_dim 必须为正数。")


def _validate_heads(hidden_dim, heads):
    if heads < 1 or hidden_dim % heads:
        raise ValueError("hidden_dim 必须能被 heads 整除，且 heads 必须为正数。")


def _validate_bag(x, in_dim):
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != in_dim:
        raise ValueError(f"MIL 输入必须为非空 [N, {in_dim}]，实际为 {tuple(x.shape)}。")


def _initialize(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


class ContextAttentionEncoder(nn.Module):
    """CA-MIL context QKV -> context attention pooling -> H-dimensional readout.

    Scaled dot-product attention avoids explicitly retaining an N-by-N matrix
    when a fused backend is available; its compute remains quadratic in N.
    """

    def __init__(self, in_dim, hidden_dim=512, dropout=0.25, attention_dim=128):
        super().__init__()
        _validate_dimensions(in_dim, hidden_dim)
        if attention_dim < 1:
            raise ValueError("attention_dim 必须为正数。")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.feature = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                     nn.Dropout(dropout))
        self.query = nn.Linear(hidden_dim, attention_dim)
        self.key = nn.Linear(hidden_dim, attention_dim)
        self.value = nn.Linear(hidden_dim, attention_dim)
        self.attention = nn.Sequential(nn.Linear(attention_dim, attention_dim),
                                       nn.Tanh(), nn.Linear(attention_dim, 1))
        self.output = nn.Sequential(nn.Linear(attention_dim, hidden_dim), nn.ReLU())
        _initialize(self)

    def forward(self, x):
        _validate_bag(x, self.in_dim)
        features = self.feature(x)
        query = self.query(features).unsqueeze(0).unsqueeze(0)
        key = self.key(features).unsqueeze(0).unsqueeze(0)
        value = self.value(features).unsqueeze(0).unsqueeze(0)
        context = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)[0, 0]
        weights = F.softmax(self.attention(context).transpose(0, 1), dim=-1)
        return self.output(weights @ context)


class _ILRAAttention(nn.Module):
    """Projected residual multi-head attention with optional multiplicative gate."""

    def __init__(self, query_dim, key_dim, out_dim, heads, layer_norm=False, gated=False):
        super().__init__()
        self.fc_q = nn.Linear(query_dim, out_dim)
        self.fc_k = nn.Linear(key_dim, out_dim)
        self.fc_v = nn.Linear(key_dim, out_dim)
        self.multihead_attn = nn.MultiheadAttention(out_dim, heads, batch_first=True)
        self.fc_o = nn.Linear(out_dim, out_dim)
        self.norm0 = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.norm1 = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.gate = nn.Sequential(nn.Linear(query_dim, out_dim), nn.SiLU()) if gated else None

    def forward(self, query, key):
        raw_query = query
        projected_query = self.fc_q(query)
        attended, _ = self.multihead_attn(projected_query, self.fc_k(key), self.fc_v(key),
                                         need_weights=False)
        result = self.norm0(projected_query + attended)
        result = self.norm1(result + F.relu(self.fc_o(result)))
        if self.gate is not None:
            result = result * self.gate(raw_query)
        return result


class _ILRAGAB(nn.Module):
    """Low-rank gather-to-latents then project-back block, including both gates."""

    def __init__(self, in_dim, hidden_dim, heads, inducing_points, layer_norm):
        super().__init__()
        self.latent = nn.Parameter(torch.empty(1, inducing_points, hidden_dim))
        nn.init.xavier_uniform_(self.latent)
        self.project_forward = _ILRAAttention(hidden_dim, in_dim, hidden_dim, heads,
                                              layer_norm=layer_norm, gated=True)
        self.project_backward = _ILRAAttention(in_dim, hidden_dim, hidden_dim, heads,
                                               layer_norm=layer_norm, gated=True)

    def forward(self, x):
        latent = self.latent.expand(x.shape[0], -1, -1)
        compressed = self.project_forward(latent, x)
        return self.project_backward(x, compressed)


class ILRAEncoder(nn.Module):
    """ILRA GAB stack and learned-seed non-local pooling, without classifier.

    The first GAB maps raw in_dim features directly to hidden_dim, as in the
    reference implementation. A single non-local seed gives one bag vector.
    Input dropout is the standardized prognosis dropout; the low-rank attention
    and residual/gating equations are retained from the reference.
    """

    def __init__(self, in_dim, hidden_dim=512, dropout=0.25, heads=4, depth=2,
                 inducing_points=16, layer_norm=False):
        super().__init__()
        _validate_dimensions(in_dim, hidden_dim)
        _validate_heads(hidden_dim, heads)
        if depth < 1 or inducing_points < 1:
            raise ValueError("ILRA depth 和 inducing_points 必须为正数。")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.gab_blocks = nn.ModuleList([
            _ILRAGAB(in_dim if index == 0 else hidden_dim, hidden_dim, heads,
                      inducing_points, layer_norm)
            for index in range(depth)
        ])
        self.pool_seed = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.pool_seed)
        self.pooling = _ILRAAttention(hidden_dim, hidden_dim, hidden_dim, heads,
                                      layer_norm=layer_norm)
        _initialize(self)

    def forward(self, x):
        _validate_bag(x, self.in_dim)
        x = self.input_dropout(x).unsqueeze(0)
        for block in self.gab_blocks:
            x = block(x)
        seed = self.pool_seed.expand(x.shape[0], -1, -1)
        return self.pooling(seed, x).squeeze(1)


class _LinearAttention(nn.Module):
    """Reference Lin-MIL ReLU-kernel attention with learned head temperature."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.full((heads, 1, 1), (dim // heads) ** -0.5))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, instances, channels = x.shape
        qkv = self.qkv(x).reshape(batch, instances, 3, self.heads, channels // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = F.relu(query), F.relu(key)
        denominator = (query @ key.sum(dim=-2, keepdim=True).transpose(-2, -1)).clamp_min(1e2)
        channel_attention = (key.transpose(-2, -1) @ value) * self.temperature
        attended = (query @ channel_attention) / denominator
        attended = attended.transpose(1, 2).reshape(batch, instances, channels)
        return self.proj(attended)


class _LinearTransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = _LinearAttention(dim, heads)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        x = x + self.drop_path(self.attention(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class LinearAttentionEncoder(nn.Module):
    """Lin-MIL projection, pre-norm residual blocks and normalized CLS readout."""

    def __init__(self, in_dim, hidden_dim=512, dropout=0.25, heads=4, depth=2,
                 mlp_ratio=4.0):
        super().__init__()
        _validate_dimensions(in_dim, hidden_dim)
        _validate_heads(hidden_dim, heads)
        if depth < 1 or mlp_ratio <= 0 or int(hidden_dim * mlp_ratio) < 1:
            raise ValueError("LIN depth 和 MLP 维度必须为正数。")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.projection = nn.Linear(in_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.embedding_dropout = nn.Dropout(dropout)
        self.transformer = nn.ModuleList([
            _LinearTransformerBlock(hidden_dim, heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        _validate_bag(x, self.in_dim)
        x = self.projection(x).unsqueeze(0)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.embedding_dropout(x)
        for block in self.transformer:
            x = block(x)
        return self.norm(x[:, 0])
