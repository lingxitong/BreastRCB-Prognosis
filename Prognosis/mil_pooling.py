"""Lightweight MIL feature aggregators; all accept [N, D] and return [1, H].

Adapted from lingxitong/MIL_BASELINE at e1c5d9fcd771f30f94437e993ecf24c0535df991
(MIT, Copyright (c) 2025 Toby; see the repository's third-party notice).
The original classification heads and classification-only auxiliary objectives
are deliberately excluded. These encoders use the common survival objective.
Top-k, log-mean-exp and GeM are additional generic pooling baselines.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def _positive_int(value, name):
    if int(value) != value or int(value) < 1:
        raise ValueError(f"{name} 必须是正整数")
    return int(value)


def _check_bag(x, in_dim):
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != in_dim:
        raise ValueError(f"MIL 输入必须为非空 [N, {in_dim}]，实际为 {tuple(x.shape)}")


def _initialize(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, (nn.LayerNorm, nn.BatchNorm1d)):
            if layer.weight is not None:
                nn.init.ones_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def _projection(in_dim, hidden_dim, dropout):
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))


class AttentionEncoder(nn.Module):
    """AB_MIL/ab_mil.py: plain tanh attention, distinct from gated ABMIL."""

    def __init__(self, in_dim, hidden_dim=512, dropout=0.25, attention_dim=128):
        super().__init__()
        self.in_dim = _positive_int(in_dim, "in_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        attention_dim = _positive_int(attention_dim, "attention_dim")
        self.feature = _projection(in_dim, hidden_dim, dropout)
        self.attention = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh(),
                                       nn.Linear(attention_dim, 1))
        _initialize(self)

    def forward(self, x):
        _check_bag(x, self.in_dim)
        h = self.feature(x)
        weights = self.attention(h).transpose(0, 1).softmax(dim=-1)
        return weights @ h


class RRTEncoder(AttentionEncoder):
    """RRT_MIL/rrt_mil.py's lightweight residual re-embedding variant.

    This is the reference repository's MLP re-embedding encoder, not the complete
    region-transformer RRT-MIL paper pipeline.
    """

    def __init__(self, in_dim, hidden_dim=512, dropout=0.25, attention_dim=128):
        super().__init__(in_dim, hidden_dim, dropout, attention_dim)
        self.re_embedding = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                           nn.LayerNorm(hidden_dim), nn.GELU(),
                                           nn.Linear(hidden_dim, hidden_dim))
        _initialize(self.re_embedding)

    def forward(self, x):
        _check_bag(x, self.in_dim)
        h = self.feature(x)
        h = h + self.re_embedding(h)
        return self.attention(h).transpose(0, 1).softmax(dim=-1) @ h


class DGEncoder(nn.Module):
    """DG_MIL/dg_mil.py's lightweight projection/BatchNorm/attention encoder."""

    def __init__(self, in_dim, hidden_dim=512, dropout=0.25, attention_dim=128):
        super().__init__()
        self.in_dim = _positive_int(in_dim, "in_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        attention_dim = _positive_int(attention_dim, "attention_dim")
        self.projection = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh(),
                                       nn.Linear(attention_dim, 1))
        _initialize(self)

    def forward(self, x):
        _check_bag(x, self.in_dim)
        h = self.projection(x)
        if self.training and len(h) == 1:
            # A single patch has no within-bag variance; use saved statistics.
            h = F.batch_norm(h, self.norm.running_mean, self.norm.running_var,
                             self.norm.weight, self.norm.bias, training=False,
                             momentum=self.norm.momentum, eps=self.norm.eps)
        else:
            h = self.norm(h)
        h = self.dropout(F.relu(h))
        return self.attention(h).transpose(0, 1).softmax(dim=-1) @ h


class _LowRankResidualPath(nn.Module):
    def __init__(self, in_features, out_features, rank, dropout):
        super().__init__()
        self.scaling = 1.0 / rank
        self.dropout = nn.Dropout(dropout)
        self.down_projection = nn.Parameter(torch.empty(rank, in_features))
        self.up_projection = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.down_projection, a=math.sqrt(5))

    def forward(self, x):
        hidden = F.gelu(F.linear(self.dropout(x), self.down_projection))
        return F.linear(hidden, self.up_projection) * self.scaling


class _ManifoldResidualBlock(nn.Linear):
    """Frozen base weight plus trainable nonlinear low-rank residual."""

    def __init__(self, in_features, out_features, rank, dropout):
        super().__init__(in_features, out_features)
        self.residual_path = _LowRankResidualPath(in_features, out_features, rank, dropout)
        self.weight.requires_grad_(False)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.residual_path(x)


class MRABEncoder(nn.Module):
    """MR_AB_MIL/mr_ab_mil.py: manifold-residual gated attention encoder.

    The base affine weights stay frozen. Zero-initialized up projections mean
    the low-rank down projections legitimately get zero gradient on step one.
    The pooled original features are mapped to the common H-dimensional space.
    """

    def __init__(self, in_dim, hidden_dim=512, dropout=0.25, attention_dim=128, rank=32):
        super().__init__()
        self.in_dim = _positive_int(in_dim, "in_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        attention_dim = _positive_int(attention_dim, "attention_dim")
        rank = _positive_int(rank, "rank")
        self.attention_a = nn.Sequential(
            _ManifoldResidualBlock(in_dim, attention_dim, rank, dropout),
            nn.Tanh(), nn.Dropout(dropout))
        self.attention_b = nn.Sequential(
            _ManifoldResidualBlock(in_dim, attention_dim, rank, dropout),
            nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(attention_dim, 1)
        self.output_projection = nn.Linear(in_dim, hidden_dim)
        # Do not reinitialize/fine-tune the frozen base maps or zero up paths.
        _initialize(self.attention_c)
        _initialize(self.output_projection)

    def forward(self, x):
        _check_bag(x, self.in_dim)
        scores = self.attention_c(self.attention_a(x) * self.attention_b(x))
        feature = scores.transpose(0, 1).softmax(dim=-1) @ x
        return self.output_projection(feature)


class NNEncoder(nn.Module):
    """NN_MIL/nn_mil.py: random feature-subspace attention.

    Train uses one random subspace. Evaluation averages fixed, deterministic
    covering subspaces (stride equals subspace width, reducing overlap/work).
    Classification heads, entropy reports and class-balanced training are not
    part of this survival feature encoder.
    """

    def __init__(self, in_dim, hidden_dim=512, dropout=0.25, attention_dim=128,
                 subspace_dim=128):
        super().__init__()
        self.in_dim = _positive_int(in_dim, "in_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        attention_dim = _positive_int(attention_dim, "attention_dim")
        self.subspace_dim = min(_positive_int(subspace_dim, "subspace_dim"), self.in_dim)
        self.V = nn.Linear(in_dim, attention_dim)
        self.U = nn.Linear(in_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(in_dim, hidden_dim)
        generator = torch.Generator(device="cpu").manual_seed(42)
        self.register_buffer("cover_permutation", torch.randperm(in_dim, generator=generator))
        _initialize(self)

    def _attention(self, x, indices):
        subset = x.index_select(-1, indices)
        a = torch.tanh(F.linear(subset, self.V.weight.index_select(1, indices), self.V.bias))
        b = torch.sigmoid(F.linear(subset, self.U.weight.index_select(1, indices), self.U.bias))
        return self.w(self.dropout(a) * self.dropout(b)).transpose(0, 1).softmax(dim=-1)

    def _cover_indices(self):
        keep = self.subspace_dim
        starts = list(range(0, self.in_dim - keep + 1, keep))
        if starts[-1] != self.in_dim - keep:
            starts.append(self.in_dim - keep)
        return [self.cover_permutation[start:start + keep] for start in starts]

    def forward(self, x):
        _check_bag(x, self.in_dim)
        if self.training:
            indices = torch.randperm(self.in_dim, device=x.device)[:self.subspace_dim]
            pooled = self._attention(x, indices) @ x
        else:
            pooled = torch.stack([self._attention(x, indices) @ x
                                  for indices in self._cover_indices()]).mean(dim=0)
        return self.output_projection(pooled)


class ACMILTokenEncoder(nn.Module):
    """AC_MIL/ac_mil.py's multi-token QK attention -> bag representation.

    The reference bag feature depends on token Q/K scores and the bag V/O maps.
    Unused per-token value/output paths, sub-classifiers, masking curriculum and
    diversity/sub-classifier objectives are excluded, rather than retained as
    parameters that cannot learn from the common survival loss.
    """

    def __init__(self, in_dim, hidden_dim=512, dropout=0.25, heads=4, tokens=5):
        super().__init__()
        self.in_dim = _positive_int(in_dim, "in_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.heads = _positive_int(heads, "heads")
        self.tokens = _positive_int(tokens, "tokens")
        if hidden_dim % heads:
            raise ValueError("ACMIL hidden_dim 必须能被 heads 整除")
        self.feature = _projection(in_dim, hidden_dim, dropout)
        self.queries = nn.Parameter(torch.empty(tokens, hidden_dim))
        self.q_projections = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(tokens)])
        self.k_projections = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(tokens)])
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        _initialize(self)
        nn.init.normal_(self.queries, std=1e-6)

    def forward(self, x):
        _check_bag(x, self.in_dim)
        h = self.feature(x)
        n, channels = h.shape
        width = channels // self.heads
        token_attentions = []
        for index in range(self.tokens):
            query = self.q_projections[index](self.queries[index]).reshape(self.heads, 1, width)
            keys = self.k_projections[index](h).reshape(n, self.heads, width).transpose(0, 1)
            logits = (query @ keys.transpose(-1, -2)) / math.sqrt(width)
            token_attentions.append(logits.softmax(dim=-1))
        attention = torch.stack(token_attentions).mean(dim=0)
        values = self.value_projection(h).reshape(n, self.heads, width).transpose(0, 1)
        pooled = (attention @ values).transpose(0, 1).reshape(1, channels)
        return self.norm(self.dropout(self.output_projection(pooled)))


class TopKEncoder(nn.Module):
    """Generic per-channel top-k mean pooling, not a separate paper claim."""

    def __init__(self, in_dim, hidden_dim=512, dropout=0.25, topk=8):
        super().__init__()
        self.in_dim = _positive_int(in_dim, "in_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.topk = _positive_int(topk, "topk")
        self.feature = _projection(in_dim, hidden_dim, dropout)
        _initialize(self)

    def forward(self, x):
        _check_bag(x, self.in_dim)
        h = self.feature(x)
        return h.topk(min(self.topk, len(h)), dim=0).values.mean(dim=0, keepdim=True)


class LogSumExpEncoder(nn.Module):
    """Generic temperature-scaled log-mean-exp pooling (count normalized)."""

    def __init__(self, in_dim, hidden_dim=512, dropout=0.25, temperature=1.0):
        super().__init__()
        self.in_dim = _positive_int(in_dim, "in_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("LSE temperature 必须为有限正数")
        self.temperature = float(temperature)
        self.feature = _projection(in_dim, hidden_dim, dropout)
        _initialize(self)

    def forward(self, x):
        _check_bag(x, self.in_dim)
        h = self.feature(x)
        return self.temperature * (torch.logsumexp(h / self.temperature, dim=0, keepdim=True)
                                   - math.log(len(h)))


class GeMEncoder(nn.Module):
    """Generic nonnegative power-mean pooling with learned p > 1 in log space."""

    def __init__(self, in_dim, hidden_dim=512, dropout=0.25, p=3.0):
        super().__init__()
        self.in_dim = _positive_int(in_dim, "in_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        if not math.isfinite(p) or p <= 1:
            raise ValueError("GeM 初始 p 必须为有限且大于 1 的数")
        self.raw_p = nn.Parameter(torch.tensor(math.log(math.expm1(p - 1)) if p < 20 else p - 1))
        self.feature = _projection(in_dim, hidden_dim, dropout)
        _initialize(self)

    @property
    def p(self):
        return 1 + F.softplus(self.raw_p)

    def forward(self, x):
        _check_bag(x, self.in_dim)
        h = self.feature(x).clamp_min(1e-6)
        p = self.p
        log_mean = (torch.logsumexp(p * h.log(), dim=0, keepdim=True) - math.log(len(h))) / p
        return log_mean.exp()
