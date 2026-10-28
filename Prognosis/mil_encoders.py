"""MIL registry and reproducible patch budgeting for survival encoders.

Reference: lingxitong/MIL_BASELINE@e1c5d9fcd771f30f94437e993ecf24c0535df991.
Only feature aggregators are adapted, not the original classification objectives.
See THIRD_PARTY_NOTICES.md for sources, exclusions, and licenses.
"""

import math

import torch
import torch.nn as nn

from mil_attention import ContextAttentionEncoder, ILRAEncoder, LinearAttentionEncoder
from mil_fourier import FourierEncoder
from mil_pooling import (
    AttentionEncoder, RRTEncoder, DGEncoder, MRABEncoder, NNEncoder,
    ACMILTokenEncoder, TopKEncoder, LogSumExpEncoder, GeMEncoder,
)


MODEL_REGISTRY = {
    "abmil": {"display": "Gated ABMIL (existing)", "permutation_invariant": True,
              "source": "Prognosis/main_prognosis.py", "note": "原有门控注意力"},
    "mean_mil": {"display": "Mean MIL (existing)", "permutation_invariant": True,
                 "source": "Prognosis/main_prognosis.py", "note": "原有均值池化"},
    "max_mil": {"display": "Max MIL (existing)", "permutation_invariant": True,
                "source": "Prognosis/main_prognosis.py", "note": "原有最大值池化"},
    "attention_mil": {"display": "Plain attention MIL", "permutation_invariant": True,
                      "source": "modules/AB_MIL/ab_mil.py", "note": "无门控 tanh 注意力"},
    "rrt_mil": {"display": "RRT lightweight encoder", "permutation_invariant": True,
                "source": "modules/RRT_MIL/rrt_mil.py", "note": "参考库的残差重嵌入版本"},
    "dg_mil": {"display": "DG lightweight encoder", "permutation_invariant": True,
               "source": "modules/DG_MIL/dg_mil.py", "note": "投影、BatchNorm 与注意力"},
    "ca_mil": {"display": "Context-aware attention encoder", "permutation_invariant": True,
               "source": "modules/CA_MIL/ca_mil.py", "note": "上下文 QKV 后注意力聚合；建议限制 patch 数"},
    "ilra_mil": {"display": "ILRA feature encoder", "permutation_invariant": True,
                 "source": "modules/ILRA_MIL/ilra_mil.py", "note": "低秩潜变量交互与非局部池化"},
    "lin_mil": {"display": "Lin-MIL feature encoder", "permutation_invariant": True,
                "source": "modules/LIN_MIL/lin_mil.py", "note": "ReLU kernel 线性注意力"},
    "fourier_mil": {"display": "Fourier-MIL feature encoder", "permutation_invariant": False,
                    "source": "modules/FOURIER_MIL/fourier_mil.py", "note": "频域过滤与 CLS；依赖存储的 patch 顺序"},
    "mr_abmil": {"display": "MR-ABMIL feature encoder", "permutation_invariant": True,
                 "source": "modules/MR_AB_MIL/mr_ab_mil.py", "note": "冻结仿射权重与低秩非线性残差"},
    "nn_mil": {"display": "nnMIL subspace encoder", "permutation_invariant": True,
               "source": "modules/NN_MIL/nn_mil.py", "note": "训练随机特征子空间；推理固定覆盖"},
    "acmil": {"display": "ACMIL multi-token encoder", "permutation_invariant": True,
              "source": "modules/AC_MIL/ac_mil.py", "note": "多 token 聚合适配，不含分类或多样性辅助损失"},
    "topk_mil": {"display": "Top-k pooling MIL", "permutation_invariant": True,
                 "source": "local generic pooling baseline", "note": "逐特征通道 top-k 均值"},
    "lse_mil": {"display": "LogSumExp pooling MIL", "permutation_invariant": True,
                "source": "local generic pooling baseline", "note": "温度控制的 log-mean-exp"},
    "gem_mil": {"display": "GeM pooling MIL", "permutation_invariant": True,
                "source": "local generic pooling baseline", "note": "可学习幂的广义均值池化"},
}

MODEL_ALIASES = {
    "ab_mil": "attention_mil", "gate_ab_mil": "abmil", "gated_abmil": "abmil",
    "plain_abmil": "attention_mil", "mr_ab_mil": "mr_abmil", "ac_mil": "acmil",
    "top_k_mil": "topk_mil", "logsumexp_mil": "lse_mil",
}

MIL_HPARAM_DEFAULTS = {
    "mil_max_instances": 4096,
    "mil_attention_dim": 128,
    "mil_heads": 4,
    "mil_depth": 2,
    "mil_inducing_points": 16,
    "mil_tokens": 5,
    "mil_rank": 32,
    "mil_subspace_dim": 128,
    "mil_topk": 8,
    "mil_lse_temperature": 1.0,
    "mil_gem_p": 3.0,
}


def canonical_model_name(value):
    name = str(value).strip().lower().replace("-", "_")
    name = MODEL_ALIASES.get(name, name)
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知 MIL 模型 {value!r}，可选: {', '.join(MODEL_REGISTRY)}")
    return name


def validate_mil_config(cfg):
    name = canonical_model_name(cfg.get("model_type", "abmil"))
    for key, default in MIL_HPARAM_DEFAULTS.items():
        value = cfg.get(key, default)
        if isinstance(default, int):
            minimum = 0 if key == "mil_max_instances" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} 必须是 >= {minimum} 的整数。")
        elif not math.isfinite(value) or value <= (1 if key == "mil_gem_p" else 0):
            raise ValueError(f"{key} 必须为有限正数（GeM 初始 p > 1）。")
    if name in {"ilra_mil", "lin_mil", "fourier_mil", "acmil"}:
        heads = cfg.get("mil_heads", MIL_HPARAM_DEFAULTS["mil_heads"])
        if cfg["hidden_dim"] % heads:
            raise ValueError(f"{name}: hidden_dim 必须能被 mil_heads={heads} 整除。")
    return name


class MILInstanceLimiter(nn.Module):
    """Same patch budget for every model; eval uses deterministic stored-order coverage.

    This operates on already loaded features, so it bounds model compute and
    activations, not H5 decoding RAM. No spatial coordinates are assumed.
    """

    def __init__(self, encoder, max_instances):
        super().__init__()
        if max_instances < 0:
            raise ValueError("max_instances 必须 >= 0。")
        self.encoder = encoder
        self.hidden_dim = encoder.hidden_dim
        self.max_instances = int(max_instances)

    def select_indices(self, n, device):
        if n < 1:
            raise ValueError("MIL bag 至少需要一个 patch。")
        if self.max_instances == 0 or n <= self.max_instances:
            return torch.arange(n, device=device)
        if self.training:
            return torch.randperm(n, device=device)[:self.max_instances].sort().values
        return torch.linspace(0, n - 1, self.max_instances, device=device).round().long()

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError("MIL 输入应为 [N_patch, feature_dim]。")
        indices = self.select_indices(x.shape[0], x.device)
        return self.encoder(x.index_select(0, indices))


def build_extra_encoder(cfg):
    name = validate_mil_config(cfg)
    params = {key: cfg.get(key, value) for key, value in MIL_HPARAM_DEFAULTS.items()}
    common = dict(in_dim=cfg["in_dim"], hidden_dim=cfg["hidden_dim"], dropout=cfg["drop_out"])
    att = params["mil_attention_dim"]
    if name == "attention_mil":
        return AttentionEncoder(**common, attention_dim=att)
    if name == "rrt_mil":
        return RRTEncoder(**common, attention_dim=att)
    if name == "dg_mil":
        return DGEncoder(**common, attention_dim=att)
    if name == "mr_abmil":
        return MRABEncoder(**common, attention_dim=att, rank=params["mil_rank"])
    if name == "nn_mil":
        return NNEncoder(**common, attention_dim=att, subspace_dim=params["mil_subspace_dim"])
    if name == "acmil":
        return ACMILTokenEncoder(**common, heads=params["mil_heads"], tokens=params["mil_tokens"])
    if name == "topk_mil":
        return TopKEncoder(**common, topk=params["mil_topk"])
    if name == "lse_mil":
        return LogSumExpEncoder(**common, temperature=params["mil_lse_temperature"])
    if name == "gem_mil":
        return GeMEncoder(**common, p=params["mil_gem_p"])
    if name == "ca_mil":
        return ContextAttentionEncoder(**common, attention_dim=att)
    if name == "ilra_mil":
        return ILRAEncoder(**common, heads=params["mil_heads"], depth=params["mil_depth"],
                           inducing_points=params["mil_inducing_points"])
    if name == "lin_mil":
        return LinearAttentionEncoder(**common, heads=params["mil_heads"], depth=params["mil_depth"])
    if name == "fourier_mil":
        return FourierEncoder(**common, heads=params["mil_heads"], depth=params["mil_depth"])
    raise ValueError(f"{name} 是原有模型，应由 main_prognosis.build_encoder 构建。")


def print_model_list():
    print("MIL encoders: 16 canonical models (3 existing + 13 new; aliases are not counted)")
    for name, info in MODEL_REGISTRY.items():
        print(f"  {name:16s} {info['display']} — {info['note']}")
