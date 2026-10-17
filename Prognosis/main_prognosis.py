#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified: stable log-domain RCB/NLL, survival-head initialization and training diagnostics.
"""
淋巴结融合预后（RCB 风格公式）
================================
在 main 分支的离散时间生存训练流程上，把输入改成：

  * 一张原发肿瘤切片特征
  * 肿瘤长径 tumor_size
  * 一张淋巴结切片特征（无转移时可空）
  * 转移淋巴结个数 n_metastatic_lymph_nodes

预后指数沿用乳腺 RCB 的两项相加结构，系数改为可训练：

  R = a * (tumor_size_norm * tumor_slide_score) ^ b
    + c * (ln_count_norm * lymph_slide_score) ^ d

转移淋巴结个数为 0 时，第二项强制为 0。
a、c 通过 softplus 保持为正；b、d 通过 sigmoid 限制在 (0, 1)，
初值对齐经典 RCB 的 1.4 与 0.17。

切片评分由各自的 MIL 编码器从 WSI 特征中得到，范围 (0, 1)，
对应 RCB 里的浸润比例 / 转移负荷，而不是手工估算。
R 再经单调读出头变成离散时间 hazard，用 NLL 生存损失训练。
c-index 直接用 R 排序（越大风险越高），因此排序只由该公式决定。
"""

from __future__ import print_function

import argparse
import json
import math
import os
import random
import warnings

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold


def _builtin_cindex(censorships, event_times, risk_scores):
    """Harrell's concordance index（含删失）的纯 numpy 实现。"""
    c = np.asarray(censorships)
    t = np.asarray(event_times, dtype=float)
    r = np.asarray(risk_scores, dtype=float)
    n = len(t)
    num, den = 0.0, 0.0
    for i in range(n):
        if c[i] == 0:
            for j in range(n):
                if t[j] > t[i]:
                    den += 1.0
                    if r[i] > r[j]:
                        num += 1.0
                    elif r[i] == r[j]:
                        num += 0.5
    return num / den if den > 0 else float("nan")


try:
    from sksurv.metrics import concordance_index_censored

    def _cindex_impl(censorships, event_times, risk_scores):
        event = (1 - np.asarray(censorships)).astype(bool)
        return concordance_index_censored(
            event, np.asarray(event_times), np.asarray(risk_scores), tied_tol=1e-8
        )[0]
except Exception:  # pragma: no cover
    try:
        from lifelines.utils import concordance_index

        def _cindex_impl(censorships, event_times, risk_scores):
            event = 1 - np.asarray(censorships)
            return concordance_index(
                np.asarray(event_times), -np.asarray(risk_scores), event_observed=event
            )
    except Exception:
        _cindex_impl = _builtin_cindex


def concordance(censorships, event_times, risk_scores):
    event = 1 - np.asarray(censorships)
    if np.sum(event) == 0:
        return float("nan")
    try:
        return float(_cindex_impl(censorships, event_times, risk_scores))
    except Exception:
        return float("nan")


HPARAM_KEYS = [
    "target", "k", "split_mode", "n_bins", "n_classes", "max_epochs", "lr", "reg",
    "drop_out", "gc", "alpha_surv", "seed", "opt", "model_type", "in_dim",
    "hidden_dim", "feat_key", "num_workers",
    "tumor_feat_col", "ln_feat_col", "tumor_size_col", "ln_count_col",
    "score_norm", "grad_clip", "numerics_version",
]

TARGET_COLS = {
    "OS": ("OS_month", "OS_status"),
    "DFS": ("DFS_month", "DFS_status"),
    "RFS": ("RFS_month", "RFS_status"),
}

# 经典 RCB（Symmans 2007）一次项系数与指数，用作 a/b/c/d 的初值
RCB_INIT_SCALE = 1.4
RCB_INIT_EXP = 0.17
NUMERICS_VERSION = 2
_EMPTY_TOKENS = {"", "nan", "none", "null", "na"}


def _inv_softplus(y):
    y = float(y)
    return math.log(math.expm1(y)) if y < 20 else y


def _inv_sigmoid(y):
    y = min(max(float(y), 1e-4), 1.0 - 1e-4)
    return math.log(y / (1.0 - y))


def nll_surv_loss(hazards, S, Y, c, alpha=0.0, eps=1e-7, logits=None):
    """离散时间 NLL；训练使用 logits 路径，避免概率下溢和 clamp 断梯度。"""
    batch_size = len(Y)
    Y = Y.view(batch_size, 1).long()
    c = c.view(batch_size, 1).float()
    if logits is not None:
        log_h = F.logsigmoid(logits)
        log_survival = torch.cumsum(F.logsigmoid(-logits), dim=1)
        log_survival_padded = F.pad(log_survival, (1, 0), value=0.0)
        uncensored_loss = -(1 - c) * (
            torch.gather(log_survival_padded, 1, Y)
            + torch.gather(log_h, 1, Y)
        )
        censored_loss = -c * torch.gather(log_survival_padded, 1, Y + 1)
        return (uncensored_loss + (1 - alpha) * censored_loss).mean()
    # 保留概率输入接口；正式训练不经过这一兼容路径。
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    return loss.mean()


def _init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class ABMILEncoder(nn.Module):
    """Gated-Attention MIL，只输出切片级表征 [1, hidden]。"""

    def __init__(self, in_dim, dropout=0.25, hidden=512, att_dim=256):
        super().__init__()
        self.hidden_dim = hidden
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden, att_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden, att_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(att_dim, 1)
        self.apply(_init_weights)

    def forward(self, x):
        h = self.fc(x)
        A = self.attention_w(self.attention_V(h) * self.attention_U(h))
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        return torch.mm(A, h)


class MeanMaxEncoder(nn.Module):
    def __init__(self, in_dim, dropout=0.25, hidden=512, pool="mean"):
        super().__init__()
        self.hidden_dim = hidden
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.pool = pool
        self.apply(_init_weights)

    def forward(self, x):
        h = self.fc(x)
        if self.pool == "mean":
            return h.mean(dim=0, keepdim=True)
        return h.max(dim=0, keepdim=True)[0]


class SlideScorer(nn.Module):
    """MIL 表征 -> (0, 1) 切片评分，对应 RCB 中的比例/负荷项。"""

    def __init__(self, encoder, score_norm="layernorm"):
        super().__init__()
        self.encoder = encoder
        if score_norm not in ("layernorm", "none"):
            raise ValueError(f"未知 score_norm: {score_norm}")
        self.score_norm = score_norm
        self.score_head = nn.Linear(encoder.hidden_dim, 1)
        nn.init.normal_(self.score_head.weight, std=0.01)
        self.score_head.bias.data.zero_()

    def forward_logits(self, x):
        pooled = self.encoder(x)
        if self.score_norm == "layernorm":
            pooled = F.layer_norm(pooled, (self.encoder.hidden_dim,))
        return self.score_head(pooled)

    def forward(self, x):
        return torch.sigmoid(self.forward_logits(x))


def build_encoder(cfg):
    mt = cfg["model_type"]
    if mt == "abmil":
        return ABMILEncoder(cfg["in_dim"], dropout=cfg["drop_out"], hidden=cfg["hidden_dim"])
    if mt == "mean_mil":
        return MeanMaxEncoder(cfg["in_dim"], dropout=cfg["drop_out"], hidden=cfg["hidden_dim"], pool="mean")
    if mt == "max_mil":
        return MeanMaxEncoder(cfg["in_dim"], dropout=cfg["drop_out"], hidden=cfg["hidden_dim"], pool="max")
    raise NotImplementedError(
        f"淋巴结融合分支仅支持 model_type=abmil/mean_mil/max_mil，收到: {mt}"
    )


class RCBIndex(nn.Module):
    """
    R = a * (size_norm * tumor_score) ^ b
      + c * (count_norm * lymph_score) ^ d     （无转移时第二项为 0）
    """

    def __init__(self):
        super().__init__()
        self.a_raw = nn.Parameter(torch.tensor(_inv_softplus(RCB_INIT_SCALE)))
        self.b_raw = nn.Parameter(torch.tensor(_inv_sigmoid(RCB_INIT_EXP)))
        self.c_raw = nn.Parameter(torch.tensor(_inv_softplus(RCB_INIT_SCALE)))
        self.d_raw = nn.Parameter(torch.tensor(_inv_sigmoid(RCB_INIT_EXP)))

    def coeff_values(self):
        a = F.softplus(self.a_raw)
        b = torch.sigmoid(self.b_raw)
        c = F.softplus(self.c_raw)
        d = torch.sigmoid(self.d_raw)
        return a, b, c, d

    def coeff_dict(self):
        a, b, c, d = self.coeff_values()
        return {
            "a": float(a.detach().cpu()),
            "b": float(b.detach().cpu()),
            "c": float(c.detach().cpu()),
            "d": float(d.detach().cpu()),
        }

    def forward(self, tumor_score, tumor_size_norm, lymph_score, ln_count_norm, has_met,
                tumor_log_score=None, lymph_log_score=None):
        a, b, c, d = self.coeff_values()
        # 不对 size * score 硬截断；logsigmoid 在 sigmoid 已下溢时仍保留梯度。
        log_score_t = (torch.log(tumor_score) if tumor_log_score is None
                       else tumor_log_score).view(1, 1)
        log_base_t = torch.log(tumor_size_norm.view(1, 1)) + log_score_t
        tumor_term = a * torch.exp(b * log_base_t)

        if float(has_met.detach().view(-1)[0]) > 0:
            log_score_n = (torch.log(lymph_score) if lymph_log_score is None
                           else lymph_log_score).view(1, 1)
            log_base_n = torch.log(ln_count_norm.view(1, 1)) + log_score_n
            lymph_term = c * torch.exp(d * log_base_n)
        else:
            lymph_term = tumor_term.new_zeros((1, 1))

        rcb = tumor_term + lymph_term
        return rcb, tumor_term, lymph_term


class RCBPrognosisModel(nn.Module):
    """双路切片评分 + RCB 公式 + 单调离散时间读出。"""

    def __init__(self, tumor_encoder, node_encoder, n_classes, score_norm="layernorm"):
        super().__init__()
        self.tumor_scorer = SlideScorer(tumor_encoder, score_norm)
        self.node_scorer = SlideScorer(node_encoder, score_norm)
        self.rcb = RCBIndex()
        # softplus 后为正，保证 R 越大、每个时间箱的 hazard 越高
        self.hazard_scale_raw = nn.Parameter(torch.tensor(_inv_softplus(0.1)))
        self.bin_bias = nn.Parameter(torch.full((n_classes,), _inv_sigmoid(0.1)))

    def formula_coeffs(self):
        coeffs = self.rcb.coeff_dict()
        coeffs["hazard_scale"] = float(F.softplus(self.hazard_scale_raw).detach().cpu())
        return coeffs

    def forward(self, tumor_feats, node_feats, tumor_size_norm, ln_count_norm, has_met):
        tumor_logit = self.tumor_scorer.forward_logits(tumor_feats)
        tumor_score = torch.sigmoid(tumor_logit)
        tumor_log_score = F.logsigmoid(tumor_logit)
        if float(has_met.detach().view(-1)[0]) > 0:
            lymph_logit = self.node_scorer.forward_logits(node_feats)
            lymph_score = torch.sigmoid(lymph_logit)
            lymph_log_score = F.logsigmoid(lymph_logit)
        else:
            lymph_score = tumor_score.new_zeros((1, 1))
            lymph_logit = tumor_logit.new_full((1, 1), float("nan"))
            lymph_log_score = tumor_logit.new_full((1, 1), float("nan"))

        rcb, tumor_term, lymph_term = self.rcb(
            tumor_score, tumor_size_norm, lymph_score, ln_count_norm, has_met,
            tumor_log_score=tumor_log_score, lymph_log_score=lymph_log_score,
        )
        scale = F.softplus(self.hazard_scale_raw)
        logits = rcb * scale + self.bin_bias.view(1, -1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        parts = {
            "rcb": rcb,
            "tumor_term": tumor_term,
            "lymph_term": lymph_term,
            "tumor_score": tumor_score,
            "lymph_score": lymph_score,
            "tumor_logit": tumor_logit,
            "lymph_logit": lymph_logit,
            "tumor_log_score": tumor_log_score,
            "lymph_log_score": lymph_log_score,
            "logits": logits,
        }
        return hazards, S, parts


def build_model(cfg, device):
    tumor_encoder = build_encoder(cfg)
    node_encoder = build_encoder(cfg)
    model = RCBPrognosisModel(tumor_encoder, node_encoder, cfg["n_classes"],
                              score_norm=cfg.get("score_norm", "layernorm"))
    return model.to(device)


def formula_param_ids(model):
    ids = {id(p) for p in model.rcb.parameters()}
    ids.add(id(model.hazard_scale_raw))
    ids.add(id(model.bin_bias))
    return ids


def detect_in_dim(feat_paths, feat_key):
    for p in feat_paths:
        if isinstance(p, str) and os.path.isfile(p):
            with h5py.File(p, "r") as f:
                key = feat_key if feat_key in f else list(f.keys())[0]
                return int(f[key].shape[-1])
    raise FileNotFoundError("未能找到任何可用的 h5 特征文件以推断特征维度。")


def load_features(path, feat_key):
    with h5py.File(path, "r") as f:
        key = feat_key if feat_key in f else list(f.keys())[0]
        feats = f[key][:]
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError(f"特征形状应为 [N_patch, dim]，实际 {feats.shape}: {path}")
    if not feats.shape[0] or not feats.shape[1] or not np.isfinite(feats).all():
        raise ValueError(f"特征为空或含 NaN/Inf: {path}")
    return feats


def _clean_path(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.lower() in _EMPTY_TOKENS:
        return ""
    return text


def discretize_time(pt, n_bins):
    eps = 1e-6
    uncensored = pt[pt["censorship"] == 0]
    src = uncensored if len(uncensored) >= n_bins else pt
    try:
        _, q_bins = pd.qcut(src["event_time"], q=n_bins, retbins=True,
                            labels=False, duplicates="drop")
    except Exception:
        q_bins = np.linspace(pt["event_time"].min(), pt["event_time"].max(), n_bins + 1)
    q_bins = np.asarray(q_bins, dtype=float)
    q_bins[0] = pt["event_time"].min() - eps
    q_bins[-1] = pt["event_time"].max() + eps
    q_bins = np.unique(q_bins)
    disc = pd.cut(pt["event_time"], bins=q_bins, labels=False,
                  right=False, include_lowest=True)
    disc = disc.fillna(0).astype(int).values
    n_classes = len(q_bins) - 1
    return disc, n_classes


def build_patient_table(df, target, n_bins, cfg):
    """每个 case 一行：原发灶路径、长径、淋巴结路径、转移个数、生存标签。"""
    month_col, status_col = TARGET_COLS[target]
    tumor_col = cfg["tumor_feat_col"]
    ln_col = cfg["ln_feat_col"]
    size_col = cfg["tumor_size_col"]
    count_col = cfg["ln_count_col"]
    required = ["case_id", tumor_col, size_col, count_col, month_col, status_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少列: {missing}")
    if ln_col not in df.columns:
        raise KeyError(f"CSV 缺少淋巴结特征列: {ln_col}")

    df = df.copy()
    df[month_col] = pd.to_numeric(df[month_col], errors="coerce")
    df[status_col] = pd.to_numeric(df[status_col], errors="coerce")
    df[size_col] = pd.to_numeric(df[size_col], errors="coerce")
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce")

    records = []
    problems = []
    n_dup = 0
    for case_id, g in df.groupby("case_id", sort=False):
        if len(g) > 1:
            n_dup += 1
        row = g.iloc[0]
        event_time = row[month_col]
        status = row[status_col]
        tumor_size = row[size_col]
        n_met = row[count_col]
        tumor_path = _clean_path(row[tumor_col])
        ln_path = _clean_path(row[ln_col]) if ln_col in g.columns else ""
        if pd.isna(event_time) or pd.isna(status) or not tumor_path:
            problems.append(f"{case_id}: 生存标签或原发灶路径缺失")
            continue
        if not np.isfinite(event_time) or float(event_time) < 0 or status not in (0, 1):
            problems.append(f"{case_id}: 时间须为有限非负数，status 须为 0/1")
            continue
        if not np.isfinite(tumor_size) or float(tumor_size) <= 0:
            problems.append(f"{case_id}: tumor_size 必须为正数，实际 {tumor_size}")
            continue
        if not np.isfinite(n_met) or float(n_met) < 0:
            problems.append(f"{case_id}: 转移淋巴结个数无效 {n_met}")
            continue
        n_met = float(n_met)
        has_met = n_met > 0
        if has_met and not ln_path:
            problems.append(f"{case_id}: 转移淋巴结个数为 {n_met}，但淋巴结切片路径为空")
            continue
        if not has_met:
            ln_path = ""
        records.append({
            "case_id": case_id,
            "event_time": float(event_time),
            "censorship": 1 - int(round(float(status))),
            "tumor_slide_feat_path": tumor_path,
            "lymph_node_slide_feat_path": ln_path,
            "tumor_size": float(tumor_size),
            "n_met": n_met,
            "has_met": int(has_met),
        })
    if n_dup:
        print(f"警告: {n_dup} 个 case_id 有多行，仅使用每个患者的第一行"
              f"（本任务约定一张原发灶 + 一张淋巴结）。")
    if problems:
        preview = "\n".join(problems[:12])
        extra = "" if len(problems) <= 12 else f"\n... 另有 {len(problems) - 12} 条"
        raise ValueError(f"有 {len(problems)} 个患者数据不合法:\n{preview}{extra}")
    if not records:
        raise ValueError("没有可用患者。")

    pt = pd.DataFrame(records).reset_index(drop=True)
    disc, n_classes = discretize_time(pt, n_bins)
    pt["disc_label"] = disc
    return pt, n_classes


class InputScaler:
    """训练集上的尺度。除以常数不改变 a*(x)^b 这一函数族的表达能力，只避免数值过大。"""

    def __init__(self):
        self.tumor_size_scale = 1.0
        self.ln_count_scale = 1.0

    def fit(self, pt):
        size_mean = float(pt["tumor_size"].mean())
        self.tumor_size_scale = size_mean if size_mean > 1e-6 else 1.0
        positive = pt.loc[pt["n_met"] > 0, "n_met"]
        if len(positive):
            count_mean = float(positive.mean())
            self.ln_count_scale = count_mean if count_mean > 1e-6 else 1.0
        else:
            self.ln_count_scale = 1.0
        return self

    def to_dict(self):
        return {
            "tumor_size_scale": self.tumor_size_scale,
            "ln_count_scale": self.ln_count_scale,
        }

    @classmethod
    def from_dict(cls, data):
        scaler = cls()
        scaler.tumor_size_scale = float(data["tumor_size_scale"])
        scaler.ln_count_scale = float(data["ln_count_scale"])
        return scaler


def save_scaler(scaler, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scaler.to_dict(), f, ensure_ascii=False, indent=2)


def load_scaler(path):
    with open(path, "r", encoding="utf-8") as f:
        return InputScaler.from_dict(json.load(f))


class LymphNodeDataset(torch.utils.data.Dataset):
    def __init__(self, pt_df, feat_key, scaler):
        self.pt = pt_df.reset_index(drop=True)
        self.feat_key = feat_key
        self.scaler = scaler

    def __len__(self):
        return len(self.pt)

    def __getitem__(self, idx):
        row = self.pt.iloc[idx]
        tumor_feats = load_features(row["tumor_slide_feat_path"], self.feat_key)
        has_met = int(row["has_met"])
        if has_met:
            node_feats = load_features(row["lymph_node_slide_feat_path"], self.feat_key)
            if node_feats.shape[-1] != tumor_feats.shape[-1]:
                raise ValueError(
                    f"{row['case_id']} 原发灶特征维 {tumor_feats.shape[-1]} "
                    f"与淋巴结特征维 {node_feats.shape[-1]} 不一致"
                )
        else:
            node_feats = np.zeros((0, tumor_feats.shape[-1]), dtype=np.float32)
        size_norm = float(row["tumor_size"]) / self.scaler.tumor_size_scale
        count_norm = float(row["n_met"]) / self.scaler.ln_count_scale
        return (
            torch.from_numpy(tumor_feats).float(),
            torch.from_numpy(node_feats).float(),
            size_norm,
            count_norm,
            has_met,
            int(row["disc_label"]),
            float(row["event_time"]),
            float(row["censorship"]),
            str(row["case_id"]),
            float(row["tumor_size"]),
            float(row["n_met"]),
        )


def collate_case(batch):
    return batch[0]


def make_loader(pt_df, cfg, training, scaler):
    ds = LymphNodeDataset(pt_df, cfg["feat_key"], scaler)
    return torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=training, num_workers=cfg["num_workers"],
        collate_fn=collate_case,
    )


def _scalar(value, device):
    return torch.tensor([float(value)], dtype=torch.float32, device=device)


@torch.no_grad()
def initialize_survival_head(model, pt_train, scaler):
    """仅用训练折事件/在险人数初始化 baseline，避免初始 hazard 普遍大于 0.5。"""
    labels = pt_train["disc_label"].to_numpy()
    events = pt_train["censorship"].to_numpy() == 0
    priors = []
    for k in range(model.bin_bias.numel()):
        n_at_risk = int(np.sum(labels >= k))
        n_events = int(np.sum(events & (labels == k)))
        priors.append((n_events + 0.5) / (n_at_risk + 1) if n_at_risk else 0.1)
    coeffs = model.rcb.coeff_dict()
    a, b, c, d = (coeffs[key] for key in ("a", "b", "c", "d"))
    size = pt_train["tumor_size"].to_numpy() / scaler.tumor_size_scale
    count = pt_train["n_met"].to_numpy() / scaler.ln_count_scale
    reference_risk = float(np.mean(a * (0.5 * size) ** b
                                   + c * (0.5 * count) ** d))
    h = model.bin_bias.new_tensor(priors)
    scale = F.softplus(model.hazard_scale_raw)
    model.bin_bias.copy_(torch.logit(h) - scale * reference_risk)
    return {"hazard_priors": priors, "reference_risk": reference_risk,
            "bin_bias": model.bin_bias.cpu().tolist()}


def _gradient_norm(module):
    grads = [p.grad.detach() for p in module.parameters() if p.grad is not None]
    if not grads:
        return None
    return float(torch.linalg.vector_norm(torch.stack(
        [torch.linalg.vector_norm(g) for g in grads]
    )).cpu())


def prediction_diagnostics(rows):
    """淋巴结统计仅包含有转移者，避免正常的缺失分支 0 掩盖评分分布。"""
    stats = {}
    for key in ("rcb", "tumor_score", "lymph_score", "tumor_logit", "lymph_logit"):
        selected = rows if not key.startswith("lymph") else [r for r in rows if r["has_met"]]
        values = np.asarray([r[key] for r in selected], dtype=float)
        stats[f"{key}_n"] = len(values)
        if not len(values):
            continue
        for name, value in (("mean", values.mean()), ("std", values.std()),
                            ("min", values.min()), ("max", values.max())):
            stats[f"{key}_{name}"] = float(value)
        if key == "rcb":
            stats["rcb_n_unique"] = int(len(np.unique(values)))
        if key.endswith("score"):
            stats[f"{key}_zero_fraction"] = float(np.mean(values == 0))
            stats[f"{key}_near_zero_fraction"] = float(np.mean(values < 1e-4))
    return stats


def print_diagnostics(stats, prefix):
    for key in ("rcb", "tumor_score", "lymph_score"):
        n = stats.get(f"{key}_n", 0)
        if not n:
            continue
        msg = (f"[{prefix} diag] {key}: n={n} mean={stats[key + '_mean']:.3e} "
               f"std={stats[key + '_std']:.3e} min={stats[key + '_min']:.3e} "
               f"max={stats[key + '_max']:.3e}")
        if key == "rcb":
            msg += f" unique={stats['rcb_n_unique']}"
        else:
            msg += f" zero={stats[key + '_zero_fraction']:.1%} <1e-4={stats[key + '_near_zero_fraction']:.1%}"
        print(msg)
    if stats.get("rcb_n", 0) > 1 and stats.get("rcb_std", 0) < 1e-8:
        warnings.warn(f"{prefix}: R 几乎没有个体差异，请检查评分分布和梯度。", RuntimeWarning)
    for key in ("tumor_score", "lymph_score"):
        if stats.get(f"{key}_n", 0) and (
            stats[f"{key}_max"] < 1e-4 or stats[f"{key}_min"] > 1 - 1e-4
        ):
            warnings.warn(f"{prefix}: {key} 已接近 sigmoid 饱和区，请检查 logits 并降低学习率。",
                          RuntimeWarning)


def run_epoch(model, loader, optimizer, cfg, device, train=True, return_details=False):
    model.train() if train else model.eval()
    total_loss = 0.0
    risks, censors, times = [], [], []
    rows = []
    grad_norms = {"tumor": [], "node": []}
    gc = max(1, int(cfg["gc"]))
    if train:
        optimizer.zero_grad()

    for i, sample in enumerate(loader):
        (tumor_feats, node_feats, size_norm, count_norm, has_met,
         label, et, c, _cid, _raw_size, _raw_count) = sample
        tumor_feats = tumor_feats.to(device, non_blocking=True)
        node_feats = node_feats.to(device, non_blocking=True)
        size_t = _scalar(size_norm, device)
        count_t = _scalar(count_norm, device)
        met_t = _scalar(has_met, device)
        label_t = torch.tensor([label], device=device)
        c_t = torch.tensor([c], device=device, dtype=torch.float32)

        with torch.set_grad_enabled(train):
            hazards, S, parts = model(tumor_feats, node_feats, size_t, count_t, met_t)
            loss = nll_surv_loss(hazards, S, label_t, c_t, alpha=cfg["alpha_surv"],
                                 logits=parts["logits"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"患者 {_cid} 的生存损失不是有限数，请检查输入特征和学习率。")

        if train:
            window_size = min(gc, len(loader) - (i // gc) * gc)
            (loss / window_size).backward()
            if (i + 1) % gc == 0 or (i + 1) == len(loader):
                if return_details:
                    for name, module in (("tumor", model.tumor_scorer), ("node", model.node_scorer)):
                        norm = _gradient_norm(module)
                        if norm is not None:
                            grad_norms[name].append(norm)
                clip = cfg.get("grad_clip", 5.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip > 0 else float("inf"),
                                               error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += float(loss.detach().cpu())
        risks.append(float(parts["rcb"].detach().cpu().view(-1)[0]))
        censors.append(c)
        times.append(et)
        if return_details:
            row = {"case_id": _cid, "has_met": has_met, "event_time": et, "censorship": c}
            for key in ("rcb", "tumor_term", "lymph_term", "tumor_score", "lymph_score",
                        "tumor_logit", "lymph_logit", "tumor_log_score", "lymph_log_score"):
                row[key] = float(parts[key].detach().cpu().reshape(-1)[0])
            rows.append(row)

    avg_loss = total_loss / max(1, len(loader))
    c_index = concordance(censors, times, risks)
    if return_details:
        stats = prediction_diagnostics(rows)
        if train:
            for name, norms in grad_norms.items():
                stats[f"{name}_grad_updates"] = len(norms)
                if norms:
                    stats[f"{name}_grad_norm_mean"] = float(np.mean(norms))
                    stats[f"{name}_grad_zero_fraction"] = float(np.mean(np.asarray(norms) == 0))
                    if all(v == 0 for v in norms):
                        warnings.warn(f"{name} 分支本轮所有更新的梯度均为 0。", RuntimeWarning)
        return avg_loss, c_index, {"diagnostics": stats, "predictions": rows}
    return avg_loss, c_index


def train_one_run(pt_train, pt_val, cfg, device, out_dir, fold_tag=""):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg["seed"])

    scaler = InputScaler().fit(pt_train)
    save_scaler(scaler, os.path.join(out_dir, "input_scaler.json"))
    print(f"[{fold_tag}] size_scale={scaler.tumor_size_scale:.4f} "
          f"ln_count_scale={scaler.ln_count_scale:.4f} "
          f"有转移 {int(pt_train['has_met'].sum())}/{len(pt_train)}")

    model = build_model(cfg, device)
    head_init = initialize_survival_head(model, pt_train, scaler)
    with open(os.path.join(out_dir, "survival_head_init.json"), "w", encoding="utf-8") as f:
        json.dump(head_init, f, ensure_ascii=False, indent=2)
    print(f"[{fold_tag}] initial hazard priors: "
          + ", ".join(f"{p:.4f}" for p in head_init["hazard_priors"]))
    optimizer = get_optimizer(model, cfg)
    train_loader = make_loader(pt_train, cfg, training=True, scaler=scaler)
    val_loader = (
        make_loader(pt_val, cfg, training=False, scaler=scaler)
        if pt_val is not None else None
    )

    history = []
    best_cindex, best_epoch = -1.0, -1
    best_loss = float("inf")
    best_val_loss = float("inf")
    best_coeffs = None
    ckpt_best = os.path.join(out_dir, "checkpoint_best.pt")
    ckpt_best_loss = os.path.join(out_dir, "checkpoint_best_loss.pt")
    ckpt_last = os.path.join(out_dir, "checkpoint_last.pt")

    for epoch in range(cfg["max_epochs"]):
        tr_loss, tr_cidx, tr_details = run_epoch(
            model, train_loader, optimizer, cfg, device, train=True, return_details=True)
        coeffs = model.formula_coeffs()
        rec = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_cindex": tr_cidx,
            "a": coeffs["a"], "b": coeffs["b"], "c": coeffs["c"], "d": coeffs["d"],
        }
        rec.update({f"train_{k}": v for k, v in tr_details["diagnostics"].items()})
        for name in ("tumor", "node"):
            norm = tr_details["diagnostics"].get(f"{name}_grad_norm_mean")
            if norm is not None:
                print(f"[{fold_tag} epoch {epoch} grad] {name}: mean_norm={norm:.3e}")
        coeff_txt = (f"a={coeffs['a']:.3f} b={coeffs['b']:.3f} "
                     f"c={coeffs['c']:.3f} d={coeffs['d']:.3f}")
        if val_loader is not None:
            val_loss, val_cidx, val_details = run_epoch(
                model, val_loader, optimizer, cfg, device, train=False, return_details=True)
            rec["val_loss"] = val_loss
            rec["val_cindex"] = val_cidx
            rec.update({f"val_{k}": v for k, v in val_details["diagnostics"].items()})
            print_diagnostics(val_details["diagnostics"], f"{fold_tag} epoch {epoch} val")
            predictions = pd.DataFrame(val_details["predictions"])
            predictions.to_csv(os.path.join(out_dir, "validation_predictions.csv"), index=False,
                               float_format="%.17g", na_rep="")
            if val_cidx == val_cidx and (val_cidx > best_cindex or
                                        (val_cidx == best_cindex and val_loss < best_val_loss)):
                best_cindex, best_epoch = val_cidx, epoch
                best_val_loss = val_loss
                best_coeffs = coeffs
                torch.save(model.state_dict(), ckpt_best)
                predictions.to_csv(os.path.join(out_dir, "validation_predictions_best.csv"),
                                   index=False, float_format="%.17g", na_rep="")
            print(f"[{fold_tag} epoch {epoch}] train_loss={tr_loss:.4f} "
                  f"train_cindex={tr_cidx:.4f} val_loss={val_loss:.4f} "
                  f"val_cindex={val_cidx:.6f} {coeff_txt}")
        else:
            if tr_loss < best_loss:
                best_loss, best_epoch = tr_loss, epoch
                best_coeffs = coeffs
                torch.save(model.state_dict(), ckpt_best_loss)
            print(f"[{fold_tag} epoch {epoch}] train_loss={tr_loss:.4f} "
                  f"train_cindex={tr_cidx:.4f} {coeff_txt}")
            print_diagnostics(tr_details["diagnostics"], f"{fold_tag} epoch {epoch} train")
        history.append(rec)
        # 每轮落盘，训练中断后仍可检查梯度、评分和指标。
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    torch.save(model.state_dict(), ckpt_last)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    if best_coeffs is None:
        best_coeffs = model.formula_coeffs()
    with open(os.path.join(out_dir, "formula.json"), "w", encoding="utf-8") as f:
        json.dump(best_coeffs, f, ensure_ascii=False, indent=2)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_cindex": best_cindex if val_loader is not None else None,
        "best_loss": best_loss if val_loader is None else None,
        "best_coeffs": best_coeffs,
    }


def train_kfold(pt, cfg, device, log_dir):
    case_ids = pt["case_id"].values
    kf = KFold(n_splits=cfg["k"], shuffle=True, random_state=cfg["seed"])
    fold_summaries = []
    best_epochs = []
    val_cindices = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(case_ids)):
        print(f"\n========== Fold {fold} / {cfg['k']} ==========")
        pt_tr = pt.iloc[tr_idx].reset_index(drop=True)
        pt_va = pt.iloc[va_idx].reset_index(drop=True)
        fold_dir = os.path.join(log_dir, f"fold_{fold}")
        res = train_one_run(pt_tr, pt_va, cfg, device, fold_dir, fold_tag=f"fold{fold}")
        best_epochs.append(res["best_epoch"])
        val_cindices.append(res["best_cindex"])
        fold_summaries.append({
            "fold": fold,
            "best_epoch": res["best_epoch"],
            "best_val_cindex": res["best_cindex"],
            "best_coeffs": res["best_coeffs"],
            "n_train": int(len(pt_tr)),
            "n_val": int(len(pt_va)),
        })

    valid = [v for v in val_cindices if v is not None and v == v]
    summary = {
        "split_mode": "kfold",
        "k": cfg["k"],
        "folds": fold_summaries,
        "best_epochs": best_epochs,
        "val_cindex_per_fold": val_cindices,
        "mean_val_cindex": float(np.mean(valid)) if valid else None,
        "std_val_cindex": float(np.std(valid)) if valid else None,
    }
    with open(os.path.join(log_dir, "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n===== K-fold 完成 =====")
    print(f"best_epochs: {best_epochs}")
    print(f"mean_val_cindex: {summary['mean_val_cindex']}")
    return summary


def train_all(pt, cfg, device, log_dir):
    print("\n========== All-train（全量训练） ==========")
    res = train_one_run(pt, None, cfg, device, log_dir, fold_tag="all")
    summary = {
        "split_mode": "all_train",
        "best_epoch_lowest_loss": res["best_epoch"],
        "lowest_train_loss": res["best_loss"],
        "best_coeffs": res["best_coeffs"],
        "n_train": int(len(pt)),
        "ckpt_last": os.path.join(log_dir, "checkpoint_last.pt"),
        "ckpt_best_loss": os.path.join(log_dir, "checkpoint_best_loss.pt"),
    }
    with open(os.path.join(log_dir, "all_train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n===== All-train 完成 =====")
    print(f"最低 train loss epoch: {res['best_epoch']}, loss={res['best_loss']:.4f}")
    print(f"公式系数: {res['best_coeffs']}")
    return summary


@torch.no_grad()
def run_inference(cfg, device, args):
    df = read_csv_smart(args.csv_path)
    pt, _ = build_patient_table(df, cfg["target"], cfg["n_bins"], cfg)

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
    scaler_path = os.path.join(ckpt_dir, "input_scaler.json")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(
            f"未找到 {scaler_path}。推理必须使用与该权重同一目录下、训练集拟合的 input_scaler.json。"
        )
    scaler = load_scaler(scaler_path)

    model = build_model(cfg, device)
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    loader = make_loader(pt, cfg, training=False, scaler=scaler)
    rows, risks, censors, times = [], [], [], []
    for sample in loader:
        (tumor_feats, node_feats, size_norm, count_norm, has_met,
         _label, et, c, cid, raw_size, raw_count) = sample
        tumor_feats = tumor_feats.to(device)
        node_feats = node_feats.to(device)
        hazards, S, parts = model(
            tumor_feats, node_feats,
            _scalar(size_norm, device), _scalar(count_norm, device), _scalar(has_met, device),
        )
        rcb = float(parts["rcb"].cpu().view(-1)[0])
        rows.append({
            "case_id": cid,
            "risk_score": rcb,
            "rcb_index": rcb,
            "tumor_term": float(parts["tumor_term"].cpu().view(-1)[0]),
            "lymph_term": float(parts["lymph_term"].cpu().view(-1)[0]),
            "tumor_slide_score": float(parts["tumor_score"].cpu().view(-1)[0]),
            "lymph_slide_score": float(parts["lymph_score"].cpu().view(-1)[0]),
            "tumor_logit": float(parts["tumor_logit"].cpu().view(-1)[0]),
            "lymph_logit": float(parts["lymph_logit"].cpu().view(-1)[0]),
            "tumor_log_score": float(parts["tumor_log_score"].cpu().view(-1)[0]),
            "lymph_log_score": float(parts["lymph_log_score"].cpu().view(-1)[0]),
            "tumor_size": raw_size,
            "n_metastatic_lymph_nodes": raw_count,
            "event_time": et,
            "censorship": c,
            "status": int(1 - c),
            "surv_risk": float(-torch.sum(S, dim=1).cpu().item()),
        })
        risks.append(rcb)
        censors.append(c)
        times.append(et)

    os.makedirs(args.save_infer_dir, exist_ok=True)
    pred_df = pd.DataFrame(rows)
    pred_csv = os.path.join(args.save_infer_dir, "risk_scores.csv")
    pred_df.to_csv(pred_csv, index=False, float_format="%.17g", na_rep="")

    c_index = concordance(censors, times, risks)
    metrics = {
        "n_cases": int(len(pred_df)),
        "n_events": int(np.sum(1 - np.asarray(censors))),
        "n_node_positive": int(sum(1 for r in rows if r["n_metastatic_lymph_nodes"] > 0)),
        "c_index": c_index,
        "target": cfg["target"],
        "formula": model.formula_coeffs(),
        "scaler": scaler.to_dict(),
        "ckpt_path": os.path.abspath(args.ckpt_path),
        "csv_path": os.path.abspath(args.csv_path),
    }
    with open(os.path.join(args.save_infer_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"推理完成：c-index={c_index:.4f}，结果保存到 {args.save_infer_dir}")
    print(f"  公式系数: {metrics['formula']}")
    print(f"  每患者 risk_score: {pred_csv}")
    return metrics


def set_seed(seed=1):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_optimizer(model, cfg):
    formula_ids = formula_param_ids(model)
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in formula_ids:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg["reg"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg["opt"] == "sgd":
        return torch.optim.SGD(groups, lr=cfg["lr"], momentum=0.9)
    return torch.optim.Adam(groups, lr=cfg["lr"])


def read_csv_smart(path):
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def build_config(args):
    cfg = {k: getattr(args, k) for k in HPARAM_KEYS if hasattr(args, k)}
    saved_version = None
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            override = json.load(f)
        saved_version = override.get("numerics_version")
        for k, v in override.items():
            if k in HPARAM_KEYS:
                cfg[k] = v
        print(f"已从 {args.config} 覆盖超参数: {[k for k in override if k in HPARAM_KEYS]}")
    if args.mode == "infer" and saved_version != NUMERICS_VERSION:
        raise ValueError(
            "该 config 来自旧版数值路径或缺少 numerics_version=2。"
            "本修复改变了评分归一化和低分区域的公式计算，请使用新实验名重新训练，"
            "并将新版 config、权重和 input_scaler 配套用于推理；旧实验需沿用旧代码复现。"
        )
    cfg["numerics_version"] = NUMERICS_VERSION
    if args.mode == "train" and cfg["lr"] >= 1e-3:
        warnings.warn(f"lr={cfg['lr']:g} 容易使 WSI 评分饱和；建议先用 --lr 1e-4 并检查评分/梯度。",
                      RuntimeWarning)
    if cfg.get("grad_clip", 5.0) < 0:
        raise ValueError("grad_clip 必须 >= 0；0 表示禁用裁剪。")
    return cfg


def get_args():
    p = argparse.ArgumentParser(description="原发灶 + 淋巴结的 RCB 风格预后训练/推理")
    p.add_argument("--mode", choices=["train", "infer"], default="train")
    p.add_argument("--csv_path", type=str, required=True, help="输入 CSV，列格式见 Example_Dataset_Csv.csv")
    p.add_argument("--log_root", type=str, default="./logs")
    p.add_argument("--exp_name", type=str, default="exp")
    p.add_argument("--config", type=str, default=None, help="超参数 json")
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--save_infer_dir", type=str, default=None)

    p.add_argument("--target", choices=["OS", "DFS", "RFS"], default="OS")
    p.add_argument("--split_mode", choices=["kfold", "all_train"], default="kfold")
    p.add_argument("--k", type=int, default=5)

    p.add_argument("--n_bins", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--reg", type=float, default=1e-5)
    p.add_argument("--drop_out", type=float, default=0.25)
    p.add_argument("--gc", type=int, default=16)
    p.add_argument("--alpha_surv", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--score_norm", choices=["layernorm", "none"], default="layernorm",
                   help="评分前对切片表征做无可训练参数的 LayerNorm；none 用于消融")
    p.add_argument("--grad_clip", type=float, default=5.0, help="梯度范数裁剪阈值；0 表示禁用")

    p.add_argument("--model_type", choices=["abmil", "mean_mil", "max_mil"], default="abmil")
    p.add_argument("--in_dim", type=int, default=-1, help="特征维度，<=0 时从 h5 自动推断")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--feat_key", type=str, default="features")

    p.add_argument("--tumor_feat_col", type=str, default="tumor_slide_feat_path")
    p.add_argument("--ln_feat_col", type=str, default="lymph_node_slide_feat_path")
    p.add_argument("--tumor_size_col", type=str, default="tumor_size", help="肿瘤长径列名")
    p.add_argument("--ln_count_col", type=str, default="n_metastatic_lymph_nodes",
                   help="转移淋巴结个数列名")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    cfg = build_config(args)

    if args.mode == "infer":
        assert args.ckpt_path and args.save_infer_dir, "推理需要 --ckpt_path 与 --save_infer_dir"
        assert args.config, "推理需要 --config 指定训练时保存的超参数 json"
        if cfg.get("in_dim", -1) is None or cfg.get("in_dim", -1) <= 0:
            df = read_csv_smart(args.csv_path)
            cfg["in_dim"] = detect_in_dim(
                df[cfg["tumor_feat_col"]].map(_clean_path).tolist(), cfg["feat_key"]
            )
        set_seed(cfg["seed"])
        run_inference(cfg, device, args)
        return

    set_seed(cfg["seed"])
    df = read_csv_smart(args.csv_path)
    pt, n_classes = build_patient_table(df, cfg["target"], cfg["n_bins"], cfg)
    cfg["n_classes"] = int(n_classes)
    if cfg["in_dim"] is None or cfg["in_dim"] <= 0:
        cfg["in_dim"] = detect_in_dim(pt["tumor_slide_feat_path"].tolist(), cfg["feat_key"])
    node_paths = [p for p in pt["lymph_node_slide_feat_path"].tolist() if p]
    if node_paths:
        node_dim = detect_in_dim(node_paths, cfg["feat_key"])
        if node_dim != cfg["in_dim"]:
            raise ValueError(f"原发灶特征维 {cfg['in_dim']} 与淋巴结特征维 {node_dim} 不一致")

    print(f"患者数: {len(pt)}, 事件数: {int((pt['censorship'] == 0).sum())}, "
          f"有淋巴结转移: {int(pt['has_met'].sum())}, "
          f"特征维度: {cfg['in_dim']}, n_classes: {cfg['n_classes']}")

    log_dir = os.path.join(args.log_root, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"超参数已保存到 {os.path.join(log_dir, 'config.json')}")

    if cfg["split_mode"] == "kfold":
        if len(pt) < cfg["k"]:
            raise ValueError(f"患者数 {len(pt)} 小于折数 k={cfg['k']}")
        train_kfold(pt, cfg, device, log_dir)
    else:
        train_all(pt, cfg, device, log_dir)


if __name__ == "__main__":
    main()
