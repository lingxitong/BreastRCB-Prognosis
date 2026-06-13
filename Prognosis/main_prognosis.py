#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_prognosis.py
==================
基于 MambaMIL 预后（生存）逻辑实现的单脚本训练/推理工具。

核心逻辑沿用 MambaMIL：
  * 离散时间生存建模：将事件时间按未删失样本分位数离散为 n_bins 个区间，
    模型输出每个区间的 hazard，S = cumprod(1 - hazard)，风险分 risk = -sum(S)。
  * 损失：NLL 生存损失 (NLLSurvLoss)。
  * 评价：sksurv 的 concordance_index_censored（event = 1 - censorship）。
  * 一个患者(case)可能有多张 slide，训练时拼接成一个 bag；若 slide 数 > max_slides
    则随机选取 max_slides 张拼接；推理时拼接全部 slide。

输入 CSV 格式见 Example_Dataset_Csv.csv，关键列：
  case_id, slide_id, slide_feat_path, OS_month, OS_status, DFS_month, DFS_status,
  RFS_month, RFS_status, ...
其中 *_status 约定 1=事件发生(死亡/复发/进展), 0=删失(censored)。
slide_feat_path 指向每张 slide 的特征 .h5 文件（数据集键默认 'features'，形状 [N_patch, dim]）。

支持三种运行模式（--mode）:
  train  : 训练。--split_mode 控制 kfold（默认）或 all_train。
  infer  : 推理。给定超参 json + 权重 + csv，输出指标和每个患者的 risk_score。

示例：
  # K 折交叉验证（默认 k=5），训练目标 OS
  python train_prognosis.py --mode train --split_mode kfold --target OS \
      --csv_path Example_Dataset_Csv.csv --log_root ./logs --exp_name os_kfold

  # 全量训练
  python train_prognosis.py --mode train --split_mode all_train --target DFS \
      --csv_path Example_Dataset_Csv.csv --log_root ./logs --exp_name dfs_all

  # 通过 json 覆盖超参数
  python train_prognosis.py --mode train --config my_hparams.json ...

  # 推理
  python train_prognosis.py --mode infer --config ./logs/os_kfold/config.json \
      --ckpt_path ./logs/os_kfold/fold_0/checkpoint_best.pt \
      --csv_path test.csv --save_infer_dir ./infer_os
"""

from __future__ import print_function

import argparse
import json
import os
import random
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

# ----------------------------------------------------------------------------
# c-index：优先 sksurv，回退 lifelines，最终回退到内置 numpy 实现（无需额外依赖）
# 约定：censorship=1 表示删失，0 表示事件；risk 越大代表生存越短。
# ----------------------------------------------------------------------------
def _builtin_cindex(censorships, event_times, risk_scores):
    """Harrell's concordance index（含删失）的纯 numpy 实现。"""
    c = np.asarray(censorships)
    t = np.asarray(event_times, dtype=float)
    r = np.asarray(risk_scores, dtype=float)
    n = len(t)
    num, den = 0.0, 0.0
    for i in range(n):
        if c[i] == 0:  # i 发生了事件，才可作为可比对的"较短生存"样本
            for j in range(n):
                if t[j] > t[i]:  # j 生存更久，构成可比对
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


# 这些 key 属于"超参数"，会被保存进 config.json，也可被 --config 的 json 覆盖
HPARAM_KEYS = [
    "target", "k", "split_mode", "n_bins", "n_classes", "max_epochs", "lr", "reg",
    "drop_out", "gc", "alpha_surv", "seed", "opt", "model_type", "in_dim",
    "hidden_dim", "max_slides_train", "feat_key", "num_workers",
    "mambamil_layer", "mambamil_rate", "mambamil_type",
    "use_clinical", "fusion_type", "clinical_hidden_dim", "clinical_in_dim",
]

# 不参与临床编码的列：标识符、路径、生存标签
META_COLS = {"case_id", "slide_id", "slide_feat_path"}
SURVIVAL_COLS = {
    "OS_month", "OS_status", "DFS_month", "DFS_status", "RFS_month", "RFS_status",
}
# 已知类别型临床列（其余 object/低基数列也会自动识别为类别）
KNOWN_CATEGORICAL = {"HER2_score_pre", "Molecular_subtype"}


# ============================================================================
# 损失函数（离散时间 NLL 生存损失，沿用 MambaMIL）
# ============================================================================
def nll_surv_loss(hazards, S, Y, c, alpha=0.0, eps=1e-7):
    """
    hazards : [B, n_classes]  每个时间区间的 hazard
    S       : [B, n_classes]  生存函数 = cumprod(1 - hazards)
    Y       : [B]             离散时间区间标签 (0..n_classes-1)
    c       : [B]             censorship (1=删失, 0=事件)
    """
    batch_size = len(Y)
    Y = Y.view(batch_size, 1).long()
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1)=1
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    return loss.mean()


# ============================================================================
# 模型：MIL 聚合器 -> 全局表征；可选与临床特征中期融合 -> 生存头
# fusion_type: concat / bilinear / gated
# ============================================================================
def _init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class ABMILBackbone(nn.Module):
    """Gated-Attention MIL，输出 slide 级全局表征 [1, hidden]。"""

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


class MeanMaxBackbone(nn.Module):
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
        return h.mean(dim=0, keepdim=True) if self.pool == "mean" else h.max(dim=0, keepdim=True)[0]


class ConcatFusion(nn.Module):
    def __init__(self, mil_dim, clin_dim, out_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(mil_dim + clin_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, mil_repr, clin_repr):
        return self.net(torch.cat([mil_repr, clin_repr], dim=-1))


class BilinearFusion(nn.Module):
    def __init__(self, mil_dim, clin_dim, out_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(mil_dim, clin_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, mil_repr, clin_repr):
        return self.norm(F.relu(self.bilinear(mil_repr, clin_repr)))


class GatedFusion(nn.Module):
    """门控融合：根据 [全局表征; 临床嵌入] 学习逐维权重。"""

    def __init__(self, mil_dim, clin_dim, out_dim):
        super().__init__()
        self.proj_mil = nn.Linear(mil_dim, out_dim)
        self.proj_clin = nn.Linear(clin_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(mil_dim + clin_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, mil_repr, clin_repr):
        gate = self.gate(torch.cat([mil_repr, clin_repr], dim=-1))
        return gate * self.proj_mil(mil_repr) + (1.0 - gate) * self.proj_clin(clin_repr)


def build_fusion(fusion_type, mil_dim, clin_dim, out_dim, dropout=0.25):
    if fusion_type == "concat":
        return ConcatFusion(mil_dim, clin_dim, out_dim, dropout)
    if fusion_type == "bilinear":
        return BilinearFusion(mil_dim, clin_dim, out_dim)
    if fusion_type == "gated":
        return GatedFusion(mil_dim, clin_dim, out_dim)
    raise ValueError(f"未知 fusion_type: {fusion_type}，可选 concat/bilinear/gated")


class PathomicSurvivalModel(nn.Module):
    """
    中期融合预后模型：
      bag -> MIL backbone -> 全局表征
      临床向量 -> MLP -> 临床嵌入
      fusion(全局表征, 临床嵌入) -> 生存 hazard 头
    """

    def __init__(self, backbone, n_classes, clinical_in_dim, fusion_type,
                 hidden_dim=512, dropout=0.25, use_clinical=True):
        super().__init__()
        self.backbone = backbone
        self.use_clinical = use_clinical and clinical_in_dim > 0
        mil_dim = backbone.hidden_dim

        if self.use_clinical:
            clin_emb_dim = max(32, hidden_dim // 2)
            self.clinical_mlp = nn.Sequential(
                nn.Linear(clinical_in_dim, clin_emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.fusion = build_fusion(fusion_type, mil_dim, clin_emb_dim, hidden_dim, dropout)
            self.classifier = nn.Linear(hidden_dim, n_classes)
        else:
            self.classifier = nn.Linear(mil_dim, n_classes)
        self.apply(_init_weights)

    def forward(self, x, clinical=None):
        bag_repr = self.backbone(x)
        if self.use_clinical:
            if clinical is None:
                raise ValueError("use_clinical=True 时必须提供 clinical 特征")
            clin = clinical.unsqueeze(0) if clinical.dim() == 1 else clinical
            clin_emb = self.clinical_mlp(clin)
            fused = self.fusion(bag_repr, clin_emb)
            logits = self.classifier(fused)
        else:
            logits = self.classifier(bag_repr)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, logits


def build_backbone(cfg):
    mt = cfg["model_type"]
    if mt == "abmil":
        return ABMILBackbone(cfg["in_dim"], dropout=cfg["drop_out"], hidden=cfg["hidden_dim"])
    if mt == "mean_mil":
        return MeanMaxBackbone(cfg["in_dim"], dropout=cfg["drop_out"],
                               hidden=cfg["hidden_dim"], pool="mean")
    if mt == "max_mil":
        return MeanMaxBackbone(cfg["in_dim"], dropout=cfg["drop_out"],
                               hidden=cfg["hidden_dim"], pool="max")
    if mt in ("mamba_mil", "trans_mil", "s4model"):
        if cfg.get("use_clinical", True):
            raise NotImplementedError(
                f"临床中期融合暂不支持 model_type={mt}，请使用 abmil/mean_mil/max_mil，"
                f"或设置 use_clinical=false"
            )
        return _build_repo_model(cfg)
    raise NotImplementedError(f"未知 model_type: {mt}")


def build_model(cfg, device):
    if cfg["model_type"] in ("mamba_mil", "trans_mil", "s4model") and not cfg.get("use_clinical", True):
        model = build_backbone(cfg)
    else:
        backbone = build_backbone(cfg)
        clinical_in_dim = int(cfg.get("clinical_in_dim", 0) or 0)
        use_clinical = bool(cfg.get("use_clinical", True)) and clinical_in_dim > 0
        model = PathomicSurvivalModel(
            backbone, cfg["n_classes"], clinical_in_dim,
            fusion_type=cfg.get("fusion_type", "concat"),
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["drop_out"],
            use_clinical=use_clinical,
        )
    return model.to(device)


def _build_repo_model(cfg):
    """从 MambaMIL 仓库动态构建模型（仅 path-only，无临床融合）。"""
    import sys

    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MambaMIL")
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    mt = cfg["model_type"]

    class RepoSurvivalWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner

        def forward(self, x, clinical=None):
            out = self.model(x)
            return out[0], out[1], out[0]

    try:
        if mt == "mamba_mil":
            from models.MambaMIL import MambaMIL
            m = MambaMIL(in_dim=cfg["in_dim"], n_classes=cfg["n_classes"],
                         dropout=cfg["drop_out"], act="gelu", survival=True,
                         layer=cfg["mambamil_layer"], rate=cfg["mambamil_rate"],
                         type=cfg["mambamil_type"])
        elif mt == "trans_mil":
            from models.TransMIL import TransMIL
            m = TransMIL(cfg["in_dim"], cfg["n_classes"], dropout=cfg["drop_out"],
                         act="relu", survival=True)
        else:
            from models.S4MIL import S4Model
            m = S4Model(in_dim=cfg["in_dim"], n_classes=cfg["n_classes"], act="gelu",
                        dropout=cfg["drop_out"], survival=True)
    except Exception as e:
        raise RuntimeError(
            f"无法从 MambaMIL 仓库加载模型 '{mt}'（可能缺少依赖）。"
            f"可改用 model_type=abmil。原始错误: {e}"
        )
    return RepoSurvivalWrapper(m)


# ============================================================================
# 数据集
# ============================================================================
TARGET_COLS = {
    "OS": ("OS_month", "OS_status"),
    "DFS": ("DFS_month", "DFS_status"),
    "RFS": ("RFS_month", "RFS_status"),
}


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
    return np.asarray(feats, dtype=np.float32)


def get_clinical_columns(df):
    return [c for c in df.columns if c not in META_COLS and c not in SURVIVAL_COLS]


class ClinicalEncoder:
    """将 CSV 临床列编码为固定长度 float 向量（数值标准化 + 类别 one-hot）。"""

    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.numeric_mean = {}
        self.numeric_std = {}
        self.cat_categories = {}
        self.output_dim = 0
        self.fitted = False

    def _detect_columns(self, df, clinical_cols):
        numeric_cols, categorical_cols = [], []
        for col in clinical_cols:
            series = df[col]
            if col in KNOWN_CATEGORICAL or series.dtype == object:
                categorical_cols.append(col)
                continue
            coerced = pd.to_numeric(series, errors="coerce")
            nunique = coerced.nunique(dropna=True)
            if nunique <= 12 and set(coerced.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                numeric_cols.append(col)
            elif nunique <= 12 and col not in KNOWN_CATEGORICAL:
                # 低基数整数列仍按数值处理（如 Stage=3）
                numeric_cols.append(col)
            else:
                numeric_cols.append(col)
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    def fit(self, df):
        clinical_cols = get_clinical_columns(df)
        if not clinical_cols:
            self.fitted = True
            self.output_dim = 0
            return self
        self._detect_columns(df, clinical_cols)
        for col in self.numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce").astype(float)
            mean = float(vals.mean()) if vals.notna().any() else 0.0
            std = float(vals.std()) if vals.notna().any() else 1.0
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0
            self.numeric_mean[col] = mean
            self.numeric_std[col] = std
        for col in self.categorical_cols:
            cats = df[col].fillna("missing").astype(str).unique().tolist()
            cats = sorted(set(cats))
            if "missing" not in cats:
                cats.append("missing")
            self.cat_categories[col] = cats
        self.output_dim = len(self.numeric_cols) + sum(
            len(self.cat_categories[c]) for c in self.categorical_cols
        )
        self.fitted = True
        return self

    def transform_row(self, row):
        if not self.fitted or self.output_dim == 0:
            return np.zeros((0,), dtype=np.float32)
        feats = []
        for col in self.numeric_cols:
            val = pd.to_numeric(row.get(col, np.nan), errors="coerce")
            val = float(val) if pd.notna(val) else self.numeric_mean[col]
            val = (val - self.numeric_mean[col]) / self.numeric_std[col]
            feats.append(val)
        for col in self.categorical_cols:
            raw = row.get(col, "missing")
            val = "missing" if pd.isna(raw) else str(raw)
            cats = self.cat_categories[col]
            onehot = [1.0 if val == c else 0.0 for c in cats]
            feats.extend(onehot)
        return np.asarray(feats, dtype=np.float32)

    def transform_df(self, pt_df):
        return np.stack([self.transform_row(pt_df.iloc[i]) for i in range(len(pt_df))], axis=0)

    def to_dict(self):
        return {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "cat_categories": self.cat_categories,
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_dict(cls, d):
        enc = cls()
        enc.numeric_cols = list(d.get("numeric_cols", []))
        enc.categorical_cols = list(d.get("categorical_cols", []))
        enc.numeric_mean = {k: float(v) for k, v in d.get("numeric_mean", {}).items()}
        enc.numeric_std = {k: float(v) for k, v in d.get("numeric_std", {}).items()}
        enc.cat_categories = {k: list(v) for k, v in d.get("cat_categories", {}).items()}
        enc.output_dim = int(d.get("output_dim", 0))
        enc.fitted = True
        return enc


def save_clinical_encoder(encoder, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(encoder.to_dict(), f, ensure_ascii=False, indent=2)


def load_clinical_encoder(path):
    with open(path, "r", encoding="utf-8") as f:
        return ClinicalEncoder.from_dict(json.load(f))


def build_patient_table(df, target, n_bins):
    """构建患者级表：每个 case 一行，含 event_time / censorship / disc_label / 特征路径 / 临床列。"""
    month_col, status_col = TARGET_COLS[target]
    if month_col not in df.columns or status_col not in df.columns:
        raise KeyError(f"CSV 缺少目标列 {month_col}/{status_col}（target={target}）")

    df = df.copy()
    df[month_col] = pd.to_numeric(df[month_col], errors="coerce")
    df[status_col] = pd.to_numeric(df[status_col], errors="coerce")
    df = df.dropna(subset=[month_col, status_col, "case_id", "slide_feat_path"])
    clinical_cols = get_clinical_columns(df)

    records = []
    for case_id, g in df.groupby("case_id", sort=False):
        event_time = float(g[month_col].iloc[0])
        status = int(round(float(g[status_col].iloc[0])))
        censorship = 1 - status
        feat_paths = list(g["slide_feat_path"].astype(str))
        rec = {
            "case_id": case_id, "event_time": event_time,
            "censorship": censorship, "feat_paths": feat_paths,
        }
        for col in clinical_cols:
            rec[col] = g[col].iloc[0]
        records.append(rec)
    pt = pd.DataFrame(records).reset_index(drop=True)

    disc, n_classes = discretize_time(pt, n_bins)
    pt["disc_label"] = disc
    return pt, n_classes, clinical_cols


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


class SurvivalBagDataset(torch.utils.data.Dataset):
    """患者级数据集；每个样本返回 bag 特征 + 临床向量。"""

    def __init__(self, pt_df, feat_key, max_slides_train, training, clinical_encoder=None):
        self.pt = pt_df.reset_index(drop=True)
        self.feat_key = feat_key
        self.max_slides_train = max_slides_train
        self.training = training
        self.clinical_encoder = clinical_encoder
        if clinical_encoder is not None and clinical_encoder.output_dim > 0:
            self.clinical_matrix = clinical_encoder.transform_df(self.pt)
        else:
            self.clinical_matrix = None

    def __len__(self):
        return len(self.pt)

    def __getitem__(self, idx):
        row = self.pt.iloc[idx]
        paths = list(row["feat_paths"])
        if self.training and self.max_slides_train > 0 and len(paths) > self.max_slides_train:
            paths = random.sample(paths, self.max_slides_train)
        feats = [load_features(p, self.feat_key) for p in paths]
        feats = np.concatenate(feats, axis=0)
        if self.clinical_matrix is not None:
            clinical = torch.from_numpy(self.clinical_matrix[idx]).float()
        else:
            clinical = torch.zeros((0,), dtype=torch.float32)
        return (
            torch.from_numpy(feats).float(),
            clinical,
            int(row["disc_label"]),
            float(row["event_time"]),
            float(row["censorship"]),
            str(row["case_id"]),
        )


def collate_bag(batch):
    feats, clinical, label, et, c, cid = batch[0]
    return feats, clinical, label, et, c, cid


def make_loader(pt_df, cfg, training, clinical_encoder=None):
    ds = SurvivalBagDataset(
        pt_df, cfg["feat_key"], cfg["max_slides_train"], training, clinical_encoder
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=training, num_workers=cfg["num_workers"],
        collate_fn=collate_bag,
    )


def prepare_clinical_encoder(pt_train, cfg, out_dir=None):
    if not cfg.get("use_clinical", True):
        cfg["clinical_in_dim"] = 0
        return None
    encoder = ClinicalEncoder().fit(pt_train)
    cfg["clinical_in_dim"] = int(encoder.output_dim)
    if out_dir is not None:
        save_clinical_encoder(encoder, os.path.join(out_dir, "clinical_encoder.json"))
    print(f"临床特征维度: {encoder.output_dim}  "
          f"(数值列 {len(encoder.numeric_cols)}, 类别列 {len(encoder.categorical_cols)})")
    if encoder.numeric_cols:
        print(f"  数值列: {encoder.numeric_cols}")
    if encoder.categorical_cols:
        print(f"  类别列: {encoder.categorical_cols}")
    return encoder


def model_forward(model, feats, clinical, cfg, device):
    use_clinical = bool(cfg.get("use_clinical", True)) and int(cfg.get("clinical_in_dim", 0) or 0) > 0
    if use_clinical:
        clinical = clinical.to(device, non_blocking=True)
        return model(feats, clinical)
    return model(feats, None)


# ============================================================================
# 训练 / 验证 / 推理
# ============================================================================
def run_epoch(model, loader, optimizer, cfg, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    risks, censors, times = [], [], []
    gc = max(1, int(cfg["gc"]))
    if train:
        optimizer.zero_grad()

    for i, (feats, clinical, label, et, c, _cid) in enumerate(loader):
        feats = feats.to(device, non_blocking=True)
        label_t = torch.tensor([label], device=device)
        c_t = torch.tensor([c], device=device, dtype=torch.float32)

        with torch.set_grad_enabled(train):
            hazards, S, _ = model_forward(model, feats, clinical, cfg, device)
            loss = nll_surv_loss(hazards, S, label_t, c_t, alpha=cfg["alpha_surv"])

        if train:
            (loss / gc).backward()
            if (i + 1) % gc == 0 or (i + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        risks.append(float(-torch.sum(S, dim=1).detach().cpu().item()))
        censors.append(c)
        times.append(et)

    avg_loss = total_loss / max(1, len(loader))
    c_index = concordance(censors, times, risks)
    return avg_loss, c_index


def train_one_run(pt_train, pt_val, cfg, device, out_dir, fold_tag=""):
    """训练单个 run（一个 fold 或 all_train）。返回该 run 的历史与最佳信息。"""
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg["seed"])

    encoder = prepare_clinical_encoder(pt_train, cfg, out_dir)
    model = build_model(cfg, device)
    optimizer = get_optimizer(model, cfg)
    train_loader = make_loader(pt_train, cfg, training=True, clinical_encoder=encoder)
    val_loader = (
        make_loader(pt_val, cfg, training=False, clinical_encoder=encoder)
        if pt_val is not None else None
    )

    history = []  # 每个 epoch 的指标
    best_cindex, best_epoch = -1.0, -1
    best_loss = float("inf")
    ckpt_best = os.path.join(out_dir, "checkpoint_best.pt")          # kfold: 按 val_cindex
    ckpt_best_loss = os.path.join(out_dir, "checkpoint_best_loss.pt")  # all_train: 按 train loss
    ckpt_last = os.path.join(out_dir, "checkpoint_last.pt")

    for epoch in range(cfg["max_epochs"]):
        tr_loss, tr_cidx = run_epoch(model, train_loader, optimizer, cfg, device, train=True)

        rec = {"epoch": epoch, "train_loss": tr_loss, "train_cindex": tr_cidx}
        if val_loader is not None:
            val_loss, val_cidx = run_epoch(model, val_loader, optimizer, cfg, device, train=False)
            rec["val_loss"] = val_loss
            rec["val_cindex"] = val_cidx
            if val_cidx == val_cidx and val_cidx > best_cindex:  # 非 NaN 且更优
                best_cindex, best_epoch = val_cidx, epoch
                torch.save(model.state_dict(), ckpt_best)
            print(f"[{fold_tag} epoch {epoch}] train_loss={tr_loss:.4f} "
                  f"train_cindex={tr_cidx:.4f} val_loss={val_loss:.4f} val_cindex={val_cidx:.4f}")
        else:
            if tr_loss < best_loss:
                best_loss, best_epoch = tr_loss, epoch
                torch.save(model.state_dict(), ckpt_best_loss)
            print(f"[{fold_tag} epoch {epoch}] train_loss={tr_loss:.4f} train_cindex={tr_cidx:.4f}")

        history.append(rec)

    torch.save(model.state_dict(), ckpt_last)

    # 保存该 run 的逐 epoch 指标
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_cindex": best_cindex if val_loader is not None else None,
        "best_loss": best_loss if val_loader is None else None,
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
            "n_train": int(len(pt_tr)),
            "n_val": int(len(pt_va)),
        })

    valid = [v for v in val_cindices if v is not None and v == v]
    summary = {
        "split_mode": "kfold",
        "k": cfg["k"],
        "folds": fold_summaries,
        "best_epochs": best_epochs,           # k 个 best_epoch
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
        "n_train": int(len(pt)),
        "ckpt_last": os.path.join(log_dir, "checkpoint_last.pt"),
        "ckpt_best_loss": os.path.join(log_dir, "checkpoint_best_loss.pt"),
    }
    with open(os.path.join(log_dir, "all_train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n===== All-train 完成 =====")
    print(f"最低 train loss epoch: {res['best_epoch']}, loss={res['best_loss']:.4f}")
    return summary


@torch.no_grad()
def run_inference(cfg, device, args):
    df = read_csv_smart(args.csv_path)
    pt, _, _ = build_patient_table(df, cfg["target"], cfg["n_bins"])

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
    encoder_path = os.path.join(ckpt_dir, "clinical_encoder.json")
    encoder = None
    if cfg.get("use_clinical", True):
        if os.path.isfile(encoder_path):
            encoder = load_clinical_encoder(encoder_path)
            cfg["clinical_in_dim"] = encoder.output_dim
        else:
            print(f"警告: 未找到 {encoder_path}，将不使用临床特征推理")
            cfg["use_clinical"] = False
            cfg["clinical_in_dim"] = 0

    model = build_model(cfg, device)
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    loader = make_loader(pt, cfg, training=False, clinical_encoder=encoder)
    rows, risks, censors, times = [], [], [], []
    for feats, clinical, label, et, c, cid in loader:
        feats = feats.to(device)
        hazards, S, _ = model_forward(model, feats, clinical, cfg, device)
        risk = float(-torch.sum(S, dim=1).cpu().item())
        rows.append({"case_id": cid, "risk_score": risk,
                     "event_time": et, "censorship": c, "status": int(1 - c)})
        risks.append(risk)
        censors.append(c)
        times.append(et)

    os.makedirs(args.save_infer_dir, exist_ok=True)
    pred_df = pd.DataFrame(rows)
    pred_csv = os.path.join(args.save_infer_dir, "risk_scores.csv")
    pred_df.to_csv(pred_csv, index=False)

    c_index = concordance(censors, times, risks)
    metrics = {
        "n_cases": int(len(pred_df)),
        "n_events": int(np.sum(1 - np.asarray(censors))),
        "c_index": c_index,
        "target": cfg["target"],
        "ckpt_path": os.path.abspath(args.ckpt_path),
        "csv_path": os.path.abspath(args.csv_path),
    }
    with open(os.path.join(args.save_infer_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"推理完成：c-index={c_index:.4f}，结果保存到 {args.save_infer_dir}")
    print(f"  - 每患者 risk_score: {pred_csv}")
    return metrics


# ============================================================================
# 工具
# ============================================================================
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
    params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg["opt"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["reg"])
    return torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["reg"])


def read_csv_smart(path):
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def build_config(args):
    """组装超参数配置：CLI 默认 -> 若指定 --config 则用 json 覆盖 HPARAM_KEYS。"""
    cfg = {k: getattr(args, k) for k in HPARAM_KEYS if hasattr(args, k)}
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            override = json.load(f)
        for k, v in override.items():
            if k in HPARAM_KEYS:
                cfg[k] = v
        print(f"已从 {args.config} 覆盖超参数: "
              f"{[k for k in override if k in HPARAM_KEYS]}")
    return cfg


def get_args():
    p = argparse.ArgumentParser(description="MambaMIL 风格预后训练/推理脚本")
    p.add_argument("--mode", choices=["train", "infer"], default="train")

    # 数据 / 路径
    p.add_argument("--csv_path", type=str, required=True, help="输入 CSV（见 Example_Dataset_Csv.csv）")
    p.add_argument("--log_root", type=str, default="./logs", help="日志根目录（train）")
    p.add_argument("--exp_name", type=str, default="exp", help="实验名（train）")
    p.add_argument("--config", type=str, default=None, help="超参数 json：覆盖默认/提供推理所需模型超参")

    # 推理
    p.add_argument("--ckpt_path", type=str, default=None, help="权重路径（infer）")
    p.add_argument("--save_infer_dir", type=str, default=None, help="推理结果保存目录（infer）")

    # 训练目标与划分
    p.add_argument("--target", choices=["OS", "DFS", "RFS"], default="OS", help="训练目标")
    p.add_argument("--split_mode", choices=["kfold", "all_train"], default="kfold")
    p.add_argument("--k", type=int, default=5, help="k 折数量（默认 5）")

    # 生存/训练超参
    p.add_argument("--n_bins", type=int, default=4, help="离散时间区间数")
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--reg", type=float, default=1e-5, help="weight decay")
    p.add_argument("--drop_out", type=float, default=0.25)
    p.add_argument("--gc", type=int, default=16, help="梯度累积步数")
    p.add_argument("--alpha_surv", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    p.add_argument("--num_workers", type=int, default=2)

    # 模型
    p.add_argument("--model_type",
                   choices=["abmil", "mean_mil", "max_mil", "mamba_mil", "trans_mil", "s4model"],
                   default="abmil")
    p.add_argument("--in_dim", type=int, default=-1, help="特征维度，<=0 时从 h5 自动推断")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--max_slides_train", type=int, default=3,
                   help="训练时单患者最多拼接的 slide 数，超过则随机采样")
    p.add_argument("--feat_key", type=str, default="features",
                   help="h5 中特征数据集的键名（默认 features）")

    # 临床特征中期融合
    p.add_argument("--use_clinical", action=argparse.BooleanOptionalAction, default=True,
                   help="是否使用 CSV 中除 slide/标签外的临床信息（默认开启）")
    p.add_argument("--fusion_type", choices=["concat", "bilinear", "gated"], default="concat",
                   help="MIL 全局表征与临床嵌入的中期融合方式")
    p.add_argument("--clinical_hidden_dim", type=int, default=256,
                   help="临床 MLP 中间层维度（实际嵌入维 = max(32, hidden_dim//2)）")

    # MambaMIL 专用
    p.add_argument("--mambamil_layer", type=int, default=2)
    p.add_argument("--mambamil_rate", type=int, default=10)
    p.add_argument("--mambamil_type", choices=["Mamba", "BiMamba", "SRMamba"], default="SRMamba")

    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    cfg = build_config(args)

    if args.mode == "infer":
        assert args.ckpt_path and args.save_infer_dir, "推理需要 --ckpt_path 与 --save_infer_dir"
        assert args.config, "推理需要 --config 指定训练时保存的超参数 json"
        # 自动推断 in_dim（若 config 未给出有效值）
        if cfg.get("in_dim", -1) is None or cfg.get("in_dim", -1) <= 0:
            df = read_csv_smart(args.csv_path)
            cfg["in_dim"] = detect_in_dim(df["slide_feat_path"].astype(str).tolist(), cfg["feat_key"])
        set_seed(cfg["seed"])
        run_inference(cfg, device, args)
        return

    # ---- 训练 ----
    set_seed(cfg["seed"])
    df = read_csv_smart(args.csv_path)
    pt, n_classes, clinical_cols = build_patient_table(df, cfg["target"], cfg["n_bins"])
    cfg["n_classes"] = int(n_classes)
    if cfg["in_dim"] is None or cfg["in_dim"] <= 0:
        all_paths = [p for ps in pt["feat_paths"] for p in ps]
        cfg["in_dim"] = detect_in_dim(all_paths, cfg["feat_key"])
    print(f"患者数: {len(pt)}, 事件数: {int((pt['censorship'] == 0).sum())}, "
          f"特征维度: {cfg['in_dim']}, n_classes(时间区间): {cfg['n_classes']}, "
          f"临床列数: {len(clinical_cols)}, fusion: {cfg.get('fusion_type', 'concat')}")

    log_dir = os.path.join(args.log_root, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    # 保存超参数到 log 路径
    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"超参数已保存到 {os.path.join(log_dir, 'config.json')}")

    if cfg["split_mode"] == "kfold":
        train_kfold(pt, cfg, device, log_dir)
    else:
        train_all(pt, cfg, device, log_dir)


if __name__ == "__main__":
    main()
