"""训练折临床编码与 WSI/临床表征融合。

先用 ``get_clinical_columns(raw_csv, cfg)`` 固定可用列，再只在训练折调用
``ClinicalEncoder.fit``。验证及推理复用保存的均值、标准差和类别词表。
结局、标识符、路径及风险公式已有的结构输入不能作为额外临床特征。
关闭淋巴结分支时还排除已知的淋巴结专属临床信息。
"""

import json
import math
import numbers

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


KNOWN_CATEGORICAL = {"her2_score_pre", "molecular_subtype"}
SURVIVAL_COLS = {
    f"{target}_{suffix}"
    for target in ("os", "dfs", "rfs")
    for suffix in ("month", "status")
}
BASE_EXCLUDED = {
    "case_id", "slide_id", "slide_feat_path", "tumor_slide_id",
    "tumor_slide_feat_path", "lymph_node_slide_id", "lymph_node_slide_feat_path",
    "node_slide_id", "node_slide_feat_path", "feat_paths", "event_time",
    "censorship", "disc_label", "status", "has_met", "n_met", "tumor_size",
    "n_metastatic_lymph_nodes", "length", "number_of_lymph_node_metastases",
    "ln_count", "node_count", "n_positive_nodes", "positive_lymph_nodes",
    "n_positive_lymph_nodes", "num_positive_nodes", "number_of_positive_lymph_nodes",
    "size_norm", "tumor_size_norm", "count_norm", "ln_count_norm", "rcb",
    "risk_score", "rcb_index", "surv_risk", "tumor_score", "lymph_score",
    "tumor_logit", "lymph_logit", "tumor_log_score", "lymph_log_score",
} | SURVIVAL_COLS
LYMPH_CLINICAL_COLS = {
    "maximum_diameter_of_lymph_node_metastasis", "lymph_node_status",
    "lymph_node_metastasis", "lymph_node_metastases", "node_positive",
    "n_positive_nodes", "positive_lymph_nodes", "lymph_node_count",
}


def _column_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = list(value)
    columns = [str(col).strip() for col in values if str(col).strip()]
    if len(columns) != len(set(columns)):
        raise ValueError("临床列配置包含重复列名。")
    return columns


def _is_excluded(column, cfg):
    name = str(column).strip().lower()
    custom = {
        str(cfg[key]).strip().lower()
        for key in ("tumor_feat_col", "ln_feat_col", "tumor_size_col", "ln_count_col")
        if cfg.get(key)
    }
    if name in BASE_EXCLUDED | custom or name.endswith(("_id", "_path")):
        return True
    if not cfg.get("use_lymphnode", True):
        return name in LYMPH_CLINICAL_COLS or "lymph" in name or "淋巴结" in name
    return False


def get_clinical_columns(df, cfg):
    """返回临床列；显式选择禁止列/缺失列时报错，自动模式过滤这些列。"""
    if not cfg.get("use_clinical", True):
        return []
    selected = _column_list(cfg.get("clinical_cols"))
    if selected is None:
        selected = [col for col in df.columns if not _is_excluded(col, cfg)]
    else:
        forbidden = [col for col in selected if _is_excluded(col, cfg)]
        if forbidden:
            raise ValueError(f"临床特征禁止包含标识符、路径、结局或已禁用的结构信息: {forbidden}")
        missing = [col for col in selected if col not in df.columns]
        if missing:
            raise KeyError(f"CSV 缺少指定临床列: {missing}")
    if not selected:
        raise ValueError("已启用临床融合，但没有可用临床列；请设置 --clinical_cols。")
    return selected


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return bool(pd.isna(value))


def _category(value):
    if _is_missing(value):
        return None
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return format(numeric, ".17g")
    return str(value).strip()


def _unused_token(base, observed):
    token = base
    while token in observed:
        token += "_"
    return token


class ClinicalEncoder:
    """训练折数值均值填充/标准化与固定词表 one-hot。

    类别的缺失值和推理时的新类别各有独立维度。数值无穷按缺失处理，
    全缺失/单样本/常量列的标准差使用 1；类别编码维度在推理时不扩张。
    """

    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.numeric_mean = {}
        self.numeric_std = {}
        self.cat_categories = {}
        self.missing_tokens = {}
        self.unknown_tokens = {}
        self.output_dim = 0
        self.fitted = False

    def fit(self, df, clinical_cols=None, categorical_cols=None):
        if len(df) == 0:
            raise ValueError("不能用空训练折拟合临床编码器。")
        selected = get_clinical_columns(df, {
            "use_clinical": True, "use_lymphnode": True,
            "clinical_cols": clinical_cols,
        })
        forced = _column_list(categorical_cols) or []
        unknown_forced = [col for col in forced if col not in selected]
        if unknown_forced:
            raise ValueError(f"指定类别列没有包含在临床特征中: {unknown_forced}")
        self.__init__()
        for col in selected:
            categorical = (
                col in forced or str(col).lower() in KNOWN_CATEGORICAL
                or not pd.api.types.is_numeric_dtype(df[col].dtype)
            )
            if categorical:
                self.categorical_cols.append(col)
            else:
                self.numeric_cols.append(col)
        for col in self.numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce").astype(float)
            vals = vals.replace([np.inf, -np.inf], np.nan)
            mean = float(vals.mean()) if vals.notna().any() else 0.0
            std = float(vals.std()) if vals.notna().any() else 1.0
            if not math.isfinite(mean):
                mean = 0.0
            if not math.isfinite(std) or std < 1e-6:
                std = 1.0
            self.numeric_mean[col] = mean
            self.numeric_std[col] = std
        for col in self.categorical_cols:
            observed = {value for raw in df[col] if (value := _category(raw)) is not None}
            missing = _unused_token("__MISSING__", observed)
            unknown = _unused_token("__UNKNOWN__", observed | {missing})
            self.missing_tokens[col] = missing
            self.unknown_tokens[col] = unknown
            self.cat_categories[col] = sorted(observed) + [missing, unknown]
        self.output_dim = len(self.numeric_cols) + sum(
            len(self.cat_categories[col]) for col in self.categorical_cols
        )
        self.fitted = True
        return self

    def transform_row(self, row):
        if not self.fitted:
            raise RuntimeError("临床编码器尚未拟合或加载。")
        features = []
        for col in self.numeric_cols:
            value = pd.to_numeric(row.get(col, np.nan), errors="coerce")
            value = float(value) if not _is_missing(value) else self.numeric_mean[col]
            if not math.isfinite(value):
                value = self.numeric_mean[col]
            features.append((value - self.numeric_mean[col]) / self.numeric_std[col])
        for col in self.categorical_cols:
            value = _category(row.get(col, None))
            if value is None:
                value = self.missing_tokens[col]
            elif value not in self.cat_categories[col]:
                value = self.unknown_tokens[col]
            features.extend(1.0 if value == category else 0.0
                            for category in self.cat_categories[col])
        # float64 标准化值转为 float32 时也保持有限；不改变常规临床范围。
        limit = np.finfo(np.float32).max
        result = np.nan_to_num(np.asarray(features, dtype=np.float64),
                               nan=0.0, posinf=limit, neginf=-limit)
        return np.clip(result, -limit, limit).astype(np.float32)

    def transform_df(self, df):
        if not self.fitted:
            raise RuntimeError("临床编码器尚未拟合或加载。")
        if len(df) == 0:
            return np.empty((0, self.output_dim), dtype=np.float32)
        return np.stack([self.transform_row(df.iloc[i]) for i in range(len(df))], axis=0)

    def to_dict(self):
        if not self.fitted:
            raise RuntimeError("不能保存尚未拟合的临床编码器。")
        return {
            "schema_version": 1,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "cat_categories": self.cat_categories,
            "missing_tokens": self.missing_tokens,
            "unknown_tokens": self.unknown_tokens,
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("schema_version", 1) != 1:
            raise ValueError("不支持的临床编码器版本。")
        encoder = cls()
        encoder.numeric_cols = list(data["numeric_cols"])
        encoder.categorical_cols = list(data["categorical_cols"])
        encoder.numeric_mean = {col: float(value) for col, value in data["numeric_mean"].items()}
        encoder.numeric_std = {col: float(value) for col, value in data["numeric_std"].items()}
        encoder.cat_categories = {col: list(values) for col, values in data["cat_categories"].items()}
        encoder.missing_tokens = dict(data["missing_tokens"])
        encoder.unknown_tokens = dict(data["unknown_tokens"])
        for col in encoder.numeric_cols:
            if (not math.isfinite(encoder.numeric_mean[col])
                    or not math.isfinite(encoder.numeric_std[col]) or encoder.numeric_std[col] <= 0):
                raise ValueError(f"临床编码器的数值统计无效: {col}")
        for col in encoder.categorical_cols:
            cats = encoder.cat_categories[col]
            if (len(cats) != len(set(cats)) or encoder.missing_tokens[col] not in cats
                    or encoder.unknown_tokens[col] not in cats
                    or encoder.missing_tokens[col] == encoder.unknown_tokens[col]):
                raise ValueError(f"临床编码器的类别词表无效: {col}")
        encoder.output_dim = len(encoder.numeric_cols) + sum(
            len(encoder.cat_categories[col]) for col in encoder.categorical_cols
        )
        if encoder.output_dim != int(data["output_dim"]):
            raise ValueError("临床编码器 output_dim 与实际词表不一致。")
        encoder.fitted = True
        return encoder


def save_clinical_encoder(encoder, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(encoder.to_dict(), handle, ensure_ascii=False, indent=2, allow_nan=False)


def load_clinical_encoder(path):
    with open(path, "r", encoding="utf-8") as handle:
        return ClinicalEncoder.from_dict(json.load(handle))


def _initialize_fusion(module):
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Bilinear)):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            if layer.elementwise_affine:
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)


class ConcatFusion(nn.Module):
    def __init__(self, mil_dim, clin_dim, out_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(mil_dim + clin_dim, out_dim),
                                 nn.ReLU(), nn.Dropout(dropout))
        _initialize_fusion(self)

    def forward(self, mil_repr, clin_repr):
        return self.net(torch.cat([mil_repr, clin_repr], dim=-1))


class BilinearFusion(nn.Module):
    def __init__(self, mil_dim, clin_dim, out_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(mil_dim, clin_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        _initialize_fusion(self)

    def forward(self, mil_repr, clin_repr):
        return self.norm(F.relu(self.bilinear(mil_repr, clin_repr)))


class GatedFusion(nn.Module):
    def __init__(self, mil_dim, clin_dim, out_dim):
        super().__init__()
        self.proj_mil = nn.Linear(mil_dim, out_dim)
        self.proj_clin = nn.Linear(clin_dim, out_dim)
        self.gate = nn.Sequential(nn.Linear(mil_dim + clin_dim, out_dim), nn.Sigmoid())
        _initialize_fusion(self)

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
