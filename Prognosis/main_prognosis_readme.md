# main_prognosis.py 使用说明

基于 MambaMIL 预后逻辑的单脚本工具，支持**训练**（K 折 / 全量）与**推理**，并支持 **WSI 特征 + 临床信息的中期融合**。

核心建模逻辑：
- **离散时间生存建模**：将事件时间按未删失样本的分位数离散成 `n_bins` 个区间，模型输出每个区间的 hazard，`S = cumprod(1 - hazard)`，风险分 `risk = -∑S`。
- **损失**：NLL 生存损失（NLLSurvLoss）。
- **评价指标**：Harrell c-index（`event = 1 - censorship`）。优先使用 `sksurv`，其次 `lifelines`，都没有时自动回退到内置 numpy 实现，**无需额外安装依赖**。
- **多 slide 拼 bag**：一个患者（`case_id`）可能有多张 slide。训练时把多张 slide 的特征拼接成一个 bag，若 slide 数 `> max_slides_train` 则随机抽取该数量；**推理时拼接全部 slide**。
- **临床中期融合**（默认开启）：MIL 聚合器得到 slide 级全局表征后，与编码后的临床向量融合，再输入生存预测头。

---

## 一、输入数据格式

输入是一个 CSV，格式见 `Example_Dataset_Csv.csv`。

### 必需列

| 列名 | 说明 |
| --- | --- |
| `case_id` | 患者 ID（同一患者多张 slide 共用） |
| `slide_id` | 切片 ID |
| `slide_feat_path` | 该切片特征 `.h5` 文件的绝对路径 |
| `OS_month` / `OS_status` | 总生存：时间（月）/ 状态（1=死亡事件，0=删失） |
| `DFS_month` / `DFS_status` | 无病生存：时间 / 状态 |
| `RFS_month` / `RFS_status` | 无复发生存：时间 / 状态 |

约定：`*_status` 中 **1 = 事件发生（死亡/复发/进展），0 = 删失（censored）**；脚本内部 `censorship = 1 - status`。

### 临床/病理列（`--use_clinical` 默认开启时使用）

除 `case_id`、`slide_id`、`slide_feat_path` 以及 OS/DFS/RFS 的 month/status 列外，**其余列均作为临床特征**自动纳入模型，例如：

| 类型 | 示例列 |
| --- | --- |
| 数值 | `Age`、`ER_pct_pre`、`Stage`、`MP`、`Tils` 等 |
| 二值 0/1 | `ER_status`、`PR_status`、`Fibrosis`、`Necrosis` 等 |
| 类别 | `HER2_score_pre`（如 `1+`）、`Molecular_subtype`（如 `LuminalB-`）等 |

**编码规则**（在训练集上拟合，验证/推理复用）：
- **数值列**：按训练集均值/标准差标准化；缺失值用训练集均值填充。
- **类别列**：one-hot 编码；缺失值归为 `missing` 类。
- 同一 `case_id` 多行 slide 时，临床信息取该患者第一条记录（各 slide 应一致）。

`.h5` 文件中特征数据集默认键名为 `features`，形状为 `[N_patch, dim]`（可用 `--feat_key` 修改）。

---

## 二、模型结构（中期融合）

```
slide bag [N, in_dim]
    ↓
MIL 聚合器 (abmil / mean_mil / max_mil)
    ↓
全局表征 [1, hidden_dim]
    ↓                          临床向量 [clinical_in_dim]
    ↓                                ↓
    └──────── fusion_type ──── 临床 MLP → 临床嵌入
                    ↓
              融合表征 [1, hidden_dim]
                    ↓
              生存 hazard 头 → n_classes
```

### 融合方式（`--fusion_type`）

| 值 | 说明 |
| --- | --- |
| `concat`（默认） | 拼接 `[全局表征; 临床嵌入]` 后经 MLP |
| `bilinear` | 双线性层 `Bilinear(全局, 临床)` + LayerNorm |
| `gated` | 门控融合：`gate * proj(全局) + (1-gate) * proj(临床)` |

关闭临床融合时使用 `--no-use_clinical`，模型退化为纯 WSI MIL 预后。

> **注意**：临床中期融合目前仅支持内置模型 `abmil` / `mean_mil` / `max_mil`。`mamba_mil` / `trans_mil` / `s4model` 仅支持 `--no-use_clinical` 的纯 path 模式。

---

## 三、超参数说明

### 运行 / 路径

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--mode` | `train` | 运行模式：`train` 训练 / `infer` 推理 |
| `--csv_path` | （必填） | 输入 CSV 路径 |
| `--log_root` | `./logs` | 训练日志根目录 |
| `--exp_name` | `exp` | 实验名，日志写到 `log_root/exp_name/` |
| `--config` | `None` | 超参数 json：训练时用于**覆盖超参**；推理时用于**提供模型结构超参** |

### 训练目标与数据划分

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--target` | `OS` | 训练目标，三选一：`OS` / `DFS` / `RFS` |
| `--split_mode` | `kfold` | `kfold` K 折交叉验证 / `all_train` 全量训练 |
| `--k` | `5` | K 折折数（按 `case_id` 划分） |

### 生存 / 训练超参

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--n_bins` | `4` | 离散时间区间数（`n_classes`） |
| `--max_epochs` | `50` | 每个 run 的训练轮数 |
| `--lr` | `1e-4` | 学习率 |
| `--reg` | `1e-5` | 权重衰减 |
| `--drop_out` | `0.25` | dropout 比例 |
| `--gc` | `16` | 梯度累积步数 |
| `--alpha_surv` | `0.0` | NLL 生存损失中未删失样本加权 |
| `--seed` | `1` | 随机种子 |
| `--opt` | `adam` | 优化器：`adam` / `sgd` |
| `--num_workers` | `2` | DataLoader 进程数 |

### 模型（MIL）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model_type` | `abmil` | `abmil` / `mean_mil` / `max_mil`（支持临床融合）；`mamba_mil` 等仅 path-only |
| `--in_dim` | `-1` | WSI 特征维度，<=0 时从 h5 自动推断 |
| `--hidden_dim` | `512` | MIL 全局表征维度；临床嵌入维度 = `max(32, hidden_dim//2)` |
| `--max_slides_train` | `3` | 训练时单患者最多拼接 slide 数 |
| `--feat_key` | `features` | h5 特征键名 |

### 临床中期融合

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--use_clinical` | 开启 | 使用 CSV 临床列；加 `--no-use_clinical` 关闭 |
| `--fusion_type` | `concat` | 融合方式：`concat` / `bilinear` / `gated` |
| `--clinical_hidden_dim` | `256` | 预留参数（当前实现中临床嵌入维由 `hidden_dim` 决定） |
| `clinical_in_dim` | 自动 | 编码后临床向量总维度，训练时写入 `config.json`，无需手动指定 |

### MambaMIL 专用（仅 `--model_type mamba_mil` 且 `--no-use_clinical`）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--mambamil_layer` | `2` | Mamba 层数 |
| `--mambamil_rate` | `10` | SRMamba 的 rate |
| `--mambamil_type` | `SRMamba` | `Mamba` / `BiMamba` / `SRMamba` |

> **超参数 json 覆盖规则**：训练时解析出的超参会保存为 `log_root/exp_name/config.json`。若通过 `--config xxx.json` 指定，则其中的超参字段会**覆盖**命令行默认值（路径/模式参数仍以命令行为准）。

---

## 四、使用方式

> 下文用 `PY=/home/chenwm/anaconda3/envs/pfm_seg/bin/python` 指代 Python 解释器。

### 1. K 折 + 临床融合（默认 concat）

```bash
$PY main_prognosis.py --mode train --split_mode kfold --target OS \
    --csv_path Example_Dataset_Csv.csv \
    --log_root ./logs --exp_name os_pathomic_concat \
    --fusion_type concat --k 5 --max_epochs 50
```

### 2. 指定融合方式为 gated / bilinear

```bash
$PY main_prognosis.py --mode train --target DFS \
    --csv_path Example_Dataset_Csv.csv \
    --log_root ./logs --exp_name dfs_gated \
    --fusion_type gated --model_type abmil
```

### 3. 仅 WSI，不使用临床信息

```bash
$PY main_prognosis.py --mode train --target OS \
    --csv_path Example_Dataset_Csv.csv \
    --log_root ./logs --exp_name os_path_only \
    --no-use_clinical
```

### 4. 全量训练（all_train）

```bash
$PY main_prognosis.py --mode train --split_mode all_train --target DFS \
    --csv_path Example_Dataset_Csv.csv \
    --log_root ./logs --exp_name dfs_all --max_epochs 50
```

### 5. 用 json 覆盖超参数

```bash
$PY main_prognosis.py --mode train --target RFS \
    --csv_path Example_Dataset_Csv.csv \
    --log_root ./logs --exp_name rfs_kfold \
    --config my_hparams.json
```

`my_hparams.json` 示例：

```json
{
    "lr": 5e-5,
    "max_epochs": 80,
    "drop_out": 0.3,
    "n_bins": 4,
    "model_type": "abmil",
    "fusion_type": "bilinear",
    "use_clinical": true,
    "max_slides_train": 3
}
```

### 6. 推理

推理需要：
- 训练保存的 `config.json`（含 `use_clinical`、`fusion_type`、`clinical_in_dim` 等）
- 与 checkpoint **同目录**下的 `clinical_encoder.json`（K 折时在各 `fold_i/` 下）
- 与训练同格式的 CSV（含临床列）
- 权重路径与结果目录

```bash
$PY main_prognosis.py --mode infer \
    --config ./logs/os_pathomic_concat/config.json \
    --ckpt_path ./logs/os_pathomic_concat/fold_0/checkpoint_best.pt \
    --csv_path test.csv \
    --save_infer_dir ./infer_os
```

> 推理时会自动从 `checkpoint_best.pt` 所在目录加载 `clinical_encoder.json`。若找不到且 `use_clinical=true`，将回退为不使用临床特征。

---

## 五、输出结果

### 训练（kfold）

```
log_root/exp_name/
├── config.json                # 全部超参数（含 fusion_type / clinical_in_dim）
├── kfold_summary.json         # k 个 best_epoch、各折/均值 val c-index
├── fold_0/
│   ├── clinical_encoder.json  # 该折训练集拟合的临床编码器（数值均值/方差、类别词表）
│   ├── metrics.csv            # 逐 epoch: train_loss / train_cindex / val_loss / val_cindex
│   ├── history.json
│   ├── checkpoint_best.pt     # val c-index 最优权重
│   └── checkpoint_last.pt
├── fold_1/ ...
└── fold_{k-1}/ ...
```

### 训练（all_train）

```
log_root/exp_name/
├── config.json
├── clinical_encoder.json      # 全量训练集拟合的临床编码器
├── all_train_summary.json
├── metrics.csv / history.json
├── checkpoint_best_loss.pt
└── checkpoint_last.pt
```

### 推理

```
save_infer_dir/
├── metrics.json               # n_cases / n_events / c_index / target 等
└── risk_scores.csv            # case_id, risk_score, event_time, censorship, status
```

---

## 六、依赖说明

- 必需：`torch`、`h5py`、`pandas`、`numpy`、`scikit-learn`。
- 可选：`sksurv` 或 `lifelines`（用于 c-index）；若均未安装，脚本自动使用内置 numpy 实现。
- 临床融合 + 内置 MIL（`abmil` 等）无需额外依赖。
- `mamba_mil` / `trans_mil` / `s4model` 需要同级 `MambaMIL` 仓库及其依赖，且仅支持 `--no-use_clinical`。
