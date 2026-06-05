# main_prognosis.py 使用说明

基于 MambaMIL 预后逻辑的单脚本工具，支持**训练**（K 折 / 全量）与**推理**。

核心建模逻辑：
- **离散时间生存建模**：将事件时间按未删失样本的分位数离散成 `n_bins` 个区间，模型输出每个区间的 hazard，`S = cumprod(1 - hazard)`，风险分 `risk = -∑S`。
- **损失**：NLL 生存损失（NLLSurvLoss）。
- **评价指标**：Harrell c-index（`event = 1 - censorship`）。优先使用 `sksurv`，其次 `lifelines`，都没有时自动回退到内置 numpy 实现，**无需额外安装依赖**。
- **多 slide 拼 bag**：一个患者（`case_id`）可能有多张 slide。训练时把多张 slide 的特征拼接成一个 bag，若 slide 数 `> max_slides_train` 则随机抽取该数量；**推理时拼接全部 slide**。

---

## 一、输入数据格式

输入是一个 CSV，格式见 `Example_Dataset_Csv.csv`。关键列：

| 列名 | 说明 |
| --- | --- |
| `case_id` | 患者 ID（同一患者多张 slide 共用） |
| `slide_id` | 切片 ID |
| `slide_feat_path` | 该切片特征 `.h5` 文件的绝对路径 |
| `OS_month` / `OS_status` | 总生存：时间（月）/ 状态（1=死亡事件，0=删失） |
| `DFS_month` / `DFS_status` | 无病生存：时间 / 状态 |
| `RFS_month` / `RFS_status` | 无复发生存：时间 / 状态 |
| 其他列 | 临床/病理特征，当前脚本不使用，可保留 |

约定：`*_status` 中 **1 = 事件发生（死亡/复发/进展），0 = 删失（censored）**；脚本内部 `censorship = 1 - status`。

`.h5` 文件中特征数据集默认键名为 `features`，形状为 `[N_patch, dim]`（可用 `--feat_key` 修改）。

---

## 二、超参数说明

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
| `--target` | `OS` | 训练目标，三选一：`OS` / `DFS` / `RFS`，自动选用对应 `*_month` / `*_status` 列 |
| `--split_mode` | `kfold` | `kfold` K 折交叉验证 / `all_train` 全量训练 |
| `--k` | `5` | K 折折数（按 `case_id` 划分，避免同一患者跨折泄漏） |

### 生存 / 训练超参

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--n_bins` | `4` | 离散时间区间数（即模型输出类别数 `n_classes`） |
| `--max_epochs` | `50` | 每个 run 的训练轮数 |
| `--lr` | `1e-4` | 学习率 |
| `--reg` | `1e-5` | 权重衰减（weight decay） |
| `--drop_out` | `0.25` | dropout 比例 |
| `--gc` | `16` | 梯度累积步数（batch 固定为 1，bag 大小可变） |
| `--alpha_surv` | `0.0` | NLL 生存损失中未删失样本的加权系数 |
| `--seed` | `1` | 随机种子 |
| `--opt` | `adam` | 优化器：`adam` / `sgd` |
| `--num_workers` | `2` | DataLoader 进程数 |

### 模型

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model_type` | `abmil` | `abmil` / `mean_mil` / `max_mil`（内置，开箱即用）；`mamba_mil` / `trans_mil` / `s4model`（从同级 `MambaMIL` 仓库动态加载，需相应依赖，如编译好的 `mamba_ssm`） |
| `--in_dim` | `-1` | 特征维度，`<=0` 时从首个可用 `.h5` 自动推断 |
| `--hidden_dim` | `512` | 隐藏层维度 |
| `--max_slides_train` | `3` | 训练时单患者最多拼接的 slide 数，超过则随机采样 |
| `--feat_key` | `features` | `.h5` 中特征数据集的键名 |

### MambaMIL 专用（仅 `--model_type mamba_mil` 时生效）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--mambamil_layer` | `2` | Mamba 层数 |
| `--mambamil_rate` | `10` | SRMamba 的 rate |
| `--mambamil_type` | `SRMamba` | `Mamba` / `BiMamba` / `SRMamba` |

> **超参数 json 覆盖规则**：训练时解析出的超参会保存为 `log_root/exp_name/config.json`。若通过 `--config xxx.json` 指定，则其中的超参字段会**覆盖**命令行默认值（仅覆盖超参类字段；`csv_path` / `log_root` / `exp_name` / `mode` 等路径与模式参数仍以命令行为准）。

---

## 三、使用方式

> 下文用 `PY=/home/chenwm/anaconda3/envs/pfm_seg/bin/python` 指代 Python 解释器。

### 1. K 折交叉验证训练

```bash
$PY main_prognosis.py --mode train --split_mode kfold --target OS \
    --csv_path Example_Dataset_Csv.csv \
    --log_root ./logs --exp_name os_kfold \
    --k 5 --max_epochs 50
```

### 2. 全量训练（all_train）

```bash
$PY main_prognosis.py --mode train --split_mode all_train --target DFS \
    --csv_path Example_Dataset_Csv.csv \
    --log_root ./logs --exp_name dfs_all --max_epochs 50
```

### 3. 用 json 覆盖超参数

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
    "max_slides_train": 3
}
```

### 4. 推理

推理需要训练时保存的 `config.json`（提供模型结构超参）、权重路径、与训练同格式的 CSV，以及结果保存目录：

```bash
$PY main_prognosis.py --mode infer \
    --config ./logs/os_kfold/config.json \
    --ckpt_path ./logs/os_kfold/fold_0/checkpoint_best.pt \
    --csv_path test.csv \
    --save_infer_dir ./infer_os
```

---

## 四、输出结果

### 训练（kfold）

```
log_root/exp_name/
├── config.json                # 本次训练的全部超参数
├── kfold_summary.json         # k 个 best_epoch、各折/均值 val c-index
├── fold_0/
│   ├── metrics.csv            # 逐 epoch: train_loss / train_cindex / val_loss / val_cindex
│   ├── history.json           # 同上（json 格式）
│   ├── checkpoint_best.pt     # 该折 val c-index 最优权重
│   └── checkpoint_last.pt     # 该折最后一轮权重
├── fold_1/ ...
└── fold_{k-1}/ ...
```

`kfold_summary.json` 关键字段：`best_epochs`（k 个最佳 epoch）、`val_cindex_per_fold`、`mean_val_cindex`、`std_val_cindex`。

### 训练（all_train）

```
log_root/exp_name/
├── config.json
├── all_train_summary.json     # 最低 train loss 的 epoch 及 loss
├── metrics.csv / history.json # 逐 epoch loss / c-index
├── checkpoint_best_loss.pt    # train loss 最低的权重
└── checkpoint_last.pt         # 最后一轮权重
```

### 推理

```
save_infer_dir/
├── metrics.json               # n_cases / n_events / c_index / target 等
└── risk_scores.csv            # 每个患者: case_id, risk_score, event_time, censorship, status
```

---

## 五、依赖说明

- 必需：`torch`、`h5py`、`pandas`、`numpy`、`scikit-learn`。
- 可选：`sksurv` 或 `lifelines`（用于 c-index）；若均未安装，脚本自动使用内置 numpy 实现，结果一致。
- `mamba_mil` / `trans_mil` / `s4model` 需要同级 `MambaMIL` 仓库及其依赖（如编译好的 `mamba_ssm`）；默认 `abmil` 无需这些依赖即可运行。
