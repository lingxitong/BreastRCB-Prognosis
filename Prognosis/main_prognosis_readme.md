# 原发灶 + 淋巴结预后（RCB 风格）

本分支从 `main` 的离散时间生存训练改来。每个患者只用四样输入：

- 一张原发肿瘤切片的特征
- 肿瘤长径
- 一张淋巴结切片的特征
- 转移淋巴结个数

模型先给两张切片各打一个 0 到 1 的分，再套进下面这个可训练公式。没有转移时，第二项直接为 0。

---

## 一、RCB 原理，以及这里怎么对应

乳腺残余肿瘤负荷（Residual Cancer Burden, RCB；Symmans 等, JCO 2007）用一个连续指数描述新辅助治疗后的残余病灶，并据此做远处复发风险分层。MD Anderson 计算器里的形式是：

```
RCB = 1.4 * (f_inv * d_prim) ^ 0.17
    + [4 * (1 - 0.75 ^ LN) * d_met] ^ 0.17
```

| 符号 | 含义 |
| --- | --- |
| `d_prim` | 残余瘤床两条最大径的几何平均，描述原发灶大小 |
| `f_inv` | 瘤床中浸润癌所占比例 |
| `LN` | 阳性淋巴结个数。`LN = 0` 时 `1 - 0.75^LN = 0`，淋巴结项消失 |
| `d_met` | 最大淋巴结转移灶直径 |
| `1.4`、`0.17` | 用 Cox 模型估出来的尺度和压缩指数 |

官方分档（仅经典 RCB，本模型不套用这些切点）：

| 分级 | 指数 |
| --- | --- |
| RCB-0 | 0，乳腺和淋巴结都无残余浸润癌 |
| RCB-I | 0 < RCB ≤ 1.36 |
| RCB-II | 1.36 < RCB ≤ 3.28 |
| RCB-III | RCB > 3.28 |

本分支按同样的「原发灶项 + 淋巴结项」来设计，但切片上的比例/负荷不再手工估计，而由网络从 WSI 特征里学；四个权重也改为端到端训练：

```
R = a * (tumor_size_norm * tumor_slide_score) ^ b
  + c * (ln_count_norm * lymph_slide_score) ^ d
```

| 本模型 | 对应的 RCB 角色 |
| --- | --- |
| `tumor_size` | 原发灶大小（这里用长径，不再用双径几何平均） |
| `tumor_slide_score` ∈ (0, 1) | 原发灶切片评分，对应浸润比例 `f_inv` |
| `n_metastatic_lymph_nodes` | 阳性淋巴结个数 |
| `lymph_slide_score` ∈ (0, 1) | 淋巴结切片评分，对应转移负荷 |
| `a, b, c, d` | 可训练权重。`a`、`c` 为正，`b`、`d` 在 (0, 1) |
| 转移个数 = 0 | 第二项强制为 0，淋巴结网络不参与该样本 |

`tumor_size_norm`、`ln_count_norm` 是用**当前训练折**的均值去除的尺度（有转移者的个数均值）。这只是数值稳定，不改变 `a * (x)^b` 能表示的函数族。尺度写在该折目录的 `input_scaler.json`，推理必须沿用同一份。

约束与初值：

- `a = softplus(a_raw)`，`c = softplus(c_raw)`，初值 1.4
- `b = sigmoid(b_raw)`，`d = sigmoid(d_raw)`，初值 0.17
- 指数限制在 (0, 1)，避免出现「肿瘤越大风险越低」或指数过大导致溢出

因此本模型的 `R` 与官方 RCB 不是同一个数，不能直接套 RCB-I/II/III 的切点。`R` 越大表示风险越高，用它计算 c-index。

---

## 二、网络与训练

```
原发肿瘤 patch 特征 [N, dim]
        │
   MIL（abmil / mean / max）
        │
 tumor_slide_score ∈ (0, 1) ──┐
                               │
 肿瘤长径 / 训练集尺度 ────────┼──► 第一项 a * (·) ^ b
                               │
转移个数 = 0 ? ──是──► 第二项 = 0
        │否
淋巴结 patch 特征 [M, dim]
        │
   另一套 MIL（不与原发灶共享参数）
        │
 lymph_slide_score ∈ (0, 1) ──┴──► 第二项 c * (·) ^ d

R = 第一项 + 第二项
logits_k = softplus(scale) * R + bias_k
hazard = sigmoid(logits)
```

评分头默认先对 MIL 表征做无可训练参数的 LayerNorm，并用小权重初始化，降低输入特征尺度导致 sigmoid 过早饱和的风险。公式使用 `log_score = logsigmoid(score_logit)` 计算 `exp(b * (log(size_norm) + log_score))`，数学上仍为同一幂函数，不再把 `size_norm * score` 强制截到 `1e-4`。

训练前，读出头的各时间箱 baseline 用**当前训练折**的事件数和在险人数做平滑初始化，并扣除评分约 0.5 时的平均 R。初始正读出系数为 0.1，仍参与训练。初始化结果保存为 `survival_head_init.json`。

- 读出头的 `scale` 经过 softplus，恒为正。所以 `R` 升高时，每个时间箱的 hazard 都升高，生存排序与 `R` 的排序一致。
- 损失是离散时间 NLL；从 logits 直接计算 log-hazard 和累计 log-survival，避免概率连乘下溢和硬截断。正常数值区域与原损失及梯度等价。时间按未删失样本分位数切成 `n_bins` 段。
- 评价指标是 Harrell c-index，风险分用 `R` 本身，不用别的融合向量。优先 `sksurv`，其次 `lifelines`，都没有则用脚本内的 numpy 实现。
- `a/b/c/d` 和读出头不做 weight decay，避免把公式系数往 0 上拉。MIL 编码器仍使用 `--reg`。
- 原发灶和淋巴结是两套编码器。两张切片的特征维必须相同。

---

## 三、CSV 列

脚本读 CSV，不直接读 xlsx。把 `dataset.xlsx` 导出成 CSV 后，列名与 `Example_Dataset_Csv.csv`、`Feature_dict.json` 一致即可。一个患者一行。

| 列名 | 必填 | 说明 |
| --- | --- | --- |
| `case_id` | 是 | 患者 ID |
| `tumor_slide_id` | 否 | 原发肿瘤切片 ID，脚本不读，便于核对 |
| `tumor_slide_feat_path` | 是 | 原发肿瘤 `.h5` 绝对路径 |
| `tumor_size` | 是 | 肿瘤长径，必须 > 0 |
| `lymph_node_slide_id` | 否 | 淋巴结切片 ID |
| `lymph_node_slide_feat_path` | 有转移时必填 | 淋巴结 `.h5`。个数为 0 时留空 |
| `n_metastatic_lymph_nodes` | 是 | 转移淋巴结个数，≥ 0。为 0 时第二项为 0 |
| `OS_month` / `OS_status` | 训练 OS 时必填 | 时间（月）；1 = 死亡，0 = 删失 |
| `DFS_month` / `DFS_status` | 训练 DFS 时必填 | 同上 |
| `RFS_month` / `RFS_status` | 训练 RFS 时必填 | 同上 |

同一 `case_id` 若有多行，只用第一行，并打印警告。

`.h5` 默认键名 `features`，形状 `[N_patch, dim]`，可用 `--feat_key` 修改。

列名不一致时用参数改，不必改表头：

```bash
--tumor_feat_col tumor_slide_feat_path \
--ln_feat_col lymph_node_slide_feat_path \
--tumor_size_col tumor_size \
--ln_count_col n_metastatic_lymph_nodes
```

若表里沿用旧字段，可写成 `--tumor_size_col Length --ln_count_col Number_of_lymph_node_metastases`，但原发灶/淋巴结特征路径仍要各自单独一列，不能再把多张切片堆在 `slide_feat_path` 里。

---

## 四、命令

下文用 `python` 指当前环境的解释器。在 `Prognosis/` 目录下执行。

### K 折

```bash
python main_prognosis.py --mode train --split_mode kfold --target OS \
    --csv_path /path/to/dataset.csv \
    --log_root ./logs --exp_name os_lymph_rcb \
    --model_type abmil --k 5 --max_epochs 50
```

### 全量训练

```bash
python main_prognosis.py --mode train --split_mode all_train --target DFS \
    --csv_path /path/to/dataset.csv \
    --log_root ./logs --exp_name dfs_lymph_all
```

### 用 json 覆盖超参

```bash
python main_prognosis.py --mode train --target RFS \
    --csv_path /path/to/dataset.csv \
    --log_root ./logs --exp_name rfs_lymph \
    --config my_hparams.json
```

```json
{
    "lr": 5e-5,
    "max_epochs": 80,
    "n_bins": 4,
    "model_type": "abmil",
    "hidden_dim": 512,
    "drop_out": 0.25
}
```

### 推理

推理需要三样东西放在一起：`config.json`、该折目录里的 `input_scaler.json`、同一目录的权重。

```bash
python main_prognosis.py --mode infer \
    --config ./logs/os_lymph_rcb/config.json \
    --ckpt_path ./logs/os_lymph_rcb/fold_0/checkpoint_best.pt \
    --csv_path /path/to/test.csv \
    --save_infer_dir ./infer_os
```

脚本会到权重所在目录读取 `input_scaler.json`。换一份训练集的尺度会使 `R` 不可比。

本修复保存的 `config.json` 包含 `numerics_version=2`。由于评分归一化、低评分区域计算及初始化已经改变，请重新训练新版权重；旧权重不能与新版 config 混用。旧版实验需使用原代码复现，新版推理会拒绝没有对应版本标记的旧 config。

---

## 五、超参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--mode` | `train` | `train` 或 `infer` |
| `--csv_path` | 必填 | 输入 CSV |
| `--log_root` / `--exp_name` | `./logs` / `exp` | 日志写到 `log_root/exp_name/` |
| `--config` | 空 | 训练时覆盖超参；推理时提供结构超参 |
| `--target` | `OS` | `OS` / `DFS` / `RFS` |
| `--split_mode` | `kfold` | `kfold` 或 `all_train` |
| `--k` | `5` | 按患者划分 |
| `--n_bins` | `4` | 离散时间箱数 |
| `--max_epochs` | `50` | 轮数 |
| `--lr` | `1e-4` | 学习率 |
| `--reg` | `1e-5` | 只作用在 MIL 权重上 |
| `--drop_out` | `0.25` | MIL dropout |
| `--gc` | `16` | 梯度累积步数。batch 固定为 1 |
| `--alpha_surv` | `0.0` | NLL 中未删失项的额外权重 |
| `--seed` | `1` | 随机种子 |
| `--opt` | `adam` | `adam` 或 `sgd` |
| `--num_workers` | `2` | DataLoader 进程数 |
| `--model_type` | `abmil` | `abmil` / `mean_mil` / `max_mil` |
| `--in_dim` | `-1` | ≤0 时从 h5 推断 |
| `--hidden_dim` | `512` | MIL 隐层维度 |
| `--feat_key` | `features` | h5 数据集键名 |

`a/b/c/d` 没有单独的命令行初值开关，固定从 1.4 和 0.17 起步，并写进每一折的 `metrics.csv`。

新增 `--score_norm layernorm`（默认）或 `none`，控制评分头前的归一化；`--grad_clip 5.0`（默认）控制梯度范数裁剪，设为 0 可关闭。不会擅自改变传入的学习率，`lr >= 1e-3` 时会提示饱和风险。建议首先使用 `--lr 1e-4`。

---

## 六、输出

### K 折

```
log_root/exp_name/
├── config.json
├── kfold_summary.json          # 各折 best c-index，以及该折最优时的 a,b,c,d
└── fold_0/
    ├── input_scaler.json       # 该折训练集的长径 / 淋巴结个数尺度
    ├── formula.json            # 验证 c-index 最优时的 a,b,c,d,hazard_scale
    ├── metrics.csv             # 每轮 loss、c-index、a,b,c,d
    ├── history.json
    ├── checkpoint_best.pt      # 验证 c-index 最优
    └── checkpoint_last.pt
```

### 全量训练

`checkpoint_best_loss.pt` 按训练 loss 最低保存，`formula.json` 对应该时刻的系数。

### 推理

```
save_infer_dir/
├── metrics.json                # c-index、公式系数、尺度
└── risk_scores.csv
```

`risk_scores.csv` 中：

| 列 | 含义 |
| --- | --- |
| `risk_score` / `rcb_index` | 公式得到的 `R`，越大风险越高 |
| `tumor_term` / `lymph_term` | 第一项、第二项。无转移时 `lymph_term = 0` |
| `tumor_slide_score` / `lymph_slide_score` | 两张切片的评分。无转移时淋巴结评分为 0 |
| `surv_risk` | `-sum(S)`，与 `R` 单调对应，仅作对照 |

验证阶段还会保存 `validation_predictions.csv`（最新一轮）和 `validation_predictions_best.csv`（对应最优权重）；二者含每位患者的 `rcb`、两项贡献、两路评分、原始 logit 和 log-score。CSV 使用 17 位有效数字，无转移者淋巴结分支的 logit/log-score 留空，评分和公式贡献为 0。推理的 `risk_scores.csv` 同样增加两路 logit/log-score。

`metrics.csv` 每轮落盘，新增 `val_rcb_std`、`val_rcb_n_unique`、评分最小/最大值、精确零值比例、接近零的比例，以及两路训练梯度范数和零梯度更新比例。验证 C-index 并列时，用更低的验证 loss 决定保存哪个权重。

---

## 七、评分全零与固定 C-index 排查

旧实现存在两个数值问题：

1. `clamp(size_norm * score, min=1e-4)` 会在底数低于阈值时，切断该样本到 WSI 评分网络的梯度。初始系数下每个被截断的项约为 `1.4 * (1e-4)^0.17 = 0.2925`；若两路均触底，R 可能几乎只剩有无淋巴结转移的差别。此时 loss 仍能靠 baseline 和公式参数下降，C-index 却不变。
2. 生存概率连乘后再 `clamp` 取对数，也会在概率过小时截断梯度。加上原始正 R 和零 baseline，使初始每箱 hazard 普遍超过 0.5，容易进一步推动 WSI 评分变小。

切片评分是 sigmoid 输出，代码没有将负评分直接归零。打印成 `0.0000` 可能只是显示精度；极端负 logit 也可能令 float32 sigmoid 真正下溢为 0。新版训练直接使用 log-score，因此不依赖已经下溢的 sigmoid 值继续计算公式。无转移者淋巴结分支为 0 是设计行为，诊断时只统计有转移者的淋巴结评分。

从仓库根目录重新运行，使用新的实验名，避免覆盖旧日志：

```bash
python Prognosis/main_prognosis.py --mode train --split_mode kfold --target DFS \
    --csv_path /mnt/d/BreastRCB-Prognosis-prognosis_with_lymphnode/Prognosis/Example_Dataset_Csv.csv \
    --log_root ./logs --exp_name dfs_lymph_rcb_stable \
    --model_type abmil --k 5 --max_epochs 50 --lr 1e-4
```

检查每轮 `diag` 中 R 的标准差/唯一值数、两路评分范围，以及 `grad` 中两路梯度范数。评分使用科学计数法显示，真正的零与小数能够区分。全部评分接近饱和或全部更新的分支梯度为 0 时会发出提示。若出现 NaN/Inf 损失或非有限梯度则中止；空 H5 或含 NaN/Inf 的特征会明确报错。

C-index 只取决于患者排序，风险的绝对数值变化并不一定改变 C-index；连续若干轮相同不能单独证明训练失效。训练 C-index 使用 dropout 打开、参数逐步更新时的预测，验证 C-index 则来自同一组参数的 eval 预测，应结合验证集原始预测和梯度判断。本仓库正式入口为 `main_prognosis.py`，用户本地的 `main_prognosis2.py` 不在当前分支中。

在仓库根目录运行合成数据回归测试：

```bash
python -m unittest discover -s Prognosis/tests -v
```

测试覆盖 sigmoid 下溢/旧阈值以下的梯度、RCB 公式等价、稳定 NLL 的数值与梯度、三种 MIL 编码器反传，以及合成 H5 的训练保存/推理流程。它们验证数值修复，不代表真实队列的 C-index 一定上升。

---

## 八、依赖

必需：`torch`、`h5py`、`pandas`、`numpy`、`scikit-learn`。

c-index 可选 `scikit-survival` 或 `lifelines`。都没装也能跑。
