# 原发灶与淋巴结预后：简化公式版

本分支按用户提供的 `main_prognosis2.py` 更新，正式入口保留为 `main_prognosis.py`。每位患者使用一张原发灶 H5 特征、一张淋巴结 H5 特征、肿瘤大小和转移淋巴结个数，训练 OS、DFS 或 RFS 生存预测。

## 模型逻辑

两套独立的 MIL 编码器生成切片表征，经可选 LayerNorm 和评分头输出 0～1 的分数：

```text
R = tumor_size_norm * tumor_score + ln_count_norm * lymph_score
logits_k = softplus(hazard_scale_raw) * R + bin_bias_k
hazard_k = sigmoid(logits_k)
```

已移除可训练 a/b/c/d 和幂变换，相当于四者均为 1。MIL、评分头、正的 hazard_scale 和各时间箱 baseline 仍参与训练。转移淋巴结个数为 0 时不读取淋巴结 H5，第二项严格为 0。

大小与个数分别除以当前训练折的均值；阳性淋巴结个数尺度只在有转移者中拟合。生存头用当前训练折事件与在险人数初始化。损失直接从 logits 计算 log-hazard 和 log-survival；C-index 使用 R 排序，越大表示风险越高。`n_classes=4` 表示四个生存时间区间。

切片评分是生存结局监督下学习的隐变量，不等同于病理浸润比例；R 不是官方 RCB，也不是复发概率。

## CSV 与 H5

一个患者一行，列格式参考 `Example_Dataset_Csv.csv`：

- `case_id`：患者 ID。
- `tumor_slide_feat_path`：原发灶 H5 特征绝对路径。
- `tumor_size`：有限且大于 0 的肿瘤大小。
- `lymph_node_slide_feat_path`：有转移时必填，无转移时可空。
- `n_metastatic_lymph_nodes`：转移淋巴结个数，有限且非负。
- `OS_month/OS_status`、`DFS_month/DFS_status` 或 `RFS_month/RFS_status`：所选结局时间和状态，1=事件、0=删失。

H5 默认键名 `features`，形状为 `[N_patch, feature_dim]`。两路特征维度必须相同；空特征或 NaN/Inf 会报错。输入是已提取的特征，不直接读取原始 WSI。同一患者多行时沿用原脚本规则，只使用第一行并提示。

可通过 `--tumor_feat_col`、`--ln_feat_col`、`--tumor_size_col`、`--ln_count_col` 修改列名。

## 训练命令

以下从仓库根目录运行。五折 DFS：

```bash
python Prognosis/main_prognosis.py --mode train --split_mode kfold --target DFS \
    --csv_path /path/to/dataset.csv \
    --log_root ./logs --exp_name dfs_lymph_linear \
    --model_type abmil --k 5 --max_epochs 50 --lr 1e-4
```

全量训练：

```bash
python Prognosis/main_prognosis.py --mode train --split_mode all_train --target DFS \
    --csv_path /path/to/dataset.csv \
    --log_root ./logs --exp_name dfs_lymph_linear_all \
    --model_type abmil --max_epochs 50 --lr 1e-4
```

支持 `abmil/mean_mil/max_mil`。默认 `--score_norm layernorm`、`--grad_clip 5`、`--gc 16`、`--n_bins 4`、`--drop_out 0.25`、`--num_workers 2`。可用 `--score_norm none` 或 `--grad_clip 0` 做相应对照。仍需检查评分饱和情况，建议从 `lr=1e-4` 开始。

用 `--config /path/to/hparams.json` 可覆盖保存的超参数。

## 推理命令与版本

```bash
python Prognosis/main_prognosis.py --mode infer \
    --config ./logs/dfs_lymph_linear/config.json \
    --ckpt_path ./logs/dfs_lymph_linear/fold_0/checkpoint_best.pt \
    --csv_path /path/to/test.csv \
    --save_infer_dir ./infer_dfs_lymph_linear
```

当前 `numerics_version=3`，推理会拒绝旧版本 config。请使用新实验名重新训练；旧权重不能与新模型混用。推理必须复用该权重目录的 `input_scaler.json`。

## 输出与诊断

每折保存 `checkpoint_best.pt`、`checkpoint_last.pt`、`input_scaler.json`、`survival_head_init.json`、`formula.json`、`metrics.csv` 和 `history.json`；根目录保存 `config.json` 与 `kfold_summary.json`。全量训练按最低训练 loss 保存 `checkpoint_best_loss.pt`。

`validation_predictions.csv` 保存最新一轮验证预测，`validation_predictions_best.csv` 与最佳权重对应。输出含 R、两项贡献、两路评分、logit 和 log-score；无转移时淋巴结贡献与评分为 0，其 logit/log-score 留空。推理输出 `risk_scores.csv` 与 `metrics.json`。

日志以科学计数法显示评分范围、R 标准差及唯一值数，并输出两路梯度范数。`metrics.csv` 同时保存精确零值比例和小于 1e-4 的比例。C-index 并列时按更低验证 loss 保存最佳权重。

本版去掉了旧乘积 hard clamp，但 R 实际直接使用 sigmoid 评分，log-score 仅用于诊断；极端 sigmoid 饱和仍可能造成分数/梯度接近零或为零。C-index 只依赖排序，数轮相同不单独说明程序错误。原有时间分箱在划分 K 折之前生成，本分支按附件保留此行为。

## 测试与依赖

必需：`torch`、`h5py`、`pandas`、`numpy`、`scikit-learn`。C-index 可选安装 `scikit-survival` 或 `lifelines`，否则使用内置实现。

```bash
python -m unittest discover -s Prognosis/tests -v
```

测试验证线性公式、梯度、稳定生存损失、三种 MIL 编码器及合成 H5 训练/保存/推理流程，不代表真实队列上的临床效能。
