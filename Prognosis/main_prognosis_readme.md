# 原发灶预后：淋巴结和临床信息独立开关

当前分支 `prognosis_with_lymphnode_and_clinical` 合并了两个来源分支：

- `prognosis_with_lymphnode`：按用户提供的 `main_prognosis2.py` 更新后的简化公式。
- `prognosis_with_clinical_info`：临床编码器及 concat/bilinear/gated 中期融合。

正式入口为 `Prognosis/main_prognosis.py`，输入是预提取的 H5 patch 特征。支持 OS、DFS 和 RFS 生存预测。

## 两个独立开关

默认开启淋巴结，关闭临床：

- `--use_lymphnode` 开启淋巴结；`--no_lymphnode` 关闭，也兼容 `--no-use_lymphnode`。
- `--use_clinical` 开启临床；`--no_clinical` 关闭，也兼容 `--no-use_clinical`。

关闭淋巴结时不创建节点评分网络、不读取淋巴结 H5、不要求淋巴结路径和转移个数列，风险第二项严格为 0；即使 CSV 保留这些列，也不参与该分支。自动临床筛选同时排除已知的淋巴结专属字段。开关不代表自动识别所有自定义分期/变量的临床含义，具体临床协变量由 `--clinical_cols` 固定。

关闭临床时不创建临床 MLP/融合层、不拟合临床编码器，也不要求任何临床列。原发灶 H5 和肿瘤大小始终为必需输入，因此“两项均关闭”是原发灶影像加大小的基线。

## 模型结构与风险公式

```text
原发灶 patch 特征 -> 原发灶 MIL -> 原发灶表征 -> 评分头 -> tumor_score
                                   ^
                                   | 临床开启时：fusion(WSI表征, 临床MLP表征)
                                   |
临床数值/类别 -> 训练折编码器 -> 临床MLP
                                   |
                                   | 同一临床表征送入另一独立融合层
                                   v
淋巴结 patch 特征 -> 淋巴结 MIL -> 淋巴结表征 -> 评分头 -> lymph_score

R = size_norm * tumor_score + count_norm * lymph_score
logits_k = softplus(hazard_scale_raw) * R + bin_bias_k
hazard_k = sigmoid(logits_k)
```

两张切片使用独立 MIL 编码器；临床 MLP 为患者级共享编码器，两路融合层独立。融合发生在评分前，所以临床信息实际影响 R 的患者排序；C-index 仍使用 R。关闭临床且开启淋巴结时，模型的评分与融合公式和附件版本一致。

没有 a/b/c/d 可训练系数或幂指数；仍训练 MIL、评分头、启用的临床模块和生存读出头。无转移患者不经过节点网络。评分是由生存结局监督得到的隐变量，R 不是官方 RCB 指数或复发概率。

`--fusion_type concat` 为拼接后 MLP，`bilinear` 为双线性交互，`gated` 为门控融合。`--clinical_hidden_dim` 默认 128，实际控制临床嵌入维度。默认在评分头前使用无可训练参数的 LayerNorm。

## CSV 与 H5 格式

每位患者一行，参考 `Example_Dataset_Csv.csv`、`Feature_dict.json`。

始终必需：

- `case_id`、`tumor_slide_feat_path`、正数 `tumor_size`。
- 选择 DFS 时为 `DFS_month/DFS_status`，OS/RFS 同理；时间为有限非负数，状态 1=事件、0=删失。当前推理接口也使用这些列计算指标。

仅淋巴结开启时必需：

- `n_metastatic_lymph_nodes`：有限非负的转移个数。
- `lymph_node_slide_feat_path`：有转移时为有效 H5 路径；无转移可空。

临床开启时提供指定临床列，例如 `Age`、`ER_status`、`HER2_score_pre`、`Molecular_subtype`。H5 默认键 `features`，形状 `[N_patch, feature_dim]`；原发灶和淋巴结维度一致，空或非有限特征会报错。一个患者若有多行，沿用附件规则只取第一行并提示。

列名自定义示例：

```bash
--tumor_feat_col tumor_slide_feat_path --ln_feat_col lymph_node_slide_feat_path \
--tumor_size_col Length --ln_count_col Number_of_lymph_node_metastases
```

## 临床编码与训练折独立性

建议显式指定临床变量，使各实验使用一致变量：

```bash
--clinical_cols Age,ER_status,HER2_score_pre,Molecular_subtype
```

未指定则自动选择非保留列。标识符、路径、全部 OS/DFS/RFS 结局、event_time/censorship/disc_label 等派生标签，以及公式已使用的大小/计数字段禁止进入额外临床向量；明确选择禁用列会报错。淋巴结关闭时还排除已知淋巴结专属临床列，避免在默认筛选中重新引入节点信息。

数值列在当前训练折上拟合均值与标准差，缺失和正负无穷按训练均值填充。类别列使用训练折固定词表，分别保留 missing 和 unknown 维度；验证/推理新类别不扩张维度。已知 `HER2_score_pre/Molecular_subtype` 和文本列按类别处理，其他数值列按数值处理；可用 `--categorical_cols Stage` 明确把数值编码的分期视为类别，且该列须已包含在 `--clinical_cols` 中。

肿瘤大小尺度、淋巴结计数尺度、临床编码器、生存头初值和生存时间内部切点均仅用训练折拟合。每折单独保存，验证/外测复用，不在外测数据重算均值、词表或分位数边界。病例数/不同事件时间不足时，实际时间箱数可以少于 `--n_bins`。

## 四种训练命令

以下均从仓库根目录运行，将 CSV 替换为实际路径。四个实验采用不同实验名，不覆盖权重。

### 原发灶基线：两个开关都关闭

```bash
python Prognosis/main_prognosis.py --mode train --split_mode kfold --target DFS \
    --csv_path /path/to/dataset.csv --log_root ./logs --exp_name dfs_tumor_only \
    --model_type abmil --k 5 --max_epochs 50 --lr 1e-4 \
    --no_lymphnode --no_clinical
```

### 原发灶加淋巴结

```bash
python Prognosis/main_prognosis.py --mode train --split_mode kfold --target DFS \
    --csv_path /path/to/dataset.csv --log_root ./logs --exp_name dfs_lymph \
    --model_type abmil --k 5 --max_epochs 50 --lr 1e-4 \
    --use_lymphnode --no_clinical
```

### 原发灶加临床

```bash
python Prognosis/main_prognosis.py --mode train --split_mode kfold --target DFS \
    --csv_path /path/to/dataset.csv --log_root ./logs --exp_name dfs_clinical \
    --model_type abmil --k 5 --max_epochs 50 --lr 1e-4 \
    --no_lymphnode --use_clinical --fusion_type concat \
    --clinical_cols Age,ER_status,HER2_score_pre,Molecular_subtype
```

### 原发灶加淋巴结加临床

```bash
python Prognosis/main_prognosis.py --mode train --split_mode kfold --target DFS \
    --csv_path /path/to/dataset.csv --log_root ./logs --exp_name dfs_both \
    --model_type abmil --k 5 --max_epochs 50 --lr 1e-4 \
    --use_lymphnode --use_clinical --fusion_type concat \
    --clinical_cols Age,ER_status,HER2_score_pre,Molecular_subtype
```

可把 `concat` 改成 `bilinear/gated`。`--model_type` 支持 `abmil/mean_mil/max_mil`。全量训练将 `--split_mode kfold` 改为 `--split_mode all_train`，并使用新的实验名。默认 `--gc 16`、`--drop_out 0.25`、`--num_workers 2`、`--score_norm layernorm`、`--grad_clip 5`。也支持 `--config /path/to/hparams.json`，配置文件中的值覆盖命令行对应超参数，布尔开关在 JSON 中使用 true/false。

## 推理与配置版本

```bash
python Prognosis/main_prognosis.py --mode infer \
    --config ./logs/dfs_both/fold_0/config.json \
    --ckpt_path ./logs/dfs_both/fold_0/checkpoint_best.pt \
    --csv_path /path/to/test.csv --save_infer_dir ./infer_dfs_both
```

推理实际从权重目录加载该折 `config.json`、`input_scaler.json`、`time_bins.json`；临床开启还必须有 `clinical_encoder.json`。缺少临床编码器时明确报错，不会静默关闭临床。推理的开关、融合结构与特征维度采用训练折配置，不通过命令行改变训练好的模型。

本分支 `numerics_version=4`。原无系数分支为版本 3，旧幂公式为版本 2，权重及配置不能跨版本混用，需新实验名重新训练。

## 输出与诊断

K 折根目录保存请求的 `config.json` 和 `kfold_summary.json`，每折保存实际的 `config.json`、scaler、时间边界、临床编码器（仅启用时）、`survival_head_init.json`、最佳/最后权重、`metrics.csv` 和 `history.json`。类别词表及模型维度可以因折而不同，推理须对应同一折。

`validation_predictions.csv` 为最新一轮验证预测；`validation_predictions_best.csv` 与最佳权重对应，含患者 R、两项贡献、评分、logit 和 log-score。无转移或关闭节点时节点贡献为 0；禁用节点配置的输出计数也记为 0，表示该配置未使用节点计数。C-index 并列时选择验证 loss 更低的权重。全量训练按训练 loss 选择 `checkpoint_best_loss.pt`。

推理保存 `risk_scores.csv` 和 `metrics.json`，记录启用的输入分支。日志和 metrics 保存评分分布、R 唯一值、各启用分支梯度。生存损失使用 logits 稳定路径；R 仍按附件直接使用 sigmoid 评分，极端饱和仍需结合诊断检查。C-index 只看排序，数轮不变不单独证明网络未学习。

## 测试与依赖

必需 `torch`、`numpy`、`pandas`、`h5py`、`scikit-learn`；C-index 可选 `scikit-survival/lifelines`，没有则使用内置实现。推荐 Python 3.9+。

```bash
python -m unittest discover -s Prognosis/tests -v
```

测试包括线性公式、稳定 NLL、MIL/临床反传、四种开关组合、三种融合、禁用输入不读取、临床列排除与未知类别、训练折时间/编码统计以及训练保存和推理一致性。均使用合成数据，不代表真实队列上的预测性能。
