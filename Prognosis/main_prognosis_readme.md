# Multi-MIL 预后：淋巴结和临床信息独立开关

当前分支 `prognosis_with_lymphnode_and_clinical_multi_mil` 从 `prognosis_with_lymphnode_and_clinical@20db2e3` 复制，并参考 `MIL_BASELINE@e1c5d9fcd771f30f94437e993ecf24c0535df991` 扩展 MIL。保留以下两个来源的融合逻辑：

- `prognosis_with_lymphnode`：按用户提供的 `main_prognosis2.py` 更新后的简化公式。
- `prognosis_with_clinical_info`：临床编码器及 concat/bilinear/gated 中期融合。

正式入口为 `Prognosis/main_prognosis.py`，输入是预提取的 H5 patch 特征。支持 OS、DFS 和 RFS 生存预测。

## 16 种 MIL 选择

以下均为独立 canonical 选项，别名不重复计数。原有 3 种保留，新增 13 种：

- `abmil`：原有 gated attention，包含 tanh 与 sigmoid 门控。
- `mean_mil`：原有均值池化。
- `max_mil`：原有最大池化。
- `attention_mil`：参考 AB_MIL 的普通 tanh 注意力，与 gated ABMIL 不同。
- `rrt_mil`：参考库的轻量残差 MLP 重嵌入加注意力，不是完整区域 Transformer。
- `dg_mil`：参考库的投影、BatchNorm、注意力轻量版本；单 patch 训练使用已有 BN 统计。
- `ca_mil`：上下文 QKV 建模后聚合，计算量随 patch 数平方增长，建议保留 patch 上限。
- `ilra_mil`：低秩潜变量双向交互加 non-local pooling。
- `lin_mil`：ReLU kernel 线性注意力与 CLS 表征。
- `fourier_mil`：APFF 频域过滤加 CLS，使用 H5 内存储的 patch 顺序，不保证任意重排不变。
- `mr_abmil`：冻结仿射注意力映射加可训练非线性低秩残差。
- `nn_mil`：训练随机特征子空间、推理固定子空间覆盖，不含原分类均衡训练流程。
- `acmil`：多 token QK 聚合器适配，不含 sub-classifier、diversity loss 或 masking curriculum。
- `topk_mil`：新增通用对照，逐特征通道 top-k 值取均值。
- `lse_mil`：新增通用对照，按 patch 数归一化的稳定 log-mean-exp 池化。
- `gem_mil`：新增通用对照，可学习幂的广义均值池化。

其中 10 种新增聚合器参考 MIL_BASELINE，Top-k/LSE/GeM 为额外简单池化对照；不声称完整复现 13 篇不同论文。编码器统一输出 `[1, hidden_dim]`，再接相同临床融合、评分头、风险公式和生存损失。未引入 Mamba、图坐标、外部 confounder 或自定义 CUDA 编译依赖。来源与许可见根目录 `THIRD_PARTY_NOTICES.md`。

无需 CSV 即可列出模型：

```bash
python Prognosis/main_prognosis.py --list_models
```

只需改变 `--model_type` 即可选择。兼容部分参考名称：`AB_MIL` 对应 `attention_mil`，`GATE_AB_MIL` 对应 `abmil`，`MR_AB_MIL` 对应 `mr_abmil`，`AC_MIL` 对应 `acmil`，如 `ILRA_MIL` 会规范为 `ilra_mil`。配置保存规范名称。

## Patch 预算与 MIL 参数

新 CLI 默认 `--mil_max_instances 4096`，对全部 16 种模型采用相同每张切片预算；设为 0 表示不限。训练每次从超长 bag 无重复随机取样并保留原索引顺序；验证/推理使用等间隔确定性索引，不依赖随机种子。原发灶和淋巴结各自应用预算。

预算限制模型计算和激活，不限制解码完整 H5 的 CPU 内存，也不生成空间坐标。公平比较请固定预算、数据划分和 seed。预算以内的 bag 不丢 patch；Fourier 保留所选 patch 的存储顺序。

可选参数均有默认值，无需逐一填写：

- `--mil_attention_dim 128`：注意力中间维度。
- `--mil_heads 4`：ILRA/Lin/ACMIL 多头数、Fourier 频域分组数，须整除 `hidden_dim`。
- `--mil_depth 2`：ILRA/Lin/Fourier 层数。
- `--mil_inducing_points 16`：ILRA 潜变量数量。
- `--mil_tokens 5`：ACMIL token 数。
- `--mil_rank 32`：MR-ABMIL 残差 rank。
- `--mil_subspace_dim 128`：nnMIL 特征子空间宽度，推理使用固定 seed 的不重叠为主覆盖。
- `--mil_topk 8`：Top-k 每通道选取数，自动不超过 bag 长度。
- `--mil_lse_temperature 1.0`：LSE 温度。
- `--mil_gem_p 3.0`：GeM 可学习幂初值，必须大于 1。

参数全部保存到每折配置，推理按权重目录配置恢复。LayerNorm、dropout、学习率与临床融合参数沿用本脚本设置。

### ILRA 双融合训练示例

```bash
python Prognosis/main_prognosis.py --mode train --split_mode kfold --target DFS \
    --csv_path /path/to/dataset.csv --log_root ./logs --exp_name dfs_ilra_both \
    --model_type ilra_mil --mil_max_instances 4096 --mil_heads 4 --mil_depth 2 \
    --k 5 --max_epochs 50 --lr 1e-4 \
    --use_lymphnode --use_clinical --fusion_type concat \
    --clinical_cols Age,ER_status,HER2_score_pre,Molecular_subtype
```

### 同一配置比较全部模型

```bash
for model in abmil mean_mil max_mil attention_mil rrt_mil dg_mil ca_mil \
             ilra_mil lin_mil fourier_mil mr_abmil nn_mil acmil \
             topk_mil lse_mil gem_mil; do
    python Prognosis/main_prognosis.py --mode train --split_mode kfold --target DFS \
        --csv_path /path/to/dataset.csv --log_root ./logs --exp_name "dfs_${model}_both" \
        --model_type "$model" --mil_max_instances 4096 --seed 1 \
        --k 5 --max_epochs 50 --lr 1e-4 \
        --use_lymphnode --use_clinical --fusion_type concat \
        --clinical_cols Age,ER_status,HER2_score_pre,Molecular_subtype
done
```

各模型独立实验目录，循环按顺序运行；替换实际 CSV 路径后使用。原有四种融合组合仍使用后文独立开关。

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

可把 `concat` 改成 `bilinear/gated`，把 `abmil` 改成上面的任意 MIL。全量训练将 `--split_mode kfold` 改为 `--split_mode all_train`，并使用新的实验名。默认 `--gc 16`、`--drop_out 0.25`、`--num_workers 2`、`--score_norm layernorm`、`--grad_clip 5`。也支持 `--config /path/to/hparams.json`，配置文件中的值覆盖命令行对应超参数，布尔开关在 JSON 中使用 true/false。

## 推理与配置版本

```bash
python Prognosis/main_prognosis.py --mode infer \
    --config ./logs/dfs_both/fold_0/config.json \
    --ckpt_path ./logs/dfs_both/fold_0/checkpoint_best.pt \
    --csv_path /path/to/test.csv --save_infer_dir ./infer_dfs_both
```

推理实际从权重目录加载该折 `config.json`、`input_scaler.json`、`time_bins.json`；临床开启还必须有 `clinical_encoder.json`。缺少临床编码器时明确报错，不会静默关闭临床。推理的开关、融合结构与特征维度采用训练折配置，不通过命令行改变训练好的模型。

本分支沿用 `numerics_version=4`。原有三种模型的旧版 v4 配置没有 MIL 专属字段时，采用不限 patch 的路径并保留原参数结构，可复用配套的原 v4 权重。新训练保存完整 MIL 参数和预算；不要给旧权重套用另一个实验的新配置。原无系数分支 v3、旧幂公式 v2 仍不兼容。

## 输出与诊断

K 折根目录保存请求的 `config.json` 和 `kfold_summary.json`，每折保存实际的 `config.json`、scaler、时间边界、临床编码器（仅启用时）、`survival_head_init.json`、最佳/最后权重、`metrics.csv` 和 `history.json`。类别词表及模型维度可以因折而不同，推理须对应同一折。

`validation_predictions.csv` 为最新一轮验证预测；`validation_predictions_best.csv` 与最佳权重对应，含患者 R、两项贡献、评分、logit 和 log-score。无转移或关闭节点时节点贡献为 0；禁用节点配置的输出计数也记为 0，表示该配置未使用节点计数。C-index 并列时选择验证 loss 更低的权重。全量训练按训练 loss 选择 `checkpoint_best_loss.pt`。

推理保存 `risk_scores.csv` 和 `metrics.json`，记录启用的输入分支。日志和 metrics 保存评分分布、R 唯一值、各启用分支梯度。生存损失使用 logits 稳定路径；R 仍按附件直接使用 sigmoid 评分，极端饱和仍需结合诊断检查。C-index 只看排序，数轮不变不单独证明网络未学习。

## 测试与依赖

必需 PyTorch 2.0+、`numpy`、`pandas`、`h5py`、`scikit-learn`；C-index 可选 `scikit-survival/lifelines`，没有则使用内置实现。推荐 Python 3.9+。无需安装 timm、einops、torch-geometric 或额外 CUDA 扩展。

```bash
python -m unittest discover -s Prognosis/tests -v
```

测试包括线性公式、稳定 NLL、MIL/临床反传、四种开关组合、三种融合、禁用输入不读取、临床列排除与未知类别、训练折时间/编码统计以及训练保存和推理一致性。均使用合成数据，不代表真实队列上的预测性能。

新增测试覆盖全部 16 个编码器的单 patch/变长输入、关键参数梯度、适用模型的排列不变性、权重重载、真实长 bag 预算限制及 MIL 配置保存。CPU 合成测试通过不等于真实队列模型优劣结论。
