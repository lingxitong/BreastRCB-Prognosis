# BreastRCB-Prognosis：Multi-MIL、可选淋巴结与临床融合

当前分支：`prognosis_with_lymphnode_and_clinical_multi_mil`。从 `prognosis_with_lymphnode_and_clinical@20db2e3` 复制，参考 [MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE/tree/e1c5d9fcd771f30f94437e993ecf24c0535df991) 增加 13 个 MIL 编码器选项，加上原有 3 个，共 16 个，保留无 a/b/c/d 的风险公式和两个独立融合开关。

- `--use_lymphnode / --no_lymphnode`：开启/关闭淋巴结影像及公式计数项，默认开启。
- `--use_clinical / --no_clinical`：开启/关闭临床中期融合，默认关闭。
- 临床融合支持 `concat`、`bilinear`、`gated`。
- 原发灶 H5 特征和肿瘤大小始终保留；两个开关可以独立组合。

支持 `abmil`、`mean_mil`、`max_mil`、`attention_mil`、`rrt_mil`、`dg_mil`、`ca_mil`、`ilra_mil`、`lin_mil`、`fourier_mil`、`mr_abmil`、`nn_mil`、`acmil`、`topk_mil`、`lse_mil`、`gem_mil`。

```bash
python Prognosis/main_prognosis.py --list_models
```

从仓库根目录运行，全开示例：

```bash
python Prognosis/main_prognosis.py --mode train --split_mode kfold --target DFS \
    --csv_path /path/to/dataset.csv --log_root ./logs --exp_name dfs_ilra_both \
    --model_type ilra_mil --mil_max_instances 4096 --k 5 --max_epochs 50 --lr 1e-4 \
    --use_lymphnode --use_clinical --fusion_type concat \
    --clinical_cols Age,ER_status,HER2_score_pre,Molecular_subtype
```

全部模型只依赖 PyTorch 和原有数据依赖；默认每张切片最多 4096 个 patch，训练随机选择、推理等间隔固定选择，设 `--mil_max_instances 0` 可使用全部 patch。参考的是表征聚合器，不包含各分类论文的全部辅助损失或专门训练流程，适配范围与第三方许可见 [说明](THIRD_PARTY_NOTICES.md)。

完整模型参数、四种融合组合、比较循环、推理和测试命令见 [中文使用说明](Prognosis/main_prognosis_readme.md)。
