# BreastRCB-Prognosis：可选淋巴结与临床融合

当前分支：`prognosis_with_lymphnode_and_clinical`。基于更新后的 `prognosis_with_lymphnode` 与 `prognosis_with_clinical_info` 合并实现，保留无 a/b/c/d 的简化风险公式。

- `--use_lymphnode / --no_lymphnode`：开启/关闭淋巴结影像及公式计数项，默认开启。
- `--use_clinical / --no_clinical`：开启/关闭临床中期融合，默认关闭。
- 临床融合支持 `concat`、`bilinear`、`gated`。
- 原发灶 H5 特征和肿瘤大小始终保留；两个开关可以独立组合。

从仓库根目录运行，全开示例：

```bash
python Prognosis/main_prognosis.py --mode train --split_mode kfold --target DFS \
    --csv_path /path/to/dataset.csv --log_root ./logs --exp_name dfs_both \
    --model_type abmil --k 5 --max_epochs 50 --lr 1e-4 \
    --use_lymphnode --use_clinical --fusion_type concat \
    --clinical_cols Age,ER_status,HER2_score_pre,Molecular_subtype
```

完整四种组合、CSV、预处理、推理、输出和测试命令见 [中文使用说明](Prognosis/main_prognosis_readme.md)。当前配置版本为 4，不能与旧分支权重混用。
