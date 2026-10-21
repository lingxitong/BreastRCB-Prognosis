# BreastRCB-Prognosis

当前分支 `prognosis_with_lymphnode` 使用用户提供的 `main_prognosis2.py` 模型逻辑；正式入口仍为 `Prognosis/main_prognosis.py`。

原发灶和淋巴结分别经 MIL 输出评分，按 `R = size_norm * tumor_score + count_norm * lymph_score` 融合，已移除 a/b/c/d 及幂指数。模型训练、输入格式、诊断输出和运行命令见 [中文使用说明](Prognosis/main_prognosis_readme.md)。

当前配置版本为 3，请使用新实验名重新训练，并将权重、config 和 scaler 配套用于推理。
