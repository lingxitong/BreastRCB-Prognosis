# BreastRCB-Prognosis

乳腺癌预后模型。当前分支 `prognosis_with_lymphnode` 在 `main` 的生存训练流程上，加入原发灶与淋巴结的融合。

每个患者的输入是：一张原发肿瘤切片、肿瘤长径、一张淋巴结切片、转移淋巴结个数。预后指数写成与乳腺 RCB 相同结构的公式，四项权重由训练得到；没有淋巴结转移时，淋巴结项为 0。

使用说明见 [Prognosis/main_prognosis_readme.md](Prognosis/main_prognosis_readme.md)。

本分支已修复低 WSI 评分触发公式硬截断后梯度归零的问题，训练改为稳定的 log-domain RCB/NLL，并保存评分、logit、梯度和验证集预测诊断。请使用新实验名重新训练，详细原因和重跑命令见使用说明的“评分全零与固定 C-index 排查”。
