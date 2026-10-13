# BreastRCB-Prognosis

乳腺癌预后模型。当前分支 `prognosis_with_lymphnode` 在 `main` 的生存训练流程上，加入原发灶与淋巴结的融合。

每个患者的输入是：一张原发肿瘤切片、肿瘤长径、一张淋巴结切片、转移淋巴结个数。预后指数写成与乳腺 RCB 相同结构的公式，四项权重由训练得到；没有淋巴结转移时，淋巴结项为 0。

使用说明见 [Prognosis/main_prognosis_readme.md](Prognosis/main_prognosis_readme.md)。
