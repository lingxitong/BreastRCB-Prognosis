# Third-party notices and adaptation scope

Reference repository: [lingxitong/MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE/tree/e1c5d9fcd771f30f94437e993ecf24c0535df991), fixed commit `e1c5d9fcd771f30f94437e993ecf24c0535df991`.

Its MIT license, Copyright (c) 2025 Toby, is retained in [third_party/MIL_BASELINE_LICENSE.txt](third_party/MIL_BASELINE_LICENSE.txt). This project's original Apache-2.0 license is retained.

## Adapted feature paths

The source paths below are from the fixed reference commit. They provide feature aggregation equations; the original classification heads are replaced by this project's common WSI scorer, optional clinical fusion and survival NLL.

- `attention_mil`: [AB_MIL/ab_mil.py](https://github.com/lingxitong/MIL_BASELINE/blob/e1c5d9fcd771f30f94437e993ecf24c0535df991/modules/AB_MIL/ab_mil.py), plain tanh attention.
- `rrt_mil`: `modules/RRT_MIL/rrt_mil.py`, the reference repository's lightweight residual MLP re-embedding variant. This is not the full region-transformer RRT-MIL paper implementation.
- `dg_mil`: `modules/DG_MIL/dg_mil.py`, the reference repository's lightweight projection/BatchNorm/attention variant. Singleton training bags use saved BN statistics.
- `ca_mil`: `modules/CA_MIL/ca_mil.py`, context QKV, context attention pooling and output projection. Native PyTorch scaled-dot-product attention replaces manual score allocation; computation is still quadratic in the number of patches.
- `ilra_mil`: `modules/ILRA_MIL/ilra_mil.py`, bidirectional low-rank GAB and non-local pooling. Configurable inducing points use one final pooling seed for one bag vector; standardized input dropout is added.
- `lin_mil`: `modules/LIN_MIL/lin_mil.py`, ReLU-kernel attention, learned per-head temperature, denominator floor, residual FFN and CLS. Native reshape/expand replace einops and stochastic depth is Identity. Default depth is reduced to two.
- `fourier_mil`: `modules/FOURIER_MIL/fourier_mil.py`, APFF blocks, square-length padding and CLS. The classification head is removed; FFT temporary dtype handling is made explicit. The stored patch order is used, without assuming physical patch coordinates.
- `mr_abmil`: `modules/MR_AB_MIL/mr_ab_mil.py`, frozen base affine weights plus trainable bias and nonlinear low-rank residual attention paths. An output projection maps raw pooled features to the common hidden width.
- `nn_mil`: `modules/NN_MIL/nn_mil.py`, random training feature subspaces and deterministic evaluation coverage. The fixed coverage permutation is saved as a buffer; evaluation stride is the subspace width, reducing overlap. An output projection maps raw bag features to the shared hidden width. Class-balanced training, fixed-length sub-bag schedules and uncertainty classification reports are not reproduced.
- `acmil`: `modules/AC_MIL/ac_mil.py`, multiple learned token Q/K attention maps averaged into bag V/O pooling. Sub-classifiers, token-only unused value paths, diversity loss and masking curriculum are not included. This is an **ACMIL aggregation adapter**, not the complete original training method.

The original `abmil`, `mean_mil`, `max_mil` are preserved. `topk_mil`, `lse_mil` and `gem_mil` are additional generic pooling controls, not claimed as additional paper reproductions. Aliases do not increase the model count.

## Upstream MIT notices

The reference LIN and MR-AB files explicitly credit these upstream MIT implementations; their full notices are retained:

- [charlotterchtr/Lin-MIL](https://github.com/charlotterchtr/Lin-MIL), Copyright (c) 2025 Charlotte Richter: [LIN_MIL_LICENSE.txt](third_party/LIN_MIL_LICENSE.txt).
- [BearCleverProud/MR-Block](https://github.com/BearCleverProud/MR-Block), Copyright (c) 2026 Conghao Xiong: [MR_BLOCK_LICENSE.txt](third_party/MR_BLOCK_LICENSE.txt).

## Intentionally excluded

No Mamba/S4, custom CUDA operators, graph coordinates, external confounder banks or memory/EMA-specific training loops are introduced. Models whose defining mechanism needs additional classification losses are not silently advertised as full-method baselines. PNEA is not included; its source header identifies a GPL-3.0 upstream.

All encoders are compared under the same prognosis objective. Different results here describe these **survival feature-encoder adaptations**, not a reproduction ranking of the original classification papers.
