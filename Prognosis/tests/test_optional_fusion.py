"""Synthetic tests for independent lymph-node and clinical fusion switches.

Run from the repository root with:
    python -m unittest discover -s Prognosis/tests -v

No patient data or GPU is required; these checks cover gradient flow, train-only
preprocessing, saved-model consistency, and input contracts, not clinical AUC.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold


_HERE = Path(__file__).resolve()
_PROGNOSIS = (_HERE.parents[1] if _HERE.parent.name == "tests"
              else _HERE.parent / "BreastRCB-Prognosis-publish" / "Prognosis")
sys.path.insert(0, str(_PROGNOSIS))
import main_prognosis as prognosis
from clinical_utils import ClinicalEncoder, get_clinical_columns, load_clinical_encoder


def config(use_lymphnode=True, use_clinical=False, fusion_type="concat"):
    return {
        "target": "DFS", "k": 2, "split_mode": "kfold", "n_bins": 3,
        "n_classes": 3, "max_epochs": 1, "lr": 1e-4, "reg": 1e-5,
        "drop_out": 0.0, "gc": 3, "alpha_surv": 0.0, "seed": 17,
        "opt": "adam", "model_type": "abmil", "in_dim": 8,
        "hidden_dim": 16, "feat_key": "features", "num_workers": 0,
        "tumor_feat_col": "tumor_slide_feat_path",
        "ln_feat_col": "lymph_node_slide_feat_path",
        "tumor_size_col": "tumor_size",
        "ln_count_col": "n_metastatic_lymph_nodes",
        "score_norm": "layernorm", "grad_clip": 5.0,
        "numerics_version": prognosis.NUMERICS_VERSION,
        "use_lymphnode": use_lymphnode, "use_clinical": use_clinical,
        "clinical_cols": "Age,Molecular_subtype", "categorical_cols": None,
        "clinical_hidden_dim": 12, "clinical_in_dim": 4 if use_clinical else 0,
        "fusion_type": fusion_type,
    }


def gradient_size(module):
    if module is None:
        return 0.0
    return sum(float(p.grad.detach().abs().sum()) for p in module.parameters()
               if p.grad is not None)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fixture(root, n=12):
    rng = np.random.default_rng(83)
    rows = []
    for i in range(n):
        tumor = root / f"tumor_{i}.h5"
        with h5py.File(tumor, "w") as handle:
            handle["features"] = rng.normal(loc=0.04 * i, size=(5 + i % 3, 8)).astype(np.float32)
        has_met = i % 2 == 1
        node = root / f"node_{i}.h5"
        if has_met:
            with h5py.File(node, "w") as handle:
                handle["features"] = rng.normal(loc=-0.07 * i, size=(4 + i % 2, 8)).astype(np.float32)
        rows.append({
            "case_id": f"case_{i:02d}", "tumor_slide_feat_path": str(tumor),
            "lymph_node_slide_feat_path": str(node) if has_met else "",
            "tumor_size": 0.5 + i * 0.8,
            "n_metastatic_lymph_nodes": 1 + i % 4 if has_met else 0,
            "DFS_month": 2.0 + 2 * i, "DFS_status": int(i % 3 != 0),
            "OS_month": 60 + i, "OS_status": int(i % 4 == 0),
            "RFS_month": 40 + i, "RFS_status": int(i % 5 == 0),
            "Age": float(35 + 2 * i),
            "Molecular_subtype": "rare_single" if i == 0 else ("LumA" if i % 2 else "LumB"),
        })
    return pd.DataFrame(rows)


class OptionalFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_cli_switches_are_independent(self):
        cases = [
            ([], True, False),
            (["--no_lymphnode"], False, False),
            (["--use_clinical"], True, True),
            (["--no_lymphnode", "--use_clinical"], False, True),
            (["--no-use_lymphnode", "--no-use_clinical"], False, False),
            (["--use_lymphnode", "--no_clinical"], True, False),
        ]
        for flags, expected_node, expected_clinical in cases:
            with self.subTest(flags=flags), patch.object(
                sys, "argv", ["main_prognosis.py", "--csv_path", "unused.csv"] + flags
            ):
                args = prognosis.get_args()
                self.assertEqual(args.use_lymphnode, expected_node)
                self.assertEqual(args.use_clinical, expected_clinical)

    def test_four_modes_and_all_fusions_preserve_formula_and_active_gradients(self):
        for use_node in (False, True):
            for use_clinical in (False, True):
                fusions = ("concat", "bilinear", "gated") if use_clinical else ("concat",)
                for fusion_type in fusions:
                    with self.subTest(node=use_node, clinical=use_clinical, fusion=fusion_type):
                        torch.manual_seed(31)
                        model = prognosis.build_model(config(use_node, use_clinical, fusion_type),
                                                       torch.device("cpu"))
                        tumor = torch.randn(7, 8)
                        # Disabled branches must not inspect invalid feature bags.
                        node = torch.randn(5, 8) if use_node else torch.full((2, 11), float("nan"))
                        clinical = torch.randn(4) if use_clinical else None
                        size, count, met = (torch.tensor([[x]]) for x in (0.8, 1.2, 1.0))
                        hazards, survival, parts = model(tumor, node, size, count, met, clinical=clinical)
                        expected = size * parts["tumor_score"]
                        if use_node:
                            expected = expected + count * parts["lymph_score"]
                        else:
                            self.assertIsNone(model.node_scorer)
                            self.assertEqual(float(parts["lymph_term"].detach()), 0.0)
                            self.assertEqual(float(parts["lymph_score"].detach()), 0.0)
                        torch.testing.assert_close(parts["rcb"], expected)
                        loss = prognosis.nll_surv_loss(
                            hazards, survival, torch.tensor([0]), torch.tensor([0.0]),
                            logits=parts["logits"],
                        )
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        self.assertGreater(gradient_size(model.tumor_scorer.encoder), 0.0)
                        if use_node:
                            self.assertGreater(gradient_size(model.node_scorer.encoder), 0.0)
                        if use_clinical:
                            self.assertGreater(gradient_size(model.clinical_mlp), 0.0)
                            self.assertTrue(any(
                                isinstance(layer, torch.nn.Linear) and layer.out_features == 12
                                for layer in model.clinical_mlp.modules()
                            ), "clinical_hidden_dim should control the clinical embedding")
                        else:
                            self.assertIsNone(getattr(model, "clinical_mlp", None))

    def test_disabled_node_accepts_missing_or_invalid_unused_columns(self):
        with tempfile.TemporaryDirectory(prefix="rcb_node_off_") as tmp:
            raw = fixture(Path(tmp), n=3)
            cfg = config(False, False)
            variants = [raw.drop(columns=["lymph_node_slide_feat_path", "n_metastatic_lymph_nodes"]),
                        raw.assign(lymph_node_slide_feat_path="missing/never_read.h5",
                                   n_metastatic_lymph_nodes="not-a-count")]
            for frame in variants:
                with self.subTest(columns=list(frame.columns)):
                    pt, _ = prognosis.build_patient_table(frame, "DFS", 3, cfg)
                    self.assertTrue((pt["has_met"] == 0).all())
                    self.assertTrue((pt["n_met"] == 0).all())
                    ds = prognosis.LymphNodeDataset(
                        pt, "features", prognosis.InputScaler().fit(pt),
                        clinical_encoder=None, use_lymphnode=False,
                    )
                    sample = ds[0]
                    self.assertEqual(len(sample), 12)
                    self.assertEqual(sample[1].numel(), 0)
                    self.assertEqual(sample[-1].numel(), 0)

    def test_clinical_encoding_is_finite_fixed_vocabulary_and_label_safe(self):
        frame = pd.DataFrame({
            "Age": [40.0, np.nan, 60.0, np.inf],
            "Molecular_subtype": ["LumA", "LumB", None, "LumA"],
            "HER2_score_pre": [0, 1, 2, 3],
            "case_id": ["a", "b", "c", "d"],
            "DFS_month": [2, 3, 4, 5], "DFS_status": [1, 0, 1, 0],
            "OS_month": [10, 11, 12, 13], "OS_status": [0, 1, 0, 1],
            "RFS_month": [4, 5, 6, 7], "RFS_status": [1, 0, 1, 0],
            "event_time": [2, 3, 4, 5], "censorship": [0, 1, 0, 1],
            "disc_label": [0, 0, 1, 1], "has_met": [0, 1, 0, 1], "n_met": [0, 2, 0, 2],
            "tumor_size": [1, 2, 3, 4], "n_metastatic_lymph_nodes": [0, 2, 0, 2],
            "custom_tumor_file": ["a.h5", "b.h5", "c.h5", "d.h5"],
            "lymph_node_slide_feat_path": ["x", "y", "z", "w"],
            "Maximum_diameter_of_lymph_node_metastasis": [0, 2, 0, 3],
        })
        cfg = config(False, True)
        cfg.update(clinical_cols=None, tumor_feat_col="custom_tumor_file")
        columns = get_clinical_columns(frame, cfg)
        self.assertEqual(set(columns), {"Age", "Molecular_subtype", "HER2_score_pre"})
        for forbidden in ("DFS_status", "OS_month", "RFS_status", "event_time",
                          "disc_label", "custom_tumor_file", "n_met"):
            with self.subTest(column=forbidden):
                bad = dict(cfg, clinical_cols=["Age", forbidden])
                with self.assertRaises(ValueError):
                    get_clinical_columns(frame, bad)
        with self.assertRaises((ValueError, KeyError)):
            get_clinical_columns(frame, dict(cfg, clinical_cols="Age,absent_column"))
        encoder = ClinicalEncoder().fit(frame, clinical_cols=["Age", "Molecular_subtype", "HER2_score_pre"],
                                        categorical_cols=["HER2_score_pre"])
        self.assertAlmostEqual(encoder.numeric_mean["Age"], 50.0)
        self.assertIn("HER2_score_pre", encoder.categorical_cols)
        known = encoder.transform_row(frame.iloc[0])
        unseen = encoder.transform_row(pd.Series({"Age": -np.inf, "Molecular_subtype": "never_seen",
                                                   "HER2_score_pre": 99}))
        missing = encoder.transform_row(pd.Series(dtype=object))
        self.assertEqual(len(unseen), encoder.output_dim)
        for values in (encoder.transform_df(frame), known, unseen, missing):
            self.assertTrue(np.isfinite(values).all())
        self.assertFalse(np.array_equal(unseen, missing), "unknown and missing categories should be distinguishable")
        restored = ClinicalEncoder.from_dict(encoder.to_dict())
        np.testing.assert_array_equal(restored.transform_row(frame.iloc[0]), known)

    def test_each_mode_train_save_infer_roundtrip_and_training_only_assets(self):
        with tempfile.TemporaryDirectory(prefix="rcb_fusion_roundtrip_") as tmp:
            root = Path(tmp)
            raw = fixture(root)
            raw.loc[8:, "Age"] = 1000.0
            raw.loc[8:, "Molecular_subtype"] = "validation_only"
            raw.loc[8:, "DFS_month"] = [1000.0, 1100.0, 1200.0, 1300.0]
            for use_node, use_clinical in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(node=use_node, clinical=use_clinical):
                    cfg = config(use_node, use_clinical)
                    frame = raw.copy()
                    if not use_node:
                        frame = frame.drop(columns=["lymph_node_slide_feat_path", "n_metastatic_lymph_nodes"])
                    if not use_clinical:
                        frame = frame.drop(columns=["Age", "Molecular_subtype"])
                    pt, _ = prognosis.build_patient_table(frame, "DFS", 3, cfg)
                    out_dir = root / f"run_{int(use_node)}_{int(use_clinical)}"
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = prognosis.train_one_run(pt.iloc[:8].copy(), pt.iloc[8:].copy(), cfg,
                                                         torch.device("cpu"), str(out_dir), "synthetic")
                    self.assertGreaterEqual(result["best_epoch"], 0)
                    for name in ("config.json", "time_bins.json", "input_scaler.json", "checkpoint_best.pt",
                                 "validation_predictions_best.csv", "metrics.csv"):
                        self.assertTrue((out_dir / name).is_file(), name)
                    saved_cfg = read_json(out_dir / "config.json")
                    self.assertEqual(saved_cfg["use_lymphnode"], use_node)
                    self.assertEqual(saved_cfg["use_clinical"], use_clinical)
                    bins = read_json(out_dir / "time_bins.json")
                    self.assertEqual(saved_cfg["n_classes"], bins["n_classes"])
                    self.assertTrue(all(2 <= value <= 16 for value in bins["cut_points"]))
                    if use_clinical:
                        encoder = load_clinical_encoder(str(out_dir / "clinical_encoder.json"))
                        self.assertEqual(saved_cfg["clinical_in_dim"], encoder.output_dim)
                        self.assertAlmostEqual(encoder.numeric_mean["Age"], raw.iloc[:8]["Age"].mean())
                        self.assertNotIn("validation_only", encoder.cat_categories["Molecular_subtype"])
                    else:
                        self.assertFalse((out_dir / "clinical_encoder.json").exists())
                    infer_csv = root / f"infer_{int(use_node)}_{int(use_clinical)}.csv"
                    frame.iloc[8:].to_csv(infer_csv, index=False)
                    infer_dir = out_dir / "inference"
                    args = Namespace(csv_path=str(infer_csv), ckpt_path=str(out_dir / "checkpoint_best.pt"),
                                     save_infer_dir=str(infer_dir))
                    with contextlib.redirect_stdout(io.StringIO()):
                        prognosis.run_inference(cfg, torch.device("cpu"), args)
                    actual = pd.read_csv(infer_dir / "risk_scores.csv").set_index("case_id").sort_index()
                    expected = pd.read_csv(out_dir / "validation_predictions_best.csv").set_index("case_id").sort_index()
                    np.testing.assert_allclose(actual["risk_score"], expected["rcb"], rtol=1e-7, atol=1e-8)
                    if not use_node:
                        self.assertTrue((actual["lymph_term"] == 0).all())
                        self.assertTrue((actual["lymph_slide_score"] == 0).all())
                    if use_node and use_clinical:
                        (out_dir / "clinical_encoder.json").unlink()
                        with self.assertRaises(FileNotFoundError) as caught:
                            prognosis.run_inference(cfg, torch.device("cpu"), args)
                        self.assertIn("clinical_encoder", str(caught.exception))

    def test_kfold_uses_training_vocabulary_and_saves_each_actual_dimension(self):
        with tempfile.TemporaryDirectory(prefix="rcb_fusion_kfold_") as tmp:
            root = Path(tmp)
            raw = fixture(root)
            cfg = config(True, True)
            pt, _ = prognosis.build_patient_table(raw, "DFS", cfg["n_bins"], cfg)
            with contextlib.redirect_stdout(io.StringIO()):
                prognosis.train_kfold(pt, cfg, torch.device("cpu"), str(root))
            dimensions = []
            for fold, (train_idx, val_idx) in enumerate(KFold(2, shuffle=True, random_state=17).split(pt)):
                fold_dir = root / f"fold_{fold}"
                encoder = load_clinical_encoder(str(fold_dir / "clinical_encoder.json"))
                saved_cfg = read_json(fold_dir / "config.json")
                dimensions.append(saved_cfg["clinical_in_dim"])
                self.assertEqual(encoder.output_dim, saved_cfg["clinical_in_dim"])
                self.assertAlmostEqual(encoder.numeric_mean["Age"], raw.iloc[train_idx]["Age"].mean())
                observed = set(raw.iloc[train_idx]["Molecular_subtype"])
                for value in {"LumA", "LumB", "rare_single"}:
                    self.assertEqual(value in encoder.cat_categories["Molecular_subtype"], value in observed)
                infer_csv = root / f"fold_input_{fold}.csv"
                raw.iloc[val_idx].to_csv(infer_csv, index=False)
                args = Namespace(csv_path=str(infer_csv), ckpt_path=str(fold_dir / "checkpoint_best.pt"),
                                 save_infer_dir=str(fold_dir / "inference"))
                with contextlib.redirect_stdout(io.StringIO()):
                    prognosis.run_inference(cfg, torch.device("cpu"), args)
                actual = pd.read_csv(fold_dir / "inference" / "risk_scores.csv").set_index("case_id").sort_index()
                expected = pd.read_csv(fold_dir / "validation_predictions_best.csv").set_index("case_id").sort_index()
                np.testing.assert_allclose(actual["risk_score"], expected["rcb"], rtol=1e-7, atol=1e-8)
            self.assertNotEqual(dimensions[0], dimensions[1], "rare training-only category should change fold dimensions")


if __name__ == "__main__":
    unittest.main()
