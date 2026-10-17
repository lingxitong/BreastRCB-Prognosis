"""Synthetic H5 training/checkpoint/inference regression tests.

Run from the repository root:
    python -m unittest discover -s Prognosis/tests -v

The tiny CPU fixtures test numerical and file-contract behavior only. They do
not assert clinical discrimination or require patient data, a GPU, or a network.
"""

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main_prognosis as prognosis


def small_config():
    return {
        "target": "DFS", "k": 2, "split_mode": "kfold", "n_bins": 3,
        "n_classes": 3, "max_epochs": 2, "lr": 1e-4, "reg": 1e-5,
        "drop_out": 0.15, "gc": 4, "alpha_surv": 0.0, "seed": 17,
        "opt": "adam", "model_type": "abmil", "in_dim": 8,
        "hidden_dim": 16, "feat_key": "features", "num_workers": 0,
        "tumor_feat_col": "tumor_slide_feat_path",
        "ln_feat_col": "lymph_node_slide_feat_path",
        "tumor_size_col": "tumor_size",
        "ln_count_col": "n_metastatic_lymph_nodes",
        "score_norm": "layernorm", "grad_clip": 5.0,
        "numerics_version": prognosis.NUMERICS_VERSION,
    }


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


class TrainingPipelineTests(unittest.TestCase):
    def test_two_fold_training_checkpoints_and_matching_inference(self):
        """Exercise both WSI branches and the final partial accumulation window."""
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        device = torch.device("cpu")
        cfg = small_config()
        rng = np.random.default_rng(29)
        with tempfile.TemporaryDirectory(prefix="rcb_pipeline_") as tmp:
            root = Path(tmp)
            records = []
            for i in range(12):
                tumor_path = root / f"tumor_{i}.h5"
                with h5py.File(tumor_path, "w") as handle:
                    handle["features"] = rng.normal(
                        loc=0.1 * i, size=(5 + i % 3, 8),
                    ).astype(np.float32)
                has_met = i % 2 == 1
                node_path = root / f"node_{i}.h5"
                if has_met:
                    with h5py.File(node_path, "w") as handle:
                        handle["features"] = rng.normal(
                            loc=-0.1 * i, size=(4 + i % 3, 8),
                        ).astype(np.float32)
                records.append({
                    "case_id": f"case_{i:02d}",
                    "tumor_slide_feat_path": str(tumor_path),
                    "lymph_node_slide_feat_path": str(node_path) if has_met else "",
                    "tumor_size": 0.5 + i * 0.8,
                    "n_metastatic_lymph_nodes": 1 + i % 4 if has_met else 0,
                    "DFS_month": 3.0 + 2 * i,
                    "DFS_status": int(i % 3 != 0),
                })
            raw = pd.DataFrame(records)
            pt, n_classes = prognosis.build_patient_table(raw, "DFS", cfg["n_bins"], cfg)
            cfg["n_classes"] = n_classes
            prognosis.set_seed(cfg["seed"])
            initial_model = prognosis.build_model(cfg, device)
            initial_state = {key: value.detach().clone()
                             for key, value in initial_model.state_dict().items()}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
            # Use a saved, version-checked config for the same path as inference.
            with contextlib.redirect_stdout(io.StringIO()):
                infer_cfg = prognosis.build_config(Namespace(mode="infer", config=str(config_path)))
                summary = prognosis.train_kfold(pt, cfg, device, str(root))
            self.assertEqual(len(summary["folds"]), 2)
            self.assertTrue((root / "kfold_summary.json").is_file())
            splits = KFold(n_splits=2, shuffle=True, random_state=cfg["seed"]).split(pt)
            for fold, (train_idx, val_idx) in enumerate(splits):
                with self.subTest(fold=fold):
                    fold_dir = root / f"fold_{fold}"
                    pt_train = pt.iloc[train_idx]
                    self.assertNotEqual(len(pt_train) % cfg["gc"], 0)
                    self.assertGreater(int(pt_train["has_met"].sum()), 0)
                    self.assertLess(int(pt_train["has_met"].sum()), len(pt_train))
                    for filename in (
                        "checkpoint_best.pt", "checkpoint_last.pt", "metrics.csv",
                        "history.json", "formula.json", "input_scaler.json",
                        "survival_head_init.json", "validation_predictions.csv",
                        "validation_predictions_best.csv",
                    ):
                        self.assertTrue((fold_dir / filename).is_file(), filename)
                        self.assertGreater((fold_dir / filename).stat().st_size, 0, filename)
                    state = torch.load(fold_dir / "checkpoint_best.pt", map_location="cpu",
                                       weights_only=True)
                    reloaded = prognosis.build_model(cfg, device)
                    reloaded.load_state_dict(state, strict=True)
                    self.assertTrue(all(torch.isfinite(value).all() for value in state.values()))
                    for branch in ("tumor_scorer", "node_scorer"):
                        self.assertTrue(any(
                            not torch.equal(value, initial_state[key])
                            for key, value in state.items() if key.startswith(branch)
                        ), f"{branch} never changed")
                    history = pd.read_csv(fold_dir / "metrics.csv")
                    self.assertEqual(len(history), 2)
                    self.assertTrue(np.isfinite(history[["train_loss", "val_loss"]]).all().all())
                    expected_updates = math.ceil(len(pt_train) / cfg["gc"])
                    for branch in ("tumor", "node"):
                        self.assertTrue((history[f"train_{branch}_grad_norm_mean"] > 0).all())
                        updates = history[f"train_{branch}_grad_updates"]
                        if branch == "tumor":
                            self.assertTrue((updates == expected_updates).all())
                        else:
                            # A shuffled window may contain only node-negative
                            # patients; the intentionally absent branch has no grad.
                            self.assertTrue(((updates > 0) & (updates <= expected_updates)).all())
                    for score in ("tumor_score", "lymph_score"):
                        self.assertTrue((history[f"val_{score}_min"] > 0).all())
                        self.assertTrue((history[f"val_{score}_max"] < 1).all())
                        self.assertTrue((history[f"val_{score}_std"] > 0).all())
                        self.assertTrue((history[f"val_{score}_zero_fraction"] == 0).all())
                    self.assertTrue((history["val_rcb_n_unique"] > 1).all())
                    # Both scaler and baseline must be fitted using this training fold.
                    scaler_dict = read_json(fold_dir / "input_scaler.json")
                    self.assertAlmostEqual(scaler_dict["tumor_size_scale"], pt_train["tumor_size"].mean())
                    self.assertAlmostEqual(scaler_dict["ln_count_scale"],
                                           pt_train.loc[pt_train["n_met"] > 0, "n_met"].mean())
                    expected_init = prognosis.initialize_survival_head(
                        prognosis.build_model(cfg, device), pt_train,
                        prognosis.InputScaler().fit(pt_train),
                    )
                    saved_init = read_json(fold_dir / "survival_head_init.json")
                    np.testing.assert_allclose(saved_init["hazard_priors"], expected_init["hazard_priors"])
                    np.testing.assert_allclose(saved_init["bin_bias"], expected_init["bin_bias"])
                    self.assertAlmostEqual(saved_init["reference_risk"], expected_init["reference_risk"])
                    # Reusing the validation patients must reproduce the selected epoch.
                    infer_csv = root / f"validation_{fold}.csv"
                    raw.iloc[val_idx].to_csv(infer_csv, index=False)
                    infer_dir = root / f"inference_{fold}"
                    args = Namespace(csv_path=str(infer_csv), ckpt_path=str(fold_dir / "checkpoint_best.pt"),
                                     save_infer_dir=str(infer_dir))
                    with contextlib.redirect_stdout(io.StringIO()):
                        metrics = prognosis.run_inference(infer_cfg, device, args)
                    predictions = pd.read_csv(infer_dir / "risk_scores.csv").set_index("case_id").sort_index()
                    best = pd.read_csv(fold_dir / "validation_predictions_best.csv").set_index("case_id").sort_index()
                    self.assertEqual(list(predictions.index), list(best.index))
                    for infer_key, val_key in (("risk_score", "rcb"),
                                               ("tumor_slide_score", "tumor_score"),
                                               ("lymph_slide_score", "lymph_score")):
                        np.testing.assert_allclose(predictions[infer_key], best[val_key], rtol=1e-7, atol=1e-8)
                    self.assertEqual(metrics["n_cases"], len(val_idx))
                    self.assertAlmostEqual(metrics["c_index"], summary["folds"][fold]["best_val_cindex"])
                    no_met = best["has_met"] == 0
                    self.assertTrue(no_met.any())
                    self.assertTrue((best.loc[no_met, "lymph_term"] == 0).all())
                    self.assertTrue((best.loc[no_met, "lymph_score"] == 0).all())

    def test_survival_head_matches_known_training_prior_and_rcb_formula(self):
        cfg = small_config()
        cfg["n_classes"] = 4
        model = prognosis.build_model(cfg, torch.device("cpu"))
        pt_train = pd.DataFrame({
            "disc_label": [0, 0, 1, 2], "censorship": [0, 1, 0, 0],
            "tumor_size": [1.0, 3.0, 5.0, 7.0], "n_met": [0.0, 1.0, 0.0, 3.0],
        })
        scaler = prognosis.InputScaler().fit(pt_train)
        self.assertEqual(scaler.tumor_size_scale, 4.0)
        self.assertEqual(scaler.ln_count_scale, 2.0)
        details = prognosis.initialize_survival_head(model, pt_train, scaler)
        expected_priors = np.array([0.3, 0.5, 0.75, 0.1])
        np.testing.assert_allclose(details["hazard_priors"], expected_priors)
        a, b, c, d = model.rcb.coeff_dict().values()
        sizes = pt_train["tumor_size"].to_numpy() / 4.0
        counts = pt_train["n_met"].to_numpy() / 2.0
        expected_reference = np.mean(a * (sizes * 0.5) ** b + c * (counts * 0.5) ** d)
        self.assertAlmostEqual(details["reference_risk"], expected_reference)
        actual_priors = torch.sigmoid(model.bin_bias.detach()
                                     + torch.nn.functional.softplus(model.hazard_scale_raw.detach())
                                     * expected_reference)
        np.testing.assert_allclose(actual_priors.numpy(), expected_priors, rtol=1e-6)


class InputContractTests(unittest.TestCase):
    def test_nonfinite_empty_and_malformed_h5_features_fail(self):
        cases = {
            "nan": np.full((2, 8), np.nan),
            "inf": np.full((2, 8), np.inf),
            "no_patches": np.empty((0, 8)),
            "no_dimensions": np.empty((2, 0)),
            "wrong_rank": np.zeros(8),
        }
        with tempfile.TemporaryDirectory(prefix="rcb_bad_h5_") as tmp:
            for name, values in cases.items():
                with self.subTest(case=name):
                    path = Path(tmp) / f"{name}.h5"
                    with h5py.File(path, "w") as handle:
                        handle["features"] = values
                    with self.assertRaises(ValueError) as caught:
                        prognosis.load_features(str(path), "features")
                    self.assertIn(str(path), str(caught.exception))

    def test_inference_rejects_legacy_or_missing_versioned_config(self):
        with tempfile.TemporaryDirectory(prefix="rcb_config_") as tmp:
            path = Path(tmp) / "config.json"
            for version in (None, 1, prognosis.NUMERICS_VERSION + 1):
                with self.subTest(version=version):
                    cfg = small_config()
                    if version is None:
                        del cfg["numerics_version"]
                    else:
                        cfg["numerics_version"] = version
                    path.write_text(json.dumps(cfg), encoding="utf-8")
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(ValueError, "numerics_version"):
                            prognosis.build_config(Namespace(mode="infer", config=str(path)))
            with self.assertRaisesRegex(ValueError, "numerics_version"):
                prognosis.build_config(Namespace(mode="infer", config=None))


if __name__ == "__main__":
    unittest.main()
