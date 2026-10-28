"""Registry-driven MIL encoder and survival integration regression tests.

These CPU fixtures check variable bag sizes, gradients, optional inputs, and
saved-model/configuration contracts. They do not assess clinical performance.
Run from the repository root: python -m unittest discover -s Prognosis/tests -v
"""

import contextlib
import io
import json
import subprocess
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
import torch.nn as nn


PROGNOSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROGNOSIS_DIR))
import main_prognosis as prognosis
from mil_encoders import (
    MILInstanceLimiter, MODEL_ALIASES, MODEL_REGISTRY, canonical_model_name,
)


BASE_MODELS = {"abmil", "mean_mil", "max_mil"}
EXPECTED_NEW_MODELS = {
    "attention_mil", "rrt_mil", "dg_mil", "ca_mil", "ilra_mil", "lin_mil",
    "fourier_mil", "mr_abmil", "nn_mil", "acmil", "topk_mil", "lse_mil", "gem_mil",
}
MIL_SETTINGS = {
    "mil_max_instances": 0, "mil_attention_dim": 12, "mil_heads": 4,
    "mil_depth": 1, "mil_inducing_points": 4, "mil_tokens": 3,
    "mil_rank": 4, "mil_subspace_dim": 4, "mil_topk": 3,
    "mil_lse_temperature": 0.7, "mil_gem_p": 2.5,
}


def small_config(name, *, use_node=True, use_clinical=False, fusion="concat"):
    return {
        "target": "DFS", "k": 2, "split_mode": "kfold", "n_bins": 2,
        "n_classes": 2, "max_epochs": 1, "lr": 1e-4, "reg": 1e-5,
        "drop_out": 0.0, "gc": 4, "alpha_surv": 0.0, "seed": 17,
        "opt": "adam", "model_type": name, "in_dim": 8,
        "hidden_dim": 16, "feat_key": "features", "num_workers": 0,
        "tumor_feat_col": "tumor_slide_feat_path",
        "ln_feat_col": "lymph_node_slide_feat_path",
        "tumor_size_col": "tumor_size", "ln_count_col": "n_metastatic_lymph_nodes",
        "score_norm": "layernorm", "grad_clip": 5.0,
        "numerics_version": prognosis.NUMERICS_VERSION,
        "use_lymphnode": use_node, "use_clinical": use_clinical,
        "clinical_cols": "Age,Molecular_subtype", "categorical_cols": None,
        "clinical_in_dim": 4 if use_clinical else 0,
        "clinical_hidden_dim": 12, "fusion_type": fusion,
        **MIL_SETTINGS,
    }


def gradient_size(module):
    if module is None:
        return 0.0
    return sum(float(p.grad.detach().abs().sum()) for p in module.parameters()
               if p.grad is not None)


def forward_survival(model, *, use_node=True, use_clinical=False):
    tumor = torch.randn(11, 8)
    # Disabled input branches must never inspect these deliberately invalid bags.
    node = torch.randn(7, 8) if use_node else torch.full((2, 13), float("nan"))
    clinical = torch.randn(4) if use_clinical else None
    return model(
        tumor, node, torch.tensor([[0.8]]), torch.tensor([[1.2]]),
        torch.tensor([[1.0]]), clinical=clinical,
    )


class RecordingEncoder(nn.Module):
    """Observe the expensive encoder's true input size without allocating N²."""

    hidden_dim = 8

    def __init__(self):
        super().__init__()
        self.seen_sizes = []

    def forward(self, bag):
        self.seen_sizes.append(len(bag))
        return bag.mean(dim=0, keepdim=True)


class MultiMILTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_registry_has_distinct_canonical_models_and_cli_aliases(self):
        self.assertTrue(BASE_MODELS <= set(MODEL_REGISTRY))
        self.assertTrue(EXPECTED_NEW_MODELS <= set(MODEL_REGISTRY))
        self.assertGreaterEqual(len(set(MODEL_REGISTRY) - BASE_MODELS), 11)
        self.assertIsInstance(MODEL_ALIASES, dict)
        for name, metadata in MODEL_REGISTRY.items():
            with self.subTest(name=name):
                self.assertEqual(canonical_model_name(name), name)
                self.assertIsInstance(metadata["permutation_invariant"], bool)
                self.assertTrue(metadata["display"])
                self.assertTrue(metadata["source"])
                with patch.object(sys, "argv", ["main_prognosis.py", "--csv_path", "unused.csv",
                                                 "--model_type", name]):
                    args = prognosis.get_args()
                self.assertEqual(canonical_model_name(args.model_type), name)
        for alias, expected in (("AB_MIL", "attention_mil"), ("GATE_AB_MIL", "abmil"),
                                ("MEAN_MIL", "mean_mil")):
            with self.subTest(alias=alias):
                self.assertEqual(canonical_model_name(alias), expected)
                with patch.object(sys, "argv", ["main_prognosis.py", "--csv_path", "unused.csv",
                                                 "--model_type", alias]):
                    self.assertEqual(canonical_model_name(prognosis.get_args().model_type), expected)
        with self.assertRaises(ValueError):
            canonical_model_name("not_a_registered_mil")

    def test_variable_bags_have_finite_representations_and_encoder_gradients(self):
        for name in MODEL_REGISTRY:
            for n_patches in (1, 3, 17):
                with self.subTest(model=name, n=n_patches):
                    torch.manual_seed(37)
                    encoder = prognosis.build_encoder(small_config(name))
                    self.assertEqual(encoder.hidden_dim, 16)
                    bag = torch.randn(n_patches, 8, requires_grad=True)
                    representation = encoder(bag)
                    self.assertEqual(tuple(representation.shape), (1, 16))
                    self.assertTrue(torch.isfinite(representation).all())
                    # A sum probe is identically zero after some LayerNorms.
                    probe = torch.linspace(-0.7, 1.3, 16).reshape(1, 16)
                    (representation * probe).sum().backward()
                    self.assertIsNotNone(bag.grad)
                    self.assertTrue(torch.isfinite(bag.grad).all())
                    self.assertGreater(float(bag.grad.abs().sum()), 0.0)
                    self.assertGreater(gradient_size(encoder), 0.0)
                    for parameter_name, parameter in encoder.named_parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                    if n_patches == 17:
                        # An unused critical-instance classifier, for example,
                        # must not be silently left untrained by the survival loss.
                        missing = [key for key, parameter in encoder.named_parameters()
                                   if parameter.requires_grad and parameter.grad is None]
                        self.assertEqual(missing, [], f"unconnected trainable parameters: {missing}")

    def test_declared_bag_invariant_encoders_ignore_patch_order(self):
        bag = torch.randn(17, 8, generator=torch.Generator().manual_seed(71))
        permutation = torch.randperm(17, generator=torch.Generator().manual_seed(72))
        for name, metadata in MODEL_REGISTRY.items():
            if not metadata["permutation_invariant"]:
                continue
            with self.subTest(model=name):
                torch.manual_seed(7)
                encoder = prognosis.build_encoder(small_config(name)).eval()
                with torch.no_grad():
                    actual = encoder(bag)
                    permuted = encoder(bag[permutation])
                torch.testing.assert_close(actual, permuted, atol=2e-5, rtol=2e-4)

    def test_every_encoder_supports_joint_survival_gradients(self):
        fusions = ("concat", "bilinear", "gated")
        for i, name in enumerate(MODEL_REGISTRY):
            with self.subTest(model=name, fusion=fusions[i % len(fusions)]):
                torch.manual_seed(31)
                cfg = small_config(name, use_node=True, use_clinical=True,
                                   fusion=fusions[i % len(fusions)])
                model = prognosis.build_model(cfg, torch.device("cpu"))
                hazards, survival, parts = forward_survival(model, use_clinical=True)
                expected = 0.8 * parts["tumor_score"] + 1.2 * parts["lymph_score"]
                torch.testing.assert_close(parts["rcb"], expected)
                loss = prognosis.nll_surv_loss(
                    hazards, survival, torch.tensor([0]), torch.tensor([0.0]),
                    logits=parts["logits"],
                )
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for module in (model.tumor_scorer.encoder, model.node_scorer.encoder,
                               model.clinical_mlp):
                    self.assertGreater(gradient_size(module), 0.0)

    def test_optional_modes_and_state_dict_roundtrip(self):
        modes = ((False, False), (True, False), (False, True), (True, True))
        for i, name in enumerate(MODEL_REGISTRY):
            use_node, use_clinical = modes[i % len(modes)]
            with self.subTest(model=name, node=use_node, clinical=use_clinical):
                torch.manual_seed(13)
                cfg = small_config(name, use_node=use_node, use_clinical=use_clinical,
                                   fusion=("concat", "bilinear", "gated")[i % 3])
                model = prognosis.build_model(cfg, torch.device("cpu")).eval()
                tumor, node, clinical = torch.randn(9, 8), torch.randn(5, 8), torch.randn(4)
                args = (tumor, node, torch.tensor([[1.1]]), torch.tensor([[0.7]]), torch.tensor([[1.0]]))
                with torch.no_grad():
                    _, _, expected = model(*args, clinical=clinical if use_clinical else None)
                if not use_node:
                    self.assertIsNone(model.node_scorer)
                    self.assertEqual(float(expected["lymph_term"]), 0.0)
                if not use_clinical:
                    self.assertIsNone(model.clinical_mlp)
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = prognosis.build_model(cfg, torch.device("cpu")).eval()
                restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
                with torch.no_grad():
                    _, _, actual = restored(*args, clinical=clinical if use_clinical else None)
                torch.testing.assert_close(actual["rcb"], expected["rcb"], rtol=0, atol=0)
                torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0, atol=0)

    def test_instance_limiter_bounds_encoder_work_and_is_deterministic_in_eval(self):
        recorder = RecordingEncoder()
        limiter = MILInstanceLimiter(recorder, max_instances=32)
        self.assertEqual(limiter.hidden_dim, recorder.hidden_dim)
        bag = torch.randn(10001, 8)
        limiter.eval()
        selected = limiter.select_indices(len(bag), bag.device)
        expected_indices = torch.linspace(0, len(bag) - 1, 32).round().long()
        torch.testing.assert_close(selected.cpu(), expected_indices)
        torch.testing.assert_close(limiter(bag), recorder(bag[selected]))
        self.assertEqual(recorder.seen_sizes[-2:], [32, 32])
        with torch.no_grad():
            first = limiter(bag)
            second = limiter(bag)
            small = bag[:3]
            torch.testing.assert_close(limiter(small), recorder(small))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        limiter.train()
        torch.manual_seed(9)
        train_indices = limiter.select_indices(len(bag), bag.device)
        self.assertEqual(train_indices.numel(), 32)
        self.assertEqual(torch.unique(train_indices).numel(), 32)
        self.assertTrue(torch.all(train_indices[1:] > train_indices[:-1]))
        self.assertGreaterEqual(int(train_indices.min()), 0)
        self.assertLess(int(train_indices.max()), len(bag))
        self.assertFalse(torch.equal(train_indices.cpu(), expected_indices))
        cfg = small_config("attention_mil")
        cfg["mil_max_instances"] = 7
        self.assertIsInstance(prognosis.build_encoder(cfg), MILInstanceLimiter)

    def test_new_hyperparameters_are_saved_with_training_and_reload_for_inference(self):
        cfg = small_config("ilra_mil", use_node=False)
        cfg["mil_max_instances"] = 7
        with tempfile.TemporaryDirectory(prefix="multi_mil_roundtrip_") as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(8)
            rows = []
            for i in range(8):
                path = root / f"tumor_{i}.h5"
                with h5py.File(path, "w") as handle:
                    handle["features"] = rng.normal(size=(10 + i, 8)).astype(np.float32)
                rows.append({"case_id": f"P{i}", "tumor_slide_feat_path": str(path),
                             "tumor_size": 0.5 + i, "DFS_month": 2.0 + 2 * i,
                             "DFS_status": int(i % 2 == 0)})
            raw = pd.DataFrame(rows)
            pt, _ = prognosis.build_patient_table(raw, "DFS", cfg["n_bins"], cfg)
            run_dir = root / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                prognosis.train_one_run(pt.iloc[:6], pt.iloc[6:], cfg, torch.device("cpu"),
                                        str(run_dir), "multi_mil")
            saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            for key in MIL_SETTINGS:
                self.assertIn(key, prognosis.HPARAM_KEYS)
                self.assertEqual(saved[key], cfg[key], key)
            self.assertEqual(saved["model_type"], "ilra_mil")
            csv = root / "validation.csv"
            raw.iloc[6:].to_csv(csv, index=False)
            args = Namespace(csv_path=str(csv), ckpt_path=str(run_dir / "checkpoint_best.pt"),
                             save_infer_dir=str(root / "infer"))
            with contextlib.redirect_stdout(io.StringIO()):
                prognosis.run_inference(cfg, torch.device("cpu"), args)
            actual = pd.read_csv(root / "infer" / "risk_scores.csv").set_index("case_id").sort_index()
            expected = pd.read_csv(run_dir / "validation_predictions_best.csv").set_index("case_id").sort_index()
            np.testing.assert_allclose(actual["risk_score"], expected["rcb"], rtol=1e-7, atol=1e-8)

    def test_model_listing_needs_no_csv_or_training_data(self):
        result = subprocess.run(
            [sys.executable, str(PROGNOSIS_DIR / "main_prognosis.py"), "--list_models"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in MODEL_REGISTRY:
            self.assertIn(name, result.stdout)


if __name__ == "__main__":
    unittest.main()
