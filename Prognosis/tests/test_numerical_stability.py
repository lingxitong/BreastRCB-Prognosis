"""Synthetic regression tests for differentiable WSI scores and survival NLL.

Run from the repository root with:
    python -m unittest discover -s Prognosis/tests -v

These tests require no slides, H5 files, GPU, or network access. They check
numerical stability and gradient flow, not clinical predictive performance.
"""

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main_prognosis as prognosis


def tensor(value, *, dtype=torch.float32):
    return torch.tensor(value, dtype=dtype)


def gradient_norm(module):
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def small_model(model_type="abmil"):
    config = {
        "model_type": model_type,
        "in_dim": 8,
        "hidden_dim": 16,
        "drop_out": 0.0,
        "n_classes": 4,
        "score_norm": "layernorm",
    }
    return prognosis.build_model(config, torch.device("cpu"))


class RCBNumericalStabilityTests(unittest.TestCase):
    def test_linear_rcb_gradients_match_sigmoid_chain_rule(self):
        """Each WSI branch contributes its normalized size/count times score."""
        rcb = prognosis.RCBIndex().double()
        tumor_logit = tensor([[-2.0]], dtype=torch.float64).requires_grad_()
        lymph_logit = tensor([[1.5]], dtype=torch.float64).requires_grad_()
        tumor_score = tumor_logit.sigmoid()
        lymph_score = lymph_logit.sigmoid()
        size = tensor([[0.43]], dtype=torch.float64).requires_grad_()
        count = tensor([[1.73]], dtype=torch.float64).requires_grad_()
        risk, tumor_term, lymph_term = rcb(
            tumor_score, size, lymph_score, count,
            tensor([[1.0]], dtype=torch.float64),
        )
        torch.testing.assert_close(tumor_term, size * tumor_score)
        torch.testing.assert_close(lymph_term, count * lymph_score)
        risk.sum().backward()
        torch.testing.assert_close(
            tumor_logit.grad, size.detach() * tumor_score.detach() * (1 - tumor_score.detach()),
        )
        torch.testing.assert_close(
            lymph_logit.grad, count.detach() * lymph_score.detach() * (1 - lymph_score.detach()),
        )
        torch.testing.assert_close(size.grad, tumor_score.detach())
        torch.testing.assert_close(count.grad, lymph_score.detach())

    def test_linear_formula_has_no_coefficients_or_score_floor(self):
        rcb = prognosis.RCBIndex().double()
        self.assertEqual(list(rcb.parameters()), [])
        self.assertEqual(rcb.coeff_dict(), {})
        self.assertEqual(dict(rcb.state_dict()), {})
        for score_values in ((0.32, 0.71), (1e-8, 2e-8), (0.0, 0.0)):
            with self.subTest(scores=score_values):
                tumor_score = tensor([[score_values[0]]], dtype=torch.float64)
                lymph_score = tensor([[score_values[1]]], dtype=torch.float64)
                size = tensor([[0.43]], dtype=torch.float64)
                count = tensor([[1.73]], dtype=torch.float64)
                expected_tumor = size * tumor_score
                expected_lymph = count * lymph_score
                risk, tumor_term, lymph_term = rcb(
                    tumor_score, size, lymph_score, count,
                    tensor([[1.0]], dtype=torch.float64),
                )
                torch.testing.assert_close(tumor_term, expected_tumor, rtol=0, atol=0)
                torch.testing.assert_close(lymph_term, expected_lymph, rtol=0, atol=0)
                torch.testing.assert_close(risk, expected_tumor + expected_lymph, rtol=0, atol=0)

    def test_no_metastasis_has_exact_zero_node_term_and_no_node_gradients(self):
        torch.manual_seed(7)
        model = small_model()
        hazards, survival, parts = model(
            torch.randn(11, 8), torch.randn(7, 8),
            tensor([[1.0]]), tensor([[0.0]]), tensor([[0.0]]),
        )
        self.assertEqual(float(parts["lymph_term"].detach()), 0.0)
        self.assertEqual(float(parts["lymph_score"].detach()), 0.0)
        torch.testing.assert_close(parts["rcb"], parts["tumor_term"])
        loss = prognosis.nll_surv_loss(
            hazards, survival, tensor([0]).long(), tensor([0]),
            logits=parts["logits"],
        )
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.node_scorer.parameters()))
        self.assertGreater(gradient_norm(model.tumor_scorer.encoder), 0.0)


class SurvivalLossStabilityTests(unittest.TestCase):
    def test_logits_and_probability_nll_match_values_and_gradients(self):
        values = [
            [-0.8, -0.1, 0.2, 0.7],
            [0.5, -0.4, 0.6, -0.2],
            [-0.3, 0.2, -0.5, 0.8],
            [0.1, -0.6, 0.4, 0.3],
        ]
        bins = tensor([0, 3, 1, 3]).long()
        censorship = tensor([0, 0, 1, 1], dtype=torch.float64)
        for alpha in (0.0, 0.25):
            with self.subTest(alpha=alpha):
                old_logits = tensor(values, dtype=torch.float64).requires_grad_()
                new_logits = old_logits.detach().clone().requires_grad_()
                old_hazards = old_logits.sigmoid()
                old_survival = (1 - old_hazards).cumprod(dim=1)
                old_loss = prognosis.nll_surv_loss(
                    old_hazards, old_survival, bins, censorship, alpha=alpha,
                )
                new_hazards = new_logits.sigmoid()
                new_loss = prognosis.nll_surv_loss(
                    new_hazards, (1 - new_hazards).cumprod(dim=1),
                    bins, censorship, alpha=alpha, logits=new_logits,
                )
                old_loss.backward()
                new_loss.backward()
                torch.testing.assert_close(new_loss, old_loss, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(
                    new_logits.grad, old_logits.grad, rtol=1e-6, atol=1e-7,
                )

    def test_extreme_logits_early_event_and_late_censor_keep_finite_gradients(self):
        for alpha in (0.0, 0.25):
            with self.subTest(alpha=alpha):
                logits = tensor([
                    [-100.0, -100.0, -100.0, -100.0],
                    [100.0, 100.0, 100.0, 100.0],
                ]).requires_grad_()
                hazards = logits.sigmoid()
                loss = prognosis.nll_surv_loss(
                    hazards, (1 - hazards).cumprod(dim=1),
                    tensor([0, 3]).long(), tensor([0, 1]),
                    alpha=alpha, logits=logits,
                )
                self.assertTrue(torch.isfinite(loss))
                # Correct NLL is 100 for the first event and 400 for censoring.
                expected = (100.0 + (1 - alpha) * 400.0) / 2
                self.assertAlmostEqual(float(loss.detach()), expected, places=4)
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertLess(float(logits.grad[0, 0]), 0.0)
                self.assertTrue((logits.grad[1] > 0).all())


class EndToEndGradientTests(unittest.TestCase):
    def test_all_mil_encoders_receive_main_survival_loss_gradients(self):
        for model_type in ("abmil", "mean_mil", "max_mil"):
            with self.subTest(model_type=model_type):
                torch.manual_seed(19)
                model = small_model(model_type)
                hazards, survival, parts = model(
                    torch.randn(13, 8), torch.randn(9, 8),
                    tensor([[0.8]]), tensor([[1.2]]), tensor([[1.0]]),
                )
                for key in (
                    "tumor_logit", "lymph_logit", "tumor_log_score",
                    "lymph_log_score", "logits",
                ):
                    self.assertIn(key, parts)
                    self.assertTrue(torch.isfinite(parts[key]).all())
                torch.testing.assert_close(parts["tumor_score"], parts["tumor_logit"].sigmoid())
                torch.testing.assert_close(parts["lymph_score"], parts["lymph_logit"].sigmoid())
                loss = prognosis.nll_surv_loss(
                    hazards, survival, tensor([0]).long(), tensor([0]),
                    logits=parts["logits"],
                )
                loss.backward()
                for branch in (model.tumor_scorer, model.node_scorer):
                    self.assertGreater(gradient_norm(branch.score_head), 0.0)
                    self.assertGreater(gradient_norm(branch.encoder), 0.0)
                    for parameter in branch.parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_high_lr_synthetic_smoke_test_retains_distinct_risks(self):
        """A tiny CPU run catches the former all-patients-at-the-floor failure."""
        torch.manual_seed(23)
        model = small_model("mean_mil")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        bags = [torch.randn(7, 8) + shift for shift in (-1.0, -0.5, 0.5, 1.0)]
        bins = tensor([3, 3, 0, 0]).long()
        censorship = tensor([1, 1, 0, 0])
        for _ in range(20):
            optimizer.zero_grad()
            logits = []
            for bag in bags:
                _, _, parts = model(
                    bag, None, tensor([[1.0]]), tensor([[0.0]]), tensor([[0.0]]),
                )
                logits.append(parts["logits"])
            logits = torch.cat(logits)
            hazards = logits.sigmoid()
            loss = prognosis.nll_surv_loss(
                hazards, (1 - hazards).cumprod(dim=1),
                bins, censorship, logits=logits,
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            risks = torch.stack([
                model(bag, None, tensor([[1.0]]), tensor([[0.0]]), tensor([[0.0]]))[2]["rcb"].squeeze()
                for bag in bags
            ])
        self.assertTrue(torch.isfinite(risks).all())
        self.assertGreater(float(risks.max() - risks.min()), 1e-5)


if __name__ == "__main__":
    unittest.main()
