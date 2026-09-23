import numpy as np
import torch
import pytest

from data.data_generation import (
    generate_regression, generate_imbalanced_classification,
)
from models.optimization import (
    MAMConfig, Meta_Additive_models, train_erm_baseline,
)


SMALL = dict(epochs=30, eval_frequency=5, print_frequency=100, warmup_epochs=1)


def make_cfg(task, total_dim, spline_dim, **kw):
    base = dict(task=task, total_dimension=total_dim, spline_dim=spline_dim, seed=0)
    base.update(SMALL)
    base.update(kw)
    return MAMConfig(**base)


class TestRegressionFunctional:
    def test_training_reduces_test_mse(self):
        train_loader, val_loader, testX, testY = generate_regression(
            number=120, dimension=8, noise_type="mean", seed=1)
        cfg = make_cfg("regression", 24, 3)
        result = Meta_Additive_models(train_loader, val_loader, testX, testY,
                                      24, task="regression", config=cfg)
        hist = result["history"]
        assert all(np.isfinite(m) for m in hist["test_metric"])
        assert hist["test_metric"][-1] < hist["test_metric"][0]

    def test_mam_beats_unweighted_baseline_under_heavy_noise(self):
        """Paper claim: meta-learned weights are robust to the shifted/
        heavy-tailed noise (eps^A / eps^B / eps^C), so MAM should not be worse
        than plain ERM (and typically clearly better) on the simulation."""
        train_loader, val_loader, testX, testY = generate_regression(
            number=200, dimension=10, noise_type="mean", seed=3)
        cfg = make_cfg("regression", 30, 3, epochs=150, lowerlr=0.05,
                       eval_frequency=50)
        mam = Meta_Additive_models(train_loader, val_loader, testX, testY,
                                   30, task="regression", config=cfg)
        base = train_erm_baseline(train_loader, testX, testY, 30,
                                  task="regression", config=cfg)
        assert mam["best_metric"] <= base * 1.05

    def test_sparsity_selects_relevant_variables(self):
        """With a meaningful group-lasso coefficient, only covariates that
        enter f1..f8 should survive (additive-model feature selection)."""
        train_loader, val_loader, testX, testY = generate_regression(
            number=200, dimension=12, noise_type="Gaussian", seed=5)
        cfg = make_cfg("regression", 36, 3, epochs=80, lowerlr=0.05, penaltycoef=0.1)
        result = Meta_Additive_models(train_loader, val_loader, testX, testY,
                                      36, task="regression", config=cfg)
        n_sel, n_groups = result["selected_variables"]
        assert 0 < n_sel <= 8 and n_groups == 12


class TestClassificationFunctional:
    def test_imbalanced_training_reaches_high_accuracy(self):
        train_loader, val_loader, testX, testY = generate_imbalanced_classification(
            number=300, dimension=8, ratio=0.15, seed=2)
        cfg = make_cfg("classification", 40, 5, epochs=300, lowerlr=0.1,
                       batch_size=50, eval_frequency=100)
        result = Meta_Additive_models(train_loader, val_loader, testX, testY,
                                      40, task="classification", config=cfg)
        assert result["best_metric"] > 0.8
        assert all(np.isfinite(m) for m in result["history"]["test_metric"])

    def test_result_payload(self):
        train_loader, val_loader, testX, testY = generate_imbalanced_classification(
            number=100, dimension=5, ratio=0.2, seed=4)
        cfg = make_cfg("classification", 25, 5, epochs=10)
        result = Meta_Additive_models(train_loader, val_loader, testX, testY,
                                      25, task="classification", config=cfg)
        for key in ["model", "weighting_net", "config", "history",
                    "best_metric", "best_epoch", "best_state", "selected_variables"]:
            assert key in result
        assert result["best_state"] is not None


class TestWeightNormOption:
    def test_weight_norm_sum_runs(self):
        train_loader, val_loader, testX, testY = generate_regression(
            number=100, dimension=8, noise_type="Gaussian", seed=6)
        cfg = make_cfg("regression", 24, 3, weight_norm="sum")
        result = Meta_Additive_models(train_loader, val_loader, testX, testY,
                                      24, task="regression", config=cfg)
        assert np.isfinite(result["best_metric"])
