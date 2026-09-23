import numpy as np
import pytest
import torch

from data.data_generation import (
    Regression, Classfication_corrupted, Classfication_imbalance,
    generate_regression, generate_imbalanced_classification,
)


class TestRawGenerators:
    def test_regression_shapes(self):
        data = Regression(200, 20, noise="Gaussian")
        (trX, trY), (vaX, vaY), (teX, teY) = data.generate_data()
        assert trX.shape == (200, 20) and trY.shape == (200, 1)
        assert vaX.shape == (200, 20) and teX.shape == (200, 20)
        assert np.all(np.isfinite(trY))

    def test_regression_only_first_8_covariates_matter(self):
        """TSpAM simulation: Y = f1..f8 depends on the first 8 covariates only."""
        data = Regression(500, 50, noise="None")
        X = np.random.RandomState(0).uniform(-1, 1, (500, 50))
        y1 = data.generate_Y(X)
        X2 = X.copy()
        X2[:, 8:] += 1.0  # perturb irrelevant covariates
        y2 = data.generate_Y(X2)
        np.testing.assert_allclose(y1, y2)

    @pytest.mark.parametrize("noise", ["mean", "modal", "studentT", "mixGauss",
                                       "chiSquare", "Gaussian", "None"])
    def test_noise_types(self, noise):
        data = Regression(300, 10, noise=noise)
        (trX, trY), _, _ = data.generate_data()
        assert trY.shape == (300, 1) and np.all(np.isfinite(trY))

    def test_corrupted_labels_flipped(self):
        data = Classfication_corrupted(400, 10, frac=0.25)
        X = data.generate_X()
        clean = data.generate_Y(X).copy()
        dirty = data.add_noise(clean.copy())
        assert np.mean(dirty != clean) == pytest.approx(0.25, abs=0.02)
        assert set(np.unique(dirty)) <= {0, 1}

    def test_imbalance_ratio(self):
        data = Classfication_imbalance(1000, 10, frac=0.15)
        trX, trX_ = data.generate_X(), None
        y = data.generate_Y(trX)
        trX, trY = data.sample_data(trX, y, 0.15)
        neg_ratio = np.mean(trY == 0)
        assert neg_ratio == pytest.approx(0.15, abs=0.02)
        assert trX.shape[0] == 1000


class TestPipelineGenerators:
    def test_regression_pipeline_shapes(self):
        train_loader, val_loader, testX, testY = generate_regression(
            number=200, dimension=20, noise_type="mean", seed=7)
        batch_X, batch_Y = next(iter(train_loader))
        assert batch_X.shape[1] == 20 * 3  # r=3 spline basis for regression
        assert testX.shape == (200, 20 * 3)
        assert np.asarray(testY).shape == (200, 1)

    def test_imbalanced_pipeline_shapes(self):
        train_loader, val_loader, testX, testY = generate_imbalanced_classification(
            number=200, dimension=20, ratio=0.15, seed=7)
        batch_X, batch_Y = next(iter(train_loader))
        assert batch_X.shape[1] == 20 * 5  # r=5 spline basis for classification
        assert testX.shape == (200, 20 * 5)

    def test_seed_reproducibility(self):
        a = generate_regression(number=100, dimension=8, noise_type="Gaussian", seed=3)
        b = generate_regression(number=100, dimension=8, noise_type="Gaussian", seed=3)
        (xa, ya) = next(iter(a[0]))
        (xb, yb) = next(iter(b[0]))
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
        np.testing.assert_allclose(np.asarray(a[2]), np.asarray(b[2]))
