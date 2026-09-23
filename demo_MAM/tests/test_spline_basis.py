import numpy as np
import pytest

from data.data_generation import bsplinebasis, transform_splines


class TestBSplineBasis:
    @pytest.mark.parametrize("r", [3, 5])
    def test_partition_of_unity(self, r):
        """B-spline basis must sum to 1 on the whole closed interval,
        including the right boundary (fixed bug)."""
        x = np.linspace(-1.0, 1.0, 41)
        B = bsplinebasis(x, [-1.0, 1.0], r)
        assert B.shape == (41, r)
        np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-10)

    @pytest.mark.parametrize("r", [3, 5])
    def test_nonnegativity(self, r):
        x = np.linspace(-1.0, 1.0, 101)
        B = bsplinebasis(x, [-1.0, 1.0], r)
        assert np.all(B >= 0.0)

    def test_boundary_points(self):
        B = bsplinebasis(np.array([-1.0, 1.0]), [-1.0, 1.0], 3)
        np.testing.assert_allclose(B[0], [1.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(B[-1], [0.0, 0.0, 1.0], atol=1e-10)


class TestTransformSplines:
    def _make(self, n=50, p=4):
        rng = np.random.RandomState(0)
        return (rng.uniform(-1, 1, (n, p)), rng.uniform(-1, 1, (n, p)),
                rng.uniform(-1, 1, (n, p)))

    @pytest.mark.parametrize("r", [3, 5])
    def test_width_and_finiteness(self, r):
        trX, vaX, teX = self._make()
        tr, va, te = transform_splines(trX, vaX, teX, r)
        assert tr.shape == (50, 4 * r)
        assert np.all(np.isfinite(tr)) and np.all(np.isfinite(va)) and np.all(np.isfinite(te))

    @pytest.mark.parametrize("r", [3, 5])
    def test_no_nan(self, r):
        trX, vaX, teX = self._make()
        tr, va, te = transform_splines(trX, vaX, teX, r)
        assert np.all(np.isfinite(tr)) and np.all(np.isfinite(va)) and np.all(np.isfinite(te))
