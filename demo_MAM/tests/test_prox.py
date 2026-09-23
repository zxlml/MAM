import math
import torch
import pytest

from models.optimization import (
    group_lasso_prox, group_lasso_prox_, group_penalty, count_selected_variables,
)


def analytic_group_prox(w, lam, eta, d):
    """Reference implementation: per-(row, group) soft-thresholding to zero."""
    out = w.clone()
    num_out, total = w.shape
    for i in range(num_out):
        for j in range(total // d):
            g = w[i, j * d:(j + 1) * d]
            ng = g.norm().item()
            factor = max(1.0 - lam * eta / ng, 0.0) if ng > 0 else 0.0
            out[i, j * d:(j + 1) * d] = g * factor
    return out


class TestGroupProx:
    @pytest.mark.parametrize("num_out,total,d", [(1, 30, 3), (2, 50, 5), (3, 12, 3)])
    def test_matches_analytic_soft_threshold(self, num_out, total, d):
        torch.manual_seed(0)
        w = torch.randn(num_out, total)
        lam, eta = 0.1, 0.05
        got = group_lasso_prox(w, lam, eta, d)
        want = analytic_group_prox(w, lam, eta, d)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

    def test_inplace_equals_functional(self):
        torch.manual_seed(1)
        w = torch.randn(2, 40)
        w_inplace = w.clone()
        group_lasso_prox_(w_inplace, 0.05, 0.01, 5)
        torch.testing.assert_close(w_inplace, group_lasso_prox(w, 0.05, 0.01, 5))

    def test_large_penalty_zeroes_groups(self):
        torch.manual_seed(2)
        w = torch.randn(1, 30)
        v = group_lasso_prox(w, lam=1e3, eta=1.0, spline_dim=3)
        assert torch.all(v == 0)

    def test_small_penalty_identity(self):
        torch.manual_seed(3)
        w = torch.randn(1, 30)
        v = group_lasso_prox(w, lam=0.0, eta=0.1, spline_dim=3)
        torch.testing.assert_close(v, w)

    def test_gradient_flows_through_prox(self):
        """The prox must stay in the autograd graph so the hypergradient
        accounts for the sparsity operator (the core bi-level fix)."""
        torch.manual_seed(4)
        w = torch.randn(1, 9, requires_grad=True)
        v = group_lasso_prox(w, lam=0.1, eta=0.01, spline_dim=3)
        v.sum().backward()
        assert w.grad is not None
        assert torch.any(w.grad != 0)

    def test_gradient_matches_finite_difference(self):
        torch.manual_seed(5)
        w = torch.randn(1, 12, dtype=torch.float64, requires_grad=True)
        lam, eta, d = 0.2, 0.05, 3

        def f(x):
            return group_lasso_prox(x, lam, eta, d).pow(2).sum()

        f(w).backward()
        eps = 1e-6
        fd = torch.zeros_like(w)
        with torch.no_grad():
            for idx in itertools_indices(w.numel()):
                orig = w.flatten()[idx].item()
                w.flatten()[idx] = orig + eps
                fp = f(w).item()
                w.flatten()[idx] = orig - eps
                fm = f(w).item()
                w.flatten()[idx] = orig
                fd.flatten()[idx] = (fp - fm) / (2 * eps)
        torch.testing.assert_close(w.grad, fd, rtol=1e-4, atol=1e-6)

    def test_penalty_value(self):
        w = torch.tensor([[3.0, 4.0, 0.0, 0.0, 0.0, 0.0]])
        # groups: ||(3,4,0)||=5 and 0 -> penalty 5
        assert group_penalty(w, spline_dim=3).item() == pytest.approx(5.0)

    def test_count_selected_variables(self):
        w = torch.zeros(1, 9)
        w[0, :3] = 1.0  # only first group active
        n_sel, n_groups = count_selected_variables(w, spline_dim=3, tol=1e-8)
        assert (n_sel, n_groups) == (1, 3)


def itertools_indices(n):
    return range(n)
