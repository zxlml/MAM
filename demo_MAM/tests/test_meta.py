import torch
import torch.nn.functional as F
import pytest

from models.optimization import (
    MetaModule, MetaLinear, UpperModel, LowerModel, build_model, to_var,
    group_lasso_prox, step1_inner, step2_upper, MAMConfig, DEVICE,
)
from data.data_generation import generate_regression


def clone_double(module):
    for name, p in module.named_params(module):
        module.set_param(module, name, to_var(p.data.clone().double(), requires_grad=True))
    return module


def make_double_models(dim=9, hidden=4):
    torch.manual_seed(0)
    model = clone_double(LowerModel(dim, 1))
    vnet = clone_double(UpperModel(1, hidden, 1))
    return model, vnet


def bilevel_value(model, vnet, Xtr, ytr, Xval, yval, lam, eta, d):
    """L(theta): validation loss after one unrolled proximal-gradient inner
    step -- exact same operations as step1_inner, in float64 for FD checks."""
    meta = clone_double(LowerModel(Xtr.size(1), 1))
    for name, p_src in model.named_params(model):
        meta.set_param(meta, name, to_var(p_src.data.clone().double(), requires_grad=True))
    cost = F.mse_loss(meta(Xtr), ytr, reduction="none").view(-1, 1)
    v_lambda = vnet(cost.data)
    l_f_meta = torch.sum(cost * v_lambda) / len(cost)
    grads = torch.autograd.grad(l_f_meta, tuple(meta.params()), create_graph=True)
    meta.update_params(lr_inner=eta, source_params=grads)
    meta.set_param(meta, "predict.weight",
                   group_lasso_prox(meta.predict.weight, lam, eta, d))
    return F.mse_loss(meta(Xval), yval)


class TestMetaModule:
    def test_update_params_matches_manual_sgd(self):
        torch.manual_seed(0)
        lin = MetaLinear(3, 1)
        w0 = lin.weight.detach().clone()
        g = torch.randn_like(w0)
        lr = 0.1
        lin.update_params(lr_inner=lr, source_params=[g])
        torch.testing.assert_close(lin.weight.detach(), w0 - lr * g)

    def test_named_params_covers_weight_and_bias(self):
        lin = MetaLinear(3, 2)
        names = [n for n, _ in lin.named_params(lin)]
        assert names == ["weight", "bias"]


class TestHypergradient:
    def test_hypergradient_matches_finite_difference(self):
        """Core convergence check: d L_val(beta_hat(theta)) / d theta computed
        through the unrolled graph equals the finite-difference gradient of the
        value function (including the differentiable proximal operator)."""
        torch.manual_seed(1)
        d, n, n_val = 9, 16, 8
        Xtr = torch.randn(n, d, dtype=torch.float64)
        ytr = torch.randn(n, 1, dtype=torch.float64)
        Xval = torch.randn(n_val, d, dtype=torch.float64)
        yval = torch.randn(n_val, 1, dtype=torch.float64)
        model, vnet = make_double_models(dim=d)
        lam, eta = 0.05, 0.05

        l_g = bilevel_value(model, vnet, Xtr, ytr, Xval, yval, lam, eta, d)
        l_g.backward()
        autodiff = {n_: p.grad.clone() for n_, p in vnet.named_params(vnet)}

        eps = 1e-6
        for name, p in vnet.named_params(vnet):
            g_fd = torch.zeros_like(p)
            flat = p.view(-1)
            g_flat = g_fd.view(-1)
            for idx in range(flat.numel()):
                orig = flat[idx].item()
                with torch.no_grad():
                    flat[idx] = orig + eps
                lp = bilevel_value(model, vnet, Xtr, ytr, Xval, yval, lam, eta, d).item()
                with torch.no_grad():
                    flat[idx] = orig - eps
                lm = bilevel_value(model, vnet, Xtr, ytr, Xval, yval, lam, eta, d).item()
                with torch.no_grad():
                    flat[idx] = orig
                g_flat[idx] = (lp - lm) / (2 * eps)
            torch.testing.assert_close(autodiff[name], g_fd, rtol=1e-4, atol=1e-8)


class TestBiLevelStepsIntegration:
    def test_step1_step2_run_and_update_vnet(self):
        train_loader, val_loader, testX, testY = generate_regression(
            number=64, dimension=8, noise_type="Gaussian", seed=0)
        cfg = MAMConfig(task="regression", total_dimension=24, spline_dim=3)
        model = build_model(24, "regression")
        vnet = UpperModel(1, 10, 1).to(DEVICE)
        opt_vnet = torch.optim.Adam(vnet.params(), 1e-3)

        before = [p.detach().clone() for p in vnet.params()]
        meta_model, xv, tv = step1_inner(*next(iter(train_loader)), model, vnet, cfg,
                                         lr_inner=1e-2, warmup=False)
        val_iter = iter(val_loader)
        val_loss, val_iter = step2_upper(meta_model, val_iter, val_loader, vnet,
                                         opt_vnet, cfg, warmup=False)
        after = [p.detach().clone() for p in vnet.params()]
        assert any(not torch.equal(a, b) for a, b in zip(before, after))
        assert val_loss > 0

    def test_warmup_freezes_vnet(self):
        train_loader, val_loader, testX, testY = generate_regression(
            number=64, dimension=8, noise_type="Gaussian", seed=0)
        cfg = MAMConfig(task="regression", total_dimension=24, spline_dim=3)
        model = build_model(24, "regression")
        vnet = UpperModel(1, 10, 1).to(DEVICE)
        opt_vnet = torch.optim.Adam(vnet.params(), 1e-3)

        before = [p.detach().clone() for p in vnet.params()]
        meta_model, xv, tv = step1_inner(*next(iter(train_loader)), model, vnet, cfg,
                                         lr_inner=1e-2, warmup=True)
        val_loss, _ = step2_upper(meta_model, val_loader and iter(val_loader), val_loader,
                                  vnet, opt_vnet, cfg, warmup=True)
        after = [p.detach().clone() for p in vnet.params()]
        assert all(torch.equal(a, b) for a, b in zip(before, after))

    def test_val_iterator_recycles(self):
        """Persistent iterator must recycle instead of rebuilding each call
        (MW-Net engineering; also guards against batches being skipped)."""
        train_loader, val_loader, testX, testY = generate_regression(
            number=64, dimension=8, noise_type="Gaussian", seed=0)
        cfg = MAMConfig(task="regression", total_dimension=24, spline_dim=3)
        model = build_model(24, "regression")
        vnet = UpperModel(1, 10, 1).to(DEVICE)
        opt_vnet = torch.optim.Adam(vnet.params(), 1e-3)

        meta_model, _, _ = step1_inner(*next(iter(train_loader)), model, vnet, cfg,
                                       lr_inner=1e-2, warmup=True)
        val_iter = iter(val_loader)
        for _ in range(5 * len(val_loader)):  # far more batches than the loader has
            _, val_iter = step2_upper(meta_model, val_iter, val_loader, vnet,
                                      opt_vnet, cfg, warmup=True)
