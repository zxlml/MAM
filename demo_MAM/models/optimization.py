# -*- coding: utf-8 -*-
"""
Meta Additive Model (MAM) -- refactored bi-level optimization.

Why the scheme must differ from MW-Net
--------------------------------------
MW-Net learns per-sample weights for a *smooth* deep network whose inner
solver is plain SGD, so the hypergradient only flows through the unrolled
gradient step. Here the lower level is a *sparse additive model* fitted on
B-spline basis features:

    upper:  min_theta  L_val( beta_hat(theta) )
    lower:  beta_hat(theta) = prox_{lam*eta*Omega}( beta - eta * (1/n) sum_i
                        sigma(f_theta(cost_i(beta))) * grad_beta cost_i(beta) )

i.e. the inner problem is a *composite* (smooth loss + non-smooth group-lasso
penalty Omega) convex problem. The inner solver is therefore a proximal
gradient step, and the proximal operator is part of the solution map. For the
bi-level recursion to converge, the hypergradient must account for the proximal
operator as well: we keep it inside the autograd graph (its Jacobian exists
a.e. away from the kink ||w_g|| = lam*eta). The production update for the lower
model applies the identical proximal operator in-place on .data (exact
composite update, no graph needed).

What is borrowed from MW-Net (paper + MW-Net.py engineering)
------------------------------------------------------------
* single unrolled inner step with create_graph=True (memory-friendly
  hypergradient through one gradient step);
* the weighting network is evaluated on *detached* per-sample costs
  (vnet(cost.data)) inside the inner loss;
* one validation batch per training batch, drawn from a persistent validation
  iterator (with StopIteration recycling);
* Adam on the weighting net (MW-Net uses Adam(1e-3, wd=1e-4));
* warm-up: for the first `warmup_epochs` epochs all sample weights are fixed
  to one (MW-Net paper, Sec. "warm-up"), so the bi-level recursion starts from
  a sensible beta and the weighting net is frozen;
* the meta (ghost) model is rebuilt from the production model every iteration,
  exactly like MW-Net's build_model()/load_state_dict() pattern.
"""

import os
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("MAM")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = get_device()


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Meta-learning modules (adopted from MW-Net, Adrien Ecoffet's MetaModule)
# =============================================================================
def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return x.clone().detach().requires_grad_(requires_grad)


class MetaModule(nn.Module):
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


# =============================================================================
# Upper level: the weighting network  theta -> sigma(f_theta(cost))
# Single hidden-layer MLP with sigmoid output (paper: 100 hidden nodes).
# =============================================================================
class UpperModel(MetaModule):
    def __init__(self, input, hidden1=100, output=1):
        super(UpperModel, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


# =============================================================================
# Lower level: sparse additive model = linear model on B-spline basis features
#   regression:      f(x) = beta^T phi(x) + b
#   classification:  logits = beta^T phi(x) + b   (softmax / cross-entropy)
# The group-lasso penalty couples the `spline_dim` coefficients of each
# covariate (L2,1 over R^{spline_dim} groups), inducing variable selection.
# =============================================================================
class LowerModel(MetaModule):
    def __init__(self, n_feature, n_output=1):
        super(LowerModel, self).__init__()
        self.predict = MetaLinear(n_feature, n_output)

    def forward(self, x):
        return self.predict(x)


class LowerModel_classification(MetaModule):
    def __init__(self, n_feature, n_output=2):
        super(LowerModel_classification, self).__init__()
        self.predict = MetaLinear(n_feature, n_output)

    def forward(self, x):
        return self.predict(x)


def build_model(dimension, task, num_classes=2):
    if task == 'regression':
        model = LowerModel(dimension, 1)
    else:
        model = LowerModel_classification(dimension, num_classes)
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
    return model


# =============================================================================
# Group-lasso proximal operator (sparsity-induced regularization)
#   w: (num_outputs, P*spline_dim) -> group g_j = w[:, j*d:(j+1)*d]
#   prox: w_g <- w_g * max(1 - lam*eta/||w_g||_2, 0)
# This is the exact proximal step of the group-L2,1 penalty (equivalently, a
# gradient step on lam*Omega away from the kink), so evaluating it inside the
# autograd graph yields hypergradients consistent with the composite inner
# problem. The original implementation (a) reshaped the classification weight
# matrix wrongly (mixing classes and splitting covariates across groups) and
# (b) detached it via .data assignment, so the hypergradient ignored the
# sparsity operator entirely; both issues are fixed here.
# =============================================================================
def _group_norms(w, spline_dim):
    num_out, total = w.shape
    n_groups = total // spline_dim
    return w.reshape(num_out, n_groups, spline_dim).norm(p=2, dim=2, keepdim=True), n_groups


def group_lasso_prox(w, lam, eta, spline_dim):
    """Differentiable (a.e.) group-lasso proximal step; returns a new tensor
    that stays connected to the autograd graph."""
    norms, _ = _group_norms(w, spline_dim)
    alpha = torch.clamp(1.0 - lam * eta / (norms + 1e-12), min=0.0)
    num_out, total = w.shape
    return (w.reshape(num_out, -1, spline_dim) * alpha).reshape(num_out, total)


@torch.no_grad()
def group_lasso_prox_(w, lam, eta, spline_dim):
    """In-place proximal step for the production lower-model update."""
    norms, _ = _group_norms(w, spline_dim)
    alpha = torch.clamp(1.0 - lam * eta / (norms + 1e-12), min=0.0)
    w.copy_((w.reshape(w.shape[0], -1, spline_dim) * alpha).reshape(w.shape))


def group_penalty(w, spline_dim):
    """sum of group L2 norms (group L2,1 norm up to the outer sum), for logging
    or for adding the penalty explicitly into a loss."""
    norms, _ = _group_norms(w, spline_dim)
    return norms.sum()


def count_selected_variables(w, spline_dim, tol=1e-8):
    """Number of covariate groups with non-zero coefficients (feature selection
    diagnostic for the additive model)."""
    with torch.no_grad():
        norms, n_groups = _group_norms(w, spline_dim)
        group_active = (norms.sum(dim=0).squeeze(-1) > tol)
        return int(group_active.sum().item()), n_groups


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class MAMConfig:
    task: str = 'regression'            # 'regression' | 'classification'
    total_dimension: int = 300          # P * spline_dim (spline-basis features)
    spline_dim: int = 3                 # r: 3 for regression, 5 for classification
    num_classes: int = 2
    epochs: int = 1000
    lowerlr: float = 5e-2               # inner / lower-level SGD lr
    upperlr: float = 1e-3               # weighting-net Adam lr (MW-Net engineering)
    penaltycoef: float = 1e-3           # group-lasso coefficient lam (large enough
                                        # to zero out noise covariates at the paper
                                        # scale; the original 1e-5 selected 100/100)
    hidden: int = 100                   # weighting net hidden nodes
    warmup_epochs: int = 1              # equal-weight warm-up (MW-Net paper)
    weight_norm: str = 'none'           # 'none' (paper/MW-Net) | 'sum' (weighted mean)
    lr_milestones: tuple = (0.5, 0.8)   # fractions of epochs for lr decay
    lr_gamma: float = 0.1
    print_frequency: int = 50
    eval_frequency: int = 50
    batch_size: int = 200
    seed: int = None
    max_grad_norm: float = 5.0          # gradient clipping: the simulation noises
                                        # (eps^A/eps^B/eps^C) contain huge outliers whose
                                        # squared-error gradients would otherwise blow up
                                        # the SGD lower-level update into NaN
    log_file: str = None

    def __post_init__(self):
        if self.spline_dim is None:
            self.spline_dim = 3 if self.task == 'regression' else 5


def _setup_logging(log_file=None):
    handlers = [logging.StreamHandler()]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=handlers, force=True)


# =============================================================================
# Per-sample losses
# =============================================================================
def per_sample_loss(pred, target, task):
    if task == 'regression':
        return F.mse_loss(pred, target, reduction='none').view(-1, 1)
    return F.cross_entropy(pred, target, reduction='none').view(-1, 1)


def full_loss(pred, target, task):
    if task == 'regression':
        return F.mse_loss(pred, target)
    return F.cross_entropy(pred, target)


def prepare_batch(input, target, task, device):
    input = input.to(device).float()
    if task == 'regression':
        target = target.to(device).float()
        if target.dim() == 1:
            target = target.view(-1, 1)
    else:
        target = target.to(device).long().view(-1)
    return input, target


# =============================================================================
# Bi-level steps
# =============================================================================
def step1_inner(input, target, model, weighting_net, cfg, lr_inner, warmup):
    """Step 1: build the ghost model beta_hat(theta) with one unrolled,
    differentiable proximal-gradient inner step."""
    meta_model = build_model(input.size(1), cfg.task, cfg.num_classes)
    # clone every leaf weight from the production model (no stale autograd
    # history is reused, equivalent to MW-Net's build_model()+load_state_dict)
    for name, p_src in model.named_params(model):
        meta_model.set_param(meta_model, name, to_var(p_src.data.clone(), requires_grad=True))

    input_var, target_var = prepare_batch(input, target, cfg.task, DEVICE)
    y_f_hat = meta_model(input_var)
    cost_v = per_sample_loss(y_f_hat, target_var, cfg.task)

    if warmup:
        # MW-Net warm-up: uniform weights, weighting net not involved.
        v_lambda = torch.ones_like(cost_v)
    else:
        v_lambda = weighting_net(cost_v.data)

    l_f_meta = torch.sum(cost_v * v_lambda) / len(cost_v)

    meta_model.zero_grad()
    grads = torch.autograd.grad(l_f_meta, tuple(meta_model.params()), create_graph=True)
    meta_model.update_params(lr_inner=lr_inner, source_params=grads)
    del grads
    # Composite inner solver: the proximal operator belongs to the solution map
    # and stays in the graph so the hypergradient accounts for the sparsity step.
    prox_w = group_lasso_prox(meta_model.predict.weight, cfg.penaltycoef, lr_inner,
                              cfg.spline_dim)
    meta_model.set_param(meta_model, 'predict.weight', prox_w)
    return meta_model, input_var, target_var


def step2_upper(meta_model, val_iter, validation_loader, weighting_net, optimizer_vnet,
                cfg, warmup):
    """Step 2: one validation batch -> hypergradient -> update theta."""
    try:
        input_validation, target_validation = next(val_iter)
    except StopIteration:
        val_iter = iter(validation_loader)
        input_validation, target_validation = next(val_iter)
    input_validation, target_validation = prepare_batch(input_validation, target_validation,
                                                        cfg.task, DEVICE)
    y_g_hat = meta_model(input_validation)
    l_g_meta = full_loss(y_g_hat, target_validation, cfg.task)

    if not warmup and torch.isfinite(l_g_meta):
        optimizer_vnet.zero_grad()
        l_g_meta.backward()
        if cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(weighting_net.params(), cfg.max_grad_norm)
        optimizer_vnet.step()
    return l_g_meta.item(), val_iter


def step3_outer(input_var, target_var, model, weighting_net, optimizer_a, cfg, lr_outer,
                warmup):
    """Step 3: update the production lower model beta with the learned weights."""
    y_f = model(input_var)
    cost_v = per_sample_loss(y_f, target_var, cfg.task)
    prec_train = full_loss(y_f, target_var, cfg.task).item()

    with torch.no_grad():
        if warmup:
            w_new = torch.ones_like(cost_v)
        else:
            w_new = weighting_net(cost_v)
        if cfg.weight_norm == 'sum':
            norm_v = torch.sum(w_new)
            w_v = w_new / norm_v if norm_v != 0 else w_new
        else:
            # paper / MW-Net objective: (1/n) sum_i sigma(f_theta(cost_i)) * cost_i
            w_v = w_new

    l_f = torch.sum(cost_v * w_v) / len(cost_v)
    if not torch.isfinite(l_f):
        # numerical guard: a corrupted batch (extreme outlier cost) must not
        # poison the production model with NaN weights
        return prec_train, l_f.item()
    optimizer_a.zero_grad()
    l_f.backward()
    if cfg.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.params(), cfg.max_grad_norm)
    optimizer_a.step()
    # exact composite update for the production weights (no graph needed)
    group_lasso_prox_(model.predict.weight.data, cfg.penaltycoef, lr_outer, cfg.spline_dim)
    return prec_train, l_f.item()


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def evaluate(testX_tensor, testY, model, task):
    model.eval()
    y_pred = model(testX_tensor)
    if task == 'regression':
        pred = y_pred.cpu().numpy().reshape(-1)
        return float(np.mean((pred - np.asarray(testY).reshape(-1)) ** 2))
    target = torch.as_tensor(np.asarray(testY).reshape(-1), device=testX_tensor.device).long()
    _, predicted = y_pred.max(1)
    correct = predicted.eq(target).sum().item()
    acc = correct / len(target)
    # per-class accuracy (useful under class imbalance)
    per_class = []
    for c in range(y_pred.size(1)):
        mask = target == c
        if mask.sum() > 0:
            per_class.append(predicted[mask].eq(target[mask]).sum().item() / mask.sum().item())
        else:
            per_class.append(float('nan'))
    return acc, per_class


# =============================================================================
# Main entry: MAM bi-level training
# =============================================================================
def Meta_Additive_models(train_loader, validation_loader, testX, testY, total_dimension,
                         task='regression', config=None):
    cfg = config if config is not None else MAMConfig(
        task=task, total_dimension=total_dimension,
        spline_dim=3 if task == 'regression' else 5)
    if cfg.spline_dim is None:
        cfg.spline_dim = 3 if task == 'regression' else 5
    set_seed(cfg.seed)
    _setup_logging(cfg.log_file)

    # lower-level model
    model = build_model(cfg.total_dimension, cfg.task, cfg.num_classes)
    # upper-level weighting network (single hidden layer, sigmoid output)
    weighting_net = UpperModel(1, cfg.hidden, 1)
    if torch.cuda.is_available():
        weighting_net.cuda()

    optimizer_a = torch.optim.SGD(model.params(), cfg.lowerlr)
    optimizer_vnet = torch.optim.Adam(weighting_net.params(), cfg.upperlr, weight_decay=1e-4)

    testX_tensor = torch.as_tensor(np.asarray(testX), dtype=torch.float32).to(DEVICE)

    milestones = [int(m * cfg.epochs) for m in cfg.lr_milestones]

    def lr_at(epoch):
        lr = cfg.lowerlr
        for m in milestones:
            lr *= cfg.lr_gamma if epoch >= m else 1.0
        return lr

    history = {'test_metric': [], 'val_loss': [], 'train_loss': [], 'epoch': []}
    best_metric = None
    best_epoch = -1
    best_state = None

    for epoch in range(cfg.epochs):
        lr = lr_at(epoch)
        optimizer_a.param_groups[0]['lr'] = lr
        warmup = epoch < cfg.warmup_epochs

        val_iter = iter(validation_loader)
        epoch_val, epoch_train, n_batches = 0.0, 0.0, 0
        for input, target in train_loader:
            model.train()
            # Step 1: ghost model beta_hat(theta) with unrolled prox-grad step
            meta_model, input_var, target_var = step1_inner(
                input, target, model, weighting_net, cfg, lr, warmup)
            # Step 2: hypergradient -> update weighting net theta
            val_loss, val_iter = step2_upper(meta_model, val_iter, validation_loader,
                                             weighting_net, optimizer_vnet, cfg, warmup)
            # Step 3: update production beta with learned weights
            train_loss, weighted_loss = step3_outer(input_var, target_var, model,
                                                    weighting_net, optimizer_a, cfg, lr,
                                                    warmup)
            epoch_val += val_loss
            epoch_train += train_loss
            n_batches += 1

        if (epoch + 1) % cfg.eval_frequency == 0 or epoch == 0 or epoch == cfg.epochs - 1:
            if cfg.task == 'regression':
                metric = evaluate(testX_tensor, testY, model, cfg.task)
                logger.info('Epoch %d\tlr %.2g\tVal Loss %.4f\tTrain Loss %.4f\tTest MSE %.4f',
                            epoch + 1, lr, epoch_val / max(n_batches, 1),
                            epoch_train / max(n_batches, 1), metric)
            else:
                metric, per_class = evaluate(testX_tensor, testY, model, cfg.task)
                logger.info('Epoch %d\tlr %.2g\tVal Loss %.4f\tTrain Loss %.4f\tTest Acc %.4f\tPer-class %s',
                            epoch + 1, lr, epoch_val / max(n_batches, 1),
                            epoch_train / max(n_batches, 1), metric,
                            np.round(per_class, 4).tolist())
            history['test_metric'].append(metric)
            history['val_loss'].append(epoch_val / max(n_batches, 1))
            history['train_loss'].append(epoch_train / max(n_batches, 1))
            history['epoch'].append(epoch + 1)

            score = metric if cfg.task == 'regression' else metric
            improved = (best_metric is None or
                        (cfg.task == 'regression' and score < best_metric) or
                        (cfg.task == 'classification' and score > best_metric))
            if improved:
                best_metric = score
                best_epoch = epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % cfg.print_frequency == 0:
            n_sel, n_groups = count_selected_variables(model.predict.weight, cfg.spline_dim)
            logger.info('Epoch %d\tselected variables %d/%d', epoch + 1, n_sel, n_groups)

    n_sel, n_groups = count_selected_variables(model.predict.weight, cfg.spline_dim)
    logger.info('Finished. best %s = %.4f @ epoch %d; selected variables %d/%d',
                'MSE' if cfg.task == 'regression' else 'Acc', best_metric, best_epoch,
                n_sel, n_groups)

    return {
        'model': model,
        'weighting_net': weighting_net,
        'config': cfg,
        'history': history,
        'best_metric': best_metric,
        'best_epoch': best_epoch,
        'best_state': best_state,
        'selected_variables': (n_sel, n_groups),
    }


# =============================================================================
# Unweighted ERM baseline (plain empirical risk minimization with the same
# composite proximal-gradient solver) -- reference point for the simulation.
# =============================================================================
def train_erm_baseline(train_loader, testX, testY, total_dimension, task='regression',
                       config=None):
    cfg = config if config is not None else MAMConfig(
        task=task, total_dimension=total_dimension,
        spline_dim=3 if task == 'regression' else 5)
    set_seed(cfg.seed)

    model = build_model(cfg.total_dimension, cfg.task, cfg.num_classes)
    optimizer_a = torch.optim.SGD(model.params(), cfg.lowerlr)
    testX_tensor = torch.as_tensor(np.asarray(testX), dtype=torch.float32).to(DEVICE)
    milestones = [int(m * cfg.epochs) for m in cfg.lr_milestones]

    best_metric = None
    for epoch in range(cfg.epochs):
        lr = cfg.lowerlr
        for m in milestones:
            lr *= cfg.lr_gamma if epoch >= m else 1.0
        optimizer_a.param_groups[0]['lr'] = lr
        for input, target in train_loader:
            model.train()
            input_var, target_var = prepare_batch(input, target, cfg.task, DEVICE)
            y_f = model(input_var)
            loss = full_loss(y_f, target_var, cfg.task)
            if not torch.isfinite(loss):
                continue
            optimizer_a.zero_grad()
            loss.backward()
            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.params(), cfg.max_grad_norm)
            optimizer_a.step()
            group_lasso_prox_(model.predict.weight.data, cfg.penaltycoef, lr, cfg.spline_dim)
        if (epoch + 1) % cfg.eval_frequency == 0 or epoch == cfg.epochs - 1:
            metric = evaluate(testX_tensor, testY, model, cfg.task)
            metric = metric[0] if isinstance(metric, tuple) else metric
            improved = (best_metric is None or
                        (cfg.task == 'regression' and metric < best_metric) or
                        (cfg.task == 'classification' and metric > best_metric))
            if improved:
                best_metric = metric
    return best_metric
