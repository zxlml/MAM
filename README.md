# Meta Additive Models (MAM)

**English** | [简体中文](README_zh.md)

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-45%20passed-brightgreen)](demo_MAM/tests)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A faithful **reproduction and engineering refactoring of Meta Additive Models (MAM)** — a bi-level meta-learning framework that learns per-sample weights (via a weighting network, in the spirit of [MW-Net](https://github.com/xjtushujun/meta-weight-net)) for **sparse additive models** fitted on B-spline basis features with a group-lasso penalty.

> ⭐ If you find this project useful, please give it a star!

## ✨ Highlights

* 🚀 **Bi-level meta optimization** — single unrolled inner step with `create_graph=True`, MW-Net style ghost-model rebuilding, persistent validation iterator, and warm-up epochs.
* 🎯 **Composite lower-level solver** — the inner problem is *smooth loss + non-smooth group-lasso*, so the **proximal operator is kept inside the autograd graph**: the hypergradient correctly accounts for the sparsity step (this is the key difference from MW-Net, whose lower level is a smooth deep network).
* 🧮 **Numerically stable spline design** — per-block SVD whitening of the Bernstein/B-spline basis fitted on the train split (removes exact cross-block collinearity, condition number ~1e15 → ~1), with unit-variance rescaling so plain SGD converges.
* 🛡️ **Robust training** — gradient clipping + non-finite-loss guards against the extreme outliers in the paper's noise models (ε^A/ε^B/ε^C).
* ✅ **Well tested** — 45 unit/functional tests, including finite-difference verification of the hypergradient through the differentiable prox.

## 📢 News

* **[2026-09]** 🚀 Major refactoring: differentiable group-lasso prox, bi-level loop rewritten after MW-Net's engineering, train-split knots + SVD whitening for the spline basis.
* **[2026-09]** ✅ Full test suite added (unit + functional, 45 tests).
* **[2026-09]** 📊 Paper-scale simulations (n=2000, p=100) completed — raw logs and a summary CSV are available under `demo_MAM/logs/`.

## 📑 Table of Contents

* [✨ Highlights](#-highlights)
* [📢 News](#-news)
* [🔧 Installation](#-installation)
* [⚡ Quick Start](#-quick-start)
* [🏗️ Architecture](#️-architecture)
* [🧠 Algorithm Design](#-algorithm-design)
* [🧪 Testing](#-testing)
* [☑️ Todo](#️-todo)
* [🙏 Acknowledgements](#-acknowledgements)

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/zxlml/MAM.git
cd MAM/demo_MAM

# Create virtual environment (Python 3.10+)
conda create -n mam python=3.10 -y
conda activate mam

# Install dependencies
pip install torch numpy scipy scikit-learn matplotlib pytest
```

> 💡 On Windows with Anaconda, set `KMP_DUPLICATE_LIB_OK=TRUE` to avoid the OpenMP runtime conflict (`libiomp5md.dll`).

## ⚡ Quick Start

### 1. Command line (recommended)

Regression with the paper's noise models (`mean`/`modal`/`studentT` correspond to ε^A/ε^B/ε^C):

```bash
python main.py --task regression --noise_type mean \
    --number 2000 --dimension 100 --epochs 1000 --seed 1 --baseline
```

Classification scenarios (`imbalance` / `corrupted` / `multi`):

```bash
python main.py --task classification --scenario imbalance \
    --number 2000 --dimension 100 --epochs 500 --seed 1 --baseline
```

Key arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--task` | `regression` | `regression` or `classification` |
| `--noise_type` | `mean` | Regression noise: `None/Gaussian/mean/modal/studentT/chiSquare/mixGauss` |
| `--scenario` | `imbalance` | Classification scenario: `imbalance/corrupted/multi` |
| `--number` | `2000` | Samples per split |
| `--dimension` | `100` | Number of covariates p |
| `--epochs` | `1000` | Training epochs |
| `--lowerlr` | `5e-2` | Lower-level (additive model) SGD lr |
| `--upperlr` | `1e-3` | Weighting-network Adam lr |
| `--penaltycoef` | `1e-3` | Group-lasso coefficient λ |
| `--warmup` | `1` | Warm-up epochs (uniform weights) |
| `--baseline` | off | Also run the unweighted ERM baseline |

### 2. Python API

```python
from data.data_generation import generate_regression, generate_imbalanced_classification
from models.optimization import Meta_Additive_models, MAMConfig

# Regression (paper setting: n=2000, p=100, noise = eps^A)
train_loader, val_loader, testX, testY = generate_regression(
    number=2000, dimension=100, noise_type='mean', seed=1)

cfg = MAMConfig(task='regression', total_dimension=100*3, spline_dim=3,
                epochs=1000, lowerlr=5e-2, penaltycoef=1e-3, seed=1)
result = Meta_Additive_models(train_loader, val_loader, testX, testY,
                              total_dimension=100*3, task='regression', config=cfg)
print(result['best_metric'], result['best_epoch'], result['selected_variables'])

# Classification (imbalance / corrupted / multi)
train_loader, val_loader, testX, testY = generate_imbalanced_classification(
    number=2000, dimension=100, ratio=0.15, seed=1)
cfg = MAMConfig(task='classification', total_dimension=100*5, spline_dim=5,
                epochs=500, lowerlr=5e-2, penaltycoef=1e-3, seed=1)
result = Meta_Additive_models(train_loader, val_loader, testX, testY,
                              total_dimension=100*5, task='classification', config=cfg)
```

## 🏗️ Architecture

```
MAM/
├── demo_MAM/
│   ├── main.py                  # CLI entry: data generation + bi-level training + baseline
│   ├── data/
│   │   └── data_generation.py   # TSpAM-style simulations; B-spline basis + SVD whitening
│   ├── models/
│   │   └── optimization.py      # bi-level solver, differentiable prox, ERM baseline
│   ├── tests/                   # 45 unit / functional tests (pytest)
│   ├── baselines/               # original baseline implementations
│   └── logs/                    # training / simulation logs
└── README.md
```

## 🧠 Algorithm Design

The bi-level problem differs fundamentally from MW-Net because the lower level is a *composite convex* problem:

```text
upper:  min_theta  L_val( beta_hat(theta) )
lower:  beta_hat(theta) = prox_{lam*eta*Omega}( beta - eta * (1/n) sum_i
                     sigma(f_theta(cost_i(beta))) * grad_beta cost_i(beta) )
```

* **Differentiable prox in the hypergradient.** The group-lasso proximal operator `w_g ← w_g·max(1 − λη/‖w_g‖₂, 0)` is part of the solution map, so it is kept **inside the autograd graph** during the unrolled inner step; the production model applies the identical prox in-place on `.data` (exact composite update). The original implementation detached the prox via `.data`, silently dropping the sparsity operator from the hypergradient.
* **What is borrowed from MW-Net.** Single unrolled inner step (`create_graph=True`), weighting net evaluated on detached costs (`vnet(cost.data)`), one validation batch per training batch from a persistent iterator (StopIteration recycling), Adam on the weighting net, and warm-up epochs with uniform weights.
* **Numerical fixes for the spline basis.** Knots are taken from the **train split only** and reused for valid/test (the original code re-fitted knots per split, so the three splits were mapped by different feature maps); each Bernstein block is SVD-whitened on train (the basis spans the constant function → exact cross-block collinearity, condition number ~1e15) and rescaled to unit variance so that plain SGD solvers converge.
* **Group-lasso reshaping fix.** The original proximal step reshaped the classification weight matrix wrongly (mixing classes and splitting covariates across groups); groups are now formed as contiguous `spline_dim` blocks per output.

## 🧪 Testing

```bash
cd demo_MAM
python -m pytest tests -q        # 45 tests: spline basis, data generation, prox,
                                 # hypergradient FD check, bi-level steps, functional
```

## ☑️ Todo

* [ ] Multi-seed averaged result tables with confidence intervals
* [ ] Real-data benchmarks
* [ ] GPU acceleration for the meta loop
* [ ] Additional sparsity regularizers (sparse group lasso)

## 🙏 Acknowledgements

* [MW-Net: Meta-Weight-Net](https://github.com/xjtushujun/meta-weight-net) — the bi-level optimization engineering this project borrows from.
* [TSpAM: Tilted Sparse Additive Models (ICML 2023)](https://dl.acm.org/doi/abs/10.5555/3618408.3619888) — the simulation data generation protocols.
