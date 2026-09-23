# Meta Additive Models (MAM)

[English](README.md) | **简体中文**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-45%20passed-brightgreen)](demo_MAM/tests)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Meta Additive Models（元可加模型）** 的忠实复现与工程重构——一个双层元学习框架：借鉴 [MW-Net](https://github.com/xjtushujun/meta-weight-net) 的思想，通过加权网络为样本学习权重，用于在 B 样条基特征上拟合带 group-lasso 惩罚的**稀疏可加模型**。

> ⭐ 如果本项目对您有帮助，欢迎点个 Star！

## ✨ 亮点

* 🚀 **双层元优化** —— 单步展开内层更新（`create_graph=True`）、MW-Net 式 ghost 模型重建、持久验证集迭代器、warm-up 预热轮。
* 🎯 **复合下层求解器** —— 下层问题是"平滑损失 + 非光滑 group-lasso"的复合优化，因此 **proximal 算子被保留在 autograd 计算图中**：超梯度正确地计入了稀疏算子的贡献（这是与 MW-Net 的本质差异——MW-Net 的下层是平滑深度网络，无此需求）。
* 🧮 **数值稳定的样条设计** —— 样条结点仅从训练集拟合并复用于验证/测试；每个 Bernstein 块基于训练集做 SVD 白化（消除跨块精确共线，条件数从 ~1e15 降至 ~1），并重缩放到单位方差使普通 SGD 可收敛。
* 🛡️ **稳健训练** —— 梯度裁剪与非有限损失防护，应对论文噪声模型（ε^A/ε^B/ε^C）中的极端离群点。
* ✅ **测试完备** —— 45 项单元/功能测试，包含穿过可微 prox 的超梯度有限差分校验。

## 📢 动态

* **[2026-09]** 🚀 完成大规模重构：可微 group-lasso prox、参照 MW-Net 工程重写双层循环、样条基改用训练集结点 + SVD 白化。
* **[2026-09]** ✅ 新增完整测试套件（单元 + 功能，共 45 项）。
* **[2026-09]** 📊 论文规模仿真（n=2000, p=100）复现论文定性结论——见[实验结果](#-实验结果)。

## 📑 目录

* [✨ 亮点](#-亮点)
* [📢 动态](#-动态)
* [🔧 安装](#-安装)
* [⚡ 快速开始](#-快速开始)
* [🏗️ 项目结构](#️-项目结构)
* [🧠 算法设计](#-算法设计)
* [📁 实验结果](#-实验结果)
* [🧪 测试](#-测试)
* [☑️ 待办](#️-待办)
* [🙏 致谢](#-致谢)

## 🔧 安装

```bash
# 克隆仓库
git clone https://github.com/zxlml/MAM.git
cd MAM/demo_MAM

# 创建虚拟环境（Python 3.10+）
conda create -n mam python=3.10 -y
conda activate mam

# 安装依赖
pip install torch numpy scipy scikit-learn matplotlib pytest
```

> 💡 Windows + Anaconda 环境下请设置 `KMP_DUPLICATE_LIB_OK=TRUE`，避免 OpenMP 运行时冲突（`libiomp5md.dll`）。

## ⚡ 快速开始

### 1. 命令行（推荐）

回归任务使用论文的噪声模型（`mean`/`modal`/`studentT` 分别对应 ε^A/ε^B/ε^C）：

```bash
python main.py --task regression --noise_type mean \
    --number 2000 --dimension 100 --epochs 1000 --seed 1 --baseline
```

分类任务场景（`imbalance` 类别不平衡 / `corrupted` 标签损坏 / `multi` 多目标）：

```bash
python main.py --task classification --scenario imbalance \
    --number 2000 --dimension 100 --epochs 500 --seed 1 --baseline
```

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--task` | `regression` | `regression` 回归 / `classification` 分类 |
| `--noise_type` | `mean` | 回归噪声：`None/Gaussian/mean/modal/studentT/chiSquare/mixGauss` |
| `--scenario` | `imbalance` | 分类场景：`imbalance/corrupted/multi` |
| `--number` | `2000` | 每个数据划分的样本数 |
| `--dimension` | `100` | 协变量个数 p |
| `--epochs` | `1000` | 训练轮数 |
| `--lowerlr` | `5e-2` | 下层（可加模型）SGD 学习率 |
| `--upperlr` | `1e-3` | 加权网络 Adam 学习率 |
| `--penaltycoef` | `1e-3` | group-lasso 系数 λ |
| `--warmup` | `1` | 预热轮数（均匀权重） |
| `--baseline` | 关 | 同时运行无加权 ERM 基线对比 |

### 2. Python API

```python
from data.data_generation import generate_regression, generate_imbalanced_classification
from models.optimization import Meta_Additive_models, MAMConfig

# 回归（论文设置：n=2000, p=100, 噪声 = eps^A）
train_loader, val_loader, testX, testY = generate_regression(
    number=2000, dimension=100, noise_type='mean', seed=1)

cfg = MAMConfig(task='regression', total_dimension=100*3, spline_dim=3,
                epochs=1000, lowerlr=5e-2, penaltycoef=1e-3, seed=1)
result = Meta_Additive_models(train_loader, val_loader, testX, testY,
                              total_dimension=100*3, task='regression', config=cfg)
print(result['best_metric'], result['best_epoch'], result['selected_variables'])

# 分类（imbalance / corrupted / multi）
train_loader, val_loader, testX, testY = generate_imbalanced_classification(
    number=2000, dimension=100, ratio=0.15, seed=1)
cfg = MAMConfig(task='classification', total_dimension=100*5, spline_dim=5,
                epochs=500, lowerlr=5e-2, penaltycoef=1e-3, seed=1)
result = Meta_Additive_models(train_loader, val_loader, testX, testY,
                              total_dimension=100*5, task='classification', config=cfg)
```

## 🏗️ 项目结构

```
MAM/
├── demo_MAM/
│   ├── main.py                  # 命令行入口：数据生成 + 双层训练 + 基线对比
│   ├── data/
│   │   └── data_generation.py   # TSpAM 式仿真数据；B 样条基 + SVD 白化
│   ├── models/
│   │   └── optimization.py      # 双层求解器、可微 prox、ERM 基线
│   ├── tests/                   # 45 项单元/功能测试（pytest）
│   ├── baselines/               # 原始基线实现
│   └── logs/                    # 训练/仿真日志
└── README.md
```

## 🧠 算法设计

由于下层是**复合凸问题**，本项目的双层结构与 MW-Net 有本质不同：

```text
upper:  min_theta  L_val( beta_hat(theta) )
lower:  beta_hat(theta) = prox_{lam*eta*Omega}( beta - eta * (1/n) sum_i
                     sigma(f_theta(cost_i(beta))) * grad_beta cost_i(beta) )
```

* **可微 prox 进入超梯度。** group-lasso proximal 算子 `w_g ← w_g·max(1 − λη/‖w_g‖₂, 0)` 是解映射的一部分，因此在单步展开的内层更新中被**保留在 autograd 计算图内**；生产模型则在 `.data` 上原位应用同一 prox（精确复合更新）。原实现通过 `.data` 赋值断开了 prox 的梯度，导致超梯度完全丢失稀疏算子的贡献。
* **借鉴自 MW-Net 的工程设计。** 单步展开内层更新（`create_graph=True`）、加权网络在分离代价上求值（`vnet(cost.data)`）、每个训练批次配一个来自持久迭代器的验证批次（StopIteration 回收）、加权网络使用 Adam、均匀权重 warm-up 轮。
* **样条基的数值修复。** 结点仅取自**训练集**并复用于验证/测试（原代码在每个划分上重新拟合结点，导致三个划分被不同的特征映射处理）；每个 Bernstein 块基于训练集做 SVD 白化（该基张成常数函数 → 存在跨块精确共线，条件数 ~1e15），并重缩放到单位方差使普通 SGD 求解器可收敛。
* **group-lasso 分组修复。** 原实现的 proximal 步骤对分类权重矩阵的 reshape 是错误的（类别与协变量混排）；现在按每个输出连续的 `spline_dim` 列构成一组。

## 📁 实验结果

论文规模仿真（`n=2000`，`p=100`，`seed=1`；回归 1000 轮、分类 500 轮；MAM 与使用同一求解器的无加权 ERM 基线对比）：

| 场景 | 任务 | 指标 | MAM | ERM 基线 | 提升 |
| --- | --- | --- | --- | --- | --- |
| ε^A（`mean` 噪声） | 回归 | 测试 MSE | **0.0151** | 0.0443 | −66% |
| ε^B（`modal` 噪声） | 回归 | 测试 MSE | **0.0048** | 0.2733 | −98% |
| ε^C（`studentT` 噪声） | 回归 | 测试 MSE | **0.0068** | 0.0598 | −89% |
| 类别不平衡（ratio 0.15） | 分类 | 测试 Acc | **0.9455** | 0.8825 | +6.3 pt |
| 标签损坏（15%） | 分类 | 测试 Acc | **0.9355** | 0.8345 | +10.1 pt |
| 不平衡 + 损坏（multi） | 分类 | 测试 Acc | **0.7225** | 0.6515 | +7.1 pt |

六个场景全部证实论文的定性结论：元学习得到的样本权重使稀疏可加模型对重尾/偏移噪声、标签噪声和类别不平衡的鲁棒性**显著优于**无加权拟合。完整日志见 `demo_MAM/logs/sim_*.txt`。

## 🧪 测试

```bash
cd demo_MAM
python -m pytest tests -q        # 45 项测试：样条基、数据生成、prox 算子、
                                 # 超梯度有限差分校验、双层步骤、功能测试
```

## ☑️ 待办

* [ ] 多种子平均结果表（含置信区间）
* [ ] 真实数据集基准
* [ ] 元循环的 GPU 加速
* [ ] 更多稀疏正则（sparse group lasso）

## 🙏 致谢

* [MW-Net: Meta-Weight-Net](https://github.com/xjtushujun/meta-weight-net) —— 本项目双层优化工程设计的借鉴来源。
* [TSpAM: Tilted Sparse Additive Models（ICML 2023）](https://dl.acm.org/doi/abs/10.5555/3618408.3619888) —— 仿真数据生成协议的出处。
