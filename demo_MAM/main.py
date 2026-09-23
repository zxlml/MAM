import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import sys
import argparse
import warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_generation import (
    generate_regression,
    generate_corrupted_classification,
    generate_imbalanced_classification,
    generate_multi_classification,
)
from models.optimization import MAMConfig, Meta_Additive_models, train_erm_baseline

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Meta Additive Model (MAM)")
    parser.add_argument("--task", type=str, default="regression",
                        choices=["regression", "classification"])
    # regression: noise_type in None/Gaussian/mean/modal/studentT/chiSquare/mixGauss
    #             ('mean','modal','studentT' correspond to eps^A, eps^B, eps^C in the paper)
    # classification: scenario in corrupted / imbalance / multi
    parser.add_argument("--noise_type", type=str, default="mean")
    parser.add_argument("--scenario", type=str, default="imbalance",
                        choices=["corrupted", "imbalance", "multi"])
    parser.add_argument("--number", type=int, default=2000, help="samples per split")
    parser.add_argument("--dimension", type=int, default=100, help="number of covariates p")
    parser.add_argument("--ratio", type=float, default=0.15,
                        help="imbalance ratio / corruption percentage / multi-objective ratio")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lowerlr", type=float, default=5e-2)
    parser.add_argument("--upperlr", type=float, default=1e-3)
    parser.add_argument("--penaltycoef", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--weight-norm", type=str, default="none", choices=["none", "sum"])
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--print_frequency", type=int, default=50)
    parser.add_argument("--eval_frequency", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--baseline", action="store_true",
                        help="also run the unweighted ERM baseline for comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    spline_dim = 3 if args.task == "regression" else 5
    total_dimension = args.dimension * spline_dim

    if args.task == "regression":
        train_loader, validation_loader, testX, testY = generate_regression(
            number=args.number, dimension=args.dimension,
            noise_type=args.noise_type, seed=args.seed)
        scenario = args.noise_type
    else:
        if args.scenario == "corrupted":
            gen = generate_corrupted_classification
            kw = dict(percentage=args.ratio)
        elif args.scenario == "multi":
            gen = generate_multi_classification
            kw = dict(ratio=args.ratio)
        else:
            gen = generate_imbalanced_classification
            kw = dict(ratio=args.ratio)
        train_loader, validation_loader, testX, testY = gen(
            number=args.number, dimension=args.dimension, seed=args.seed, **kw)
        scenario = args.scenario

    cfg = MAMConfig(
        task=args.task,
        total_dimension=total_dimension,
        spline_dim=spline_dim,
        epochs=args.epochs,
        lowerlr=args.lowerlr,
        upperlr=args.upperlr,
        penaltycoef=args.penaltycoef,
        hidden=args.hidden,
        warmup_epochs=args.warmup,
        weight_norm=args.weight_norm,
        print_frequency=args.print_frequency,
        eval_frequency=args.eval_frequency,
        seed=args.seed,
        log_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "logs", "time=%s.log" % __import__("datetime").datetime.now()
                              .strftime("%Y-%m-%d-%H-%M-%S")),
    )

    print(f"=== MAM simulation | task={args.task} scenario={scenario} "
          f"n={args.number} p={args.dimension} (features={total_dimension}) seed={args.seed} ===")
    result = Meta_Additive_models(train_loader, validation_loader, testX, testY,
                                  total_dimension, task=args.task, config=cfg)

    metric_name = "Test MSE" if args.task == "regression" else "Test Acc"
    print(f"[MAM]      best {metric_name} = {result['best_metric']:.4f} "
          f"@ epoch {result['best_epoch']}, "
          f"selected variables {result['selected_variables'][0]}/{result['selected_variables'][1]}")

    if args.baseline:
        base = train_erm_baseline(train_loader, testX, testY, total_dimension,
                                  task=args.task, config=cfg)
        print(f"[Baseline] best {metric_name} = {base:.4f}  (unweighted ERM)")


if __name__ == "__main__":
    main()
