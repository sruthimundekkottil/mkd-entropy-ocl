"""
run_vrocl.py
============
Runs the VR-OCL comparison experiments on Split CIFAR-10.

Methods compared
----------------
1. ER            — vanilla Experience Replay (baseline)
2. EWC           — Elastic Weight Consolidation (Kirkpatrick et al. 2017)
3. VR_OCL        — Variance Regularizer, fixed mu  (ours)
4. VR_OCL_Decay  — Variance Regularizer, decaying mu (ours, from paper schedule)

Usage (Kaggle / Colab notebook cell):
    exec(open('run_vrocl.py').read())

Or as script:
    python run_vrocl.py
"""

import os, sys, random, warnings
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.data import get_loaders
from src.utils import name_match
from config.parser import Parser

# ═════════════════════════════════════════════════════════════════════
# SETTINGS
# ═════════════════════════════════════════════════════════════════════
N_RUNS   = 3
MEM_SIZE = 200     # larger than before — gives methods a fair chance
DATASET  = 'cifar10'
N_TASKS  = 5
EPOCHS   = 1       # OCL standard: 1 pass per task

METHODS  = [
    'ER',
    'EWC',
    'VR_OCL',
    'VR_OCL_Decay',
]

# VR-OCL hyperparameters — tune these if needed
VR_MU        = 0.1    # base regularization strength
VR_BETA      = 0.99   # decay base (β in the paper)
VR_MU_CAP   = 10.0   # maximum mu for decay variant

# EWC hyperparameters
EWC_LAMBDA   = 1.0

RESULTS_ROOT = './results_vrocl'
# ═════════════════════════════════════════════════════════════════════


def make_args(learner_name, seed):
    base_args = [
        '--learner',        learner_name,
        '--dataset',        DATASET,
        '--n-tasks',        str(N_TASKS),
        '--mem-size',       str(MEM_SIZE),
        '--batch-size',     '10',
        '--mem-batch-size', '64',
        '--epochs',         str(EPOCHS),
        '--seed',           str(seed),
        '--training-type',  'inc',
        '--results-root',   RESULTS_ROOT,
        '--tag',            f'{learner_name}_seed{seed}',
        '--no-wandb',
        '--train',
        '-nf',              '20',
        '--mem-iters',      '1',
    ]

    parser = Parser()
    args   = parser.parse(base_args)
    args.seed = seed

    # VR-OCL params
    args.vr_mu     = VR_MU
    args.vr_beta   = VR_BETA
    args.vr_mu_cap = VR_MU_CAP

    # EWC params
    args.ewc_lambda = EWC_LAMBDA
    args.ewc_online = True

    return args


def run_one(learner_name, seed):
    print(f"\n{'─'*55}")
    print(f"  {learner_name}  |  seed={seed}")
    print(f"{'─'*55}")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    args    = make_args(learner_name, seed)
    learner = name_match.learners[learner_name](args)
    loaders = get_loaders(args)

    accs, fgts = [], []
    for task_id in range(N_TASKS):
        for _ in range(EPOCHS):
            learner.train(
                dataloader=loaders[f'train{task_id}'],
                task_name=f'train{task_id}',
                task_id=task_id,
                dataloaders=loaders,
            )
        learner.before_eval()
        avg_acc, avg_fgt = learner.evaluate(loaders, task_id)

        # EWC needs to update Fisher after each task evaluation
        if hasattr(learner, 'after_eval'):
            learner.after_eval()

        accs.append(avg_acc)
        fgts.append(avg_fgt)
        print(
            f"  Task {task_id+1}/{N_TASKS}  "
            f"acc={avg_acc:.4f}  fgt={avg_fgt:.4f}"
        )

    learner.save_results()
    return float(np.nanmean(accs)), float(np.nanmean(fgts))


def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    all_results = []

    for method in METHODS:
        method_accs, method_fgts = [], []
        for seed in range(N_RUNS):
            acc, fgt = run_one(method, seed)
            method_accs.append(acc)
            method_fgts.append(fgt)
            all_results.append({
                'method': method,
                'seed':   seed,
                'acc':    acc,
                'fgt':    fgt,
            })

        print(f"\n{'='*55}")
        print(f"  {method}")
        print(f"  Avg Acc : {np.mean(method_accs):.4f} ± {np.std(method_accs):.4f}")
        print(f"  Avg Fgt : {np.mean(method_fgts):.4f} ± {np.std(method_fgts):.4f}")
        print(f"{'='*55}\n")

    df       = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_ROOT, 'all_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print(f"\n{'='*55}")
    print(f"  FINAL SUMMARY  (Split {DATASET.upper()}, mem={MEM_SIZE})")
    print(f"{'='*55}")
    summary = df.groupby('method').agg(
        acc_mean=('acc', 'mean'),
        acc_std=('acc', 'std'),
        fgt_mean=('fgt', 'mean'),
        fgt_std=('fgt', 'std'),
    ).round(4)
    print(summary.to_string())
    summary.to_csv(os.path.join(RESULTS_ROOT, 'summary.csv'))

    return df


if __name__ == '__main__':
    main()
