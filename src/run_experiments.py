"""
run_experiments.py
==================
Runs all four methods on Split CIFAR-10 and saves results as CSV.
Designed to run in Kaggle or Colab without argparse issues.

Methods compared
----------------
1. ER            — plain Experience Replay (random retrieval)
2. ER_Entropy    — ER with entropy-guided retrieval  (Contribution A)
3. ER_EMA        — ER + MKD fixed alpha              (Michel et al. 2024 baseline)
4. ER_EMA_Entropy— ER + MKD + entropy retrieval      (Main contribution)

Usage (in a notebook cell):
    exec(open('run_experiments.py').read())

Or as a script:
    python run_experiments.py
"""

import os, sys, random, warnings
import numpy as np
import pandas as pd
import torch
import wandb
warnings.filterwarnings("ignore")

# ── Make sure the repo root is on the path ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.data import get_loaders
from src.utils import name_match
from config.parser import Parser

# ═════════════════════════════════════════════════════════════════════
# SETTINGS — edit these
# ═════════════════════════════════════════════════════════════════════
N_RUNS   = 3          # seeds (paper uses 5; 3 is fine for a mini project)
MEM_SIZE = 200        # replay buffer size (200 = small, 500 = medium)
DATASET  = 'cifar10'  # 'cifar10' or 'cifar100'
N_TASKS  = 5
EPOCHS   = 1          # OCL trains 1 epoch per task by design

# Methods to run — remove any you want to skip
METHODS  = [
    'ER',
    'ER_Entropy',
    'ER_EMA',
    'ER_EMA_Entropy',
]

RESULTS_ROOT = './results'
# ═════════════════════════════════════════════════════════════════════


def make_args(learner_name, seed, mem_size=MEM_SIZE):
    """Build an args namespace that mimics what Parser produces."""
    base_args = [
        '--learner',        learner_name,
        '--dataset',        DATASET,
        '--n-tasks',        str(N_TASKS),
        '--mem-size',       str(mem_size),
        '--batch-size',     '10',
        '--mem-batch-size', '64',
        '--epochs',         str(EPOCHS),
        '--seed',           str(seed),
        '--training-type',  'inc',
        '--results-root',   RESULTS_ROOT,
        '--tag',            f'{learner_name}_seed{seed}',
        '--no-wandb',
        '--train',
        '-nf',              '20',   # reduced ResNet18 — faster
    ]

    # MKD-specific arguments (ignored by non-EMA learners)
    base_args += [
        '--alpha-min',  '0.01',
        '--alpha-max',  '0.01',
        '--n-teacher',  '1',
        '--kd-lambda',  '1.0',
        '--kd-temperature', '4',
        '--ema-correction-step', '10',
        '--mem-iters',  '1',
    ]

    parser = Parser()
    args   = parser.parse(base_args)
    args.seed = seed

    # Inject our custom flags (not in original parser)
    args.entropy_retrieval = ('Entropy' in learner_name)
    args.stochastic_alpha  = False   # set True to test Contribution B

    return args


def run_one(learner_name, seed):
    """Train one method for one seed. Returns (avg_acc, avg_fgt)."""
    print(f"\n{'─'*55}")
    print(f"  {learner_name}  |  seed={seed}")
    print(f"{'─'*55}")

    # Reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    args      = make_args(learner_name, seed)
    learner   = name_match.learners[learner_name](args)
    loaders   = get_loaders(args)

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
        accs.append(avg_acc)
        fgts.append(avg_fgt)
        print(f"  Task {task_id+1}/{N_TASKS}  acc={avg_acc:.4f}  fgt={avg_fgt:.4f}")

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

    # ── Save all results to CSV ───────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_ROOT, 'all_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nAll results saved to: {csv_path}")

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  SUMMARY  ({N_RUNS} seeds, Split {DATASET.upper()})")
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
