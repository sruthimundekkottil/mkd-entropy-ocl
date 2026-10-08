# Save this as run_local.py in the repo root

import os, sys, random, warnings
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.data import get_loaders
from src.utils import name_match
from config.parser import Parser

# =====================================================================
# SETTINGS
# =====================================================================
N_RUNS   = 3
MEM_SIZE = 500
DATASET  = 'cifar10'
N_TASKS  = 5
EPOCHS   = 1

METHODS  = [
    'VR_OCL_Adaptive',
]

VR_MU        = 0.0005
VR_BETA      = 0.99
VR_MU_CAP    = 0.05
VR_MU_MAX    = 0.02
VR_TAU       = 0.5
VR_EMA_BETA  = 0.9
VR_MARGIN    = 0.05
EWC_LAMBDA   = 1.0

# Local results folder — saves next to the repo
RESULTS_ROOT = './results_adaptive_v2'# =====================================================================


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
    parser        = Parser()
    args          = parser.parse(base_args)
    args.seed     = seed

    args.vr_mu       = VR_MU
    args.vr_beta     = VR_BETA
    args.vr_mu_cap   = VR_MU_CAP
    args.vr_mu_max   = VR_MU_MAX
    args.vr_tau      = VR_TAU
    args.vr_ema_beta = VR_EMA_BETA
    args.vr_margin   = VR_MARGIN
    args.ewc_lambda  = EWC_LAMBDA
    args.ewc_online  = True

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
        learner.train(
            dataloader=loaders[f'train{task_id}'],
            task_name=f'train{task_id}',
            task_id=task_id,
            dataloaders=loaders,
        )
        learner.before_eval()
        avg_acc, avg_fgt = learner.evaluate(loaders, task_id)

        if hasattr(learner, 'after_eval'):
            learner.after_eval()

        accs.append(avg_acc)
        fgts.append(avg_fgt)
        print(
            f"  Task {task_id+1}/{N_TASKS}  "
            f"acc={avg_acc:.4f}  fgt={avg_fgt:.4f}"
        )

    learner.save_results()

    # Save gap history for adaptive method
    if hasattr(learner, 'gap_history') and len(learner.gap_history) > 0:
        gap_dir = os.path.join(
            RESULTS_ROOT, f'gap_{learner_name}_seed{seed}'
        )
        os.makedirs(gap_dir, exist_ok=True)
        pd.DataFrame(learner.gap_history).to_csv(
            os.path.join(gap_dir, 'gap_history.csv'), index=False
        )
        print(f"  Gap history saved — {len(learner.gap_history)} steps")

    return float(np.nanmean(accs)), float(np.nanmean(fgts))


def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    print(f"Device       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Results root : {RESULTS_ROOT}")
    print(f"Methods      : {METHODS}")
    print(f"Mem size     : {MEM_SIZE}  Seeds: {N_RUNS}  Epochs: {EPOCHS}\n")

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
        print(f"  Acc : {np.mean(method_accs):.4f} +/- {np.std(method_accs):.4f}")
        print(f"  Fgt : {np.mean(method_fgts):.4f} +/- {np.std(method_fgts):.4f}")
        print(f"{'='*55}\n")

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS_ROOT, 'all_results.csv'), index=False)

    summary = df.groupby('method').agg(
        acc_mean=('acc', 'mean'),
        acc_std=('acc',  'std'),
        fgt_mean=('fgt', 'mean'),
        fgt_std=('fgt',  'std'),
    ).round(4)
    summary.to_csv(os.path.join(RESULTS_ROOT, 'summary.csv'))

    print(f"\n{'='*55}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*55}")
    print(summary.to_string())
    print(f"\nResults saved to: {RESULTS_ROOT}")


if __name__ == '__main__':
    main()