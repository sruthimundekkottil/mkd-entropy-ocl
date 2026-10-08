"""
run_vrocl.py
============
Single runner for the VR-OCL experiments on Split CIFAR-10.
Replaces the hand-edited run_local.py and src/run_vrocl.py.

Every setting is a command-line flag, and the flags actually used are
saved to <out>/run_config.json together with the date, the git commit
and the device, so each result folder records how it was produced.

Defaults match the runs behind the report (mem 500, lr 0.0005, nf 20,
stream batch 10, memory batch 64, 1 epoch per task, seeds 0 1 2).

Examples (from the repo root):
    # ER / EWC baselines and fixed + growing-schedule VR-OCL at mu=0.0005
    python scripts/run_vrocl.py --methods ER EWC VR_OCL VR_OCL_Decay \
        --vr-mu 0.0005 --out results/local/my_run

    # Adaptive variant
    python scripts/run_vrocl.py --methods VR_OCL_Adaptive \
        --vr-mu-max 0.02 --vr-tau 0.5 --out results/local/my_adaptive_run

    # Smoke test: 5 batches per task, one seed
    python scripts/run_vrocl.py --methods ER VR_OCL --seeds 0 \
        --max-batches 5 --out results/smoke --overwrite

Per-run files (acc.csv, avg.csv, forgetting.csv, avg_forgetting.csv,
params_used.json) are written by the upstream ERLearner.save_results().
This script adds:
    all_results.csv   one row per (method, seed) with four metrics
    summary.csv       mean and std (ddof=1) per method
    gap_<method>_seed<k>/gap_history.csv   for the adaptive variant
Metrics, in [0, 1]:
    aia        average incremental accuracy (mean over task checkpoints)
    avg_fgt    average incremental forgetting
    final_acc  average accuracy over all tasks after the last task
    final_fgt  average forgetting after the last task
"""

import argparse
import datetime as dt
import json
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

VR_METHODS = ('VR_OCL', 'VR_OCL_Decay', 'VR_OCL_Adaptive')


def parse_cli(argv=None):
    p = argparse.ArgumentParser(description='Run VR-OCL / ER / EWC experiments on Split CIFAR-10.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--methods', nargs='+',
                   default=['ER', 'EWC', 'VR_OCL', 'VR_OCL_Decay', 'VR_OCL_Adaptive'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--out', required=True, help='Output folder, e.g. results/local/<name>')
    p.add_argument('--overwrite', action='store_true',
                   help='Allow writing into a non-empty output folder')
    p.add_argument('--max-batches', type=int, default=None,
                   help='Stop each task after this many stream batches (smoke tests only)')

    g = p.add_argument_group('benchmark')
    g.add_argument('--dataset', default='cifar10')
    g.add_argument('--n-tasks', type=int, default=5)
    g.add_argument('--mem-size', type=int, default=500)
    g.add_argument('--batch-size', type=int, default=10)
    g.add_argument('--mem-batch-size', type=int, default=64)
    g.add_argument('--mem-iters', type=int, default=1)
    g.add_argument('--epochs', type=int, default=1)
    g.add_argument('--nf', type=int, default=20, help='ResNet-18 width (20 = reduced)')
    g.add_argument('--lr', type=float, default=0.0005,
                   help='Adam learning rate (0.0005 is the value used for every saved run)')
    g.add_argument('--data-root-dir', default='./data/')

    g = p.add_argument_group('VR-OCL')
    g.add_argument('--vr-mu', type=float, default=0.0005)
    g.add_argument('--vr-beta', type=float, default=0.99)
    g.add_argument('--vr-mu-cap', type=float, default=0.05)
    g.add_argument('--vr-mu-max', type=float, default=0.02)
    g.add_argument('--vr-tau', type=float, default=0.5)
    g.add_argument('--vr-ema-beta', type=float, default=0.9)
    g.add_argument('--vr-margin', type=float, default=0.05)

    g = p.add_argument_group('EWC')
    g.add_argument('--ewc-lambda', type=float, default=1.0)
    g.add_argument('--ewc-offline', action='store_true',
                   help='Recompute the Fisher at each task instead of accumulating it')
    return p.parse_args(argv)


class LimitedLoader:
    """Wraps a DataLoader so a task stops after `n` batches (smoke tests)."""

    def __init__(self, loader, n):
        self.loader, self.n = loader, n

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.n:
                break
            yield batch

    def __len__(self):
        return min(self.n, len(self.loader))


def make_args(cli, method, seed):
    from config.parser import Parser

    base_args = [
        '--learner', method,
        '--dataset', cli.dataset,
        '--n-tasks', str(cli.n_tasks),
        '--mem-size', str(cli.mem_size),
        '--batch-size', str(cli.batch_size),
        '--mem-batch-size', str(cli.mem_batch_size),
        '--epochs', str(cli.epochs),
        '--learning-rate', str(cli.lr),
        '--seed', str(seed),
        '--training-type', 'inc',
        '--results-root', cli.out,
        '--data-root-dir', cli.data_root_dir,
        '--tag', f'{method}_seed{seed}',
        '--no-wandb',
        '--train',
        '-nf', str(cli.nf),
        '--mem-iters', str(cli.mem_iters),
    ]
    args = Parser().parse(base_args)
    args.seed = seed

    # Not in the upstream parser: the learners read them with getattr()
    args.vr_mu = cli.vr_mu
    args.vr_beta = cli.vr_beta
    args.vr_mu_cap = cli.vr_mu_cap
    args.vr_mu_max = cli.vr_mu_max
    args.vr_tau = cli.vr_tau
    args.vr_ema_beta = cli.vr_ema_beta
    args.vr_margin = cli.vr_margin
    args.ewc_lambda = cli.ewc_lambda
    args.ewc_online = not cli.ewc_offline
    return args


def run_one(cli, method, seed):
    import numpy as np
    import pandas as pd
    import torch
    from src.utils.data import get_loaders
    from src.utils import name_match

    print(f"\n{'-' * 55}\n  {method}  |  seed={seed}\n{'-' * 55}")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    args = make_args(cli, method, seed)
    learner = name_match.learners[method](args)
    loaders = get_loaders(args)

    accs, fgts = [], []
    for task_id in range(cli.n_tasks):
        loader = loaders[f'train{task_id}']
        if cli.max_batches is not None:
            loader = LimitedLoader(loader, cli.max_batches)
        for _ in range(cli.epochs):
            learner.train(dataloader=loader, task_name=f'train{task_id}',
                          task_id=task_id, dataloaders=loaders)
        learner.before_eval()
        avg_acc, avg_fgt = learner.evaluate(loaders, task_id)
        learner.after_eval()  # EWC updates its Fisher here; a no-op for the others
        accs.append(avg_acc)
        fgts.append(avg_fgt)
        print(f'  Task {task_id + 1}/{cli.n_tasks}  acc={avg_acc:.4f}  fgt={avg_fgt:.4f}')

    learner.save_results()

    if getattr(learner, 'gap_history', None):
        gap_dir = Path(cli.out) / f'gap_{method}_seed{seed}'
        gap_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(learner.gap_history).to_csv(gap_dir / 'gap_history.csv', index=False)

    return {
        'method': method,
        'seed': seed,
        'aia': float(np.nanmean(accs)),
        'avg_fgt': float(np.nanmean(fgts)),
        'final_acc': float(accs[-1]),
        'final_fgt': float(fgts[-1]),
    }


def git_commit():
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + (' (uncommitted changes)' if dirty else '')
    except OSError:
        return 'unknown'


def main(argv=None):
    cli = parse_cli(argv)
    warnings.filterwarnings('ignore')

    import pandas as pd
    import torch
    from src.utils import name_match

    unknown = [m for m in cli.methods if m not in name_match.learners]
    if unknown:
        sys.exit(f'Unknown method(s): {unknown}')

    out = Path(cli.out)
    if out.exists() and any(out.iterdir()) and not cli.overwrite:
        sys.exit(f'Refusing to write into non-empty {out} (use --overwrite)')
    out.mkdir(parents=True, exist_ok=True)

    config = {
        'cli': vars(cli),
        'started': dt.datetime.now().isoformat(timespec='seconds'),
        'git_commit': git_commit(),
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'torch': torch.__version__,
        'python': sys.version.split()[0],
    }
    (out / 'run_config.json').write_text(json.dumps(config, indent=2))
    print(f"Output: {out}\nDevice: {config['device']}\nMethods: {cli.methods}  Seeds: {cli.seeds}")

    rows = [run_one(cli, m, s) for m in cli.methods for s in cli.seeds]

    df = pd.DataFrame(rows)
    df.to_csv(out / 'all_results.csv', index=False)
    metrics = ['aia', 'avg_fgt', 'final_acc', 'final_fgt']
    summary = df.groupby('method', sort=False)[metrics].agg(['mean', 'std']).round(4)
    summary.columns = [f'{m}_{s}' for m, s in summary.columns]
    summary.to_csv(out / 'summary.csv')

    config['finished'] = dt.datetime.now().isoformat(timespec='seconds')
    (out / 'run_config.json').write_text(json.dumps(config, indent=2))
    print(f'\nSUMMARY\n{summary.to_string()}\n\nSaved to {out}')


if __name__ == '__main__':
    main()
