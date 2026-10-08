"""
generate_figures.py
===================
Draws the report figures from the tables written by
scripts/make_report_tables.py. No numbers are hardcoded here.

Figures use only the headline (local) runs in results/tables/; the
archived Kaggle runs are not plotted (see results/archive_kaggle/README.md).

Run from the repo root:
    python scripts/make_report_tables.py
    python generate_figures.py

Saves to: ./report_figures/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
TABLES = REPO_ROOT / 'results' / 'tables'
RESULTS = REPO_ROOT / 'results'
OUT = REPO_ROOT / 'report_figures'
ENVS = ['local']  # headline runs only; Kaggle runs are archived

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
    'axes.grid':         True,
    'grid.linestyle':    ':',
    'grid.alpha':        0.4,
})

C = {
    'ER':              '#4C72B0',
    'EWC':             '#55A868',
    'VR_OCL':          '#DD8452',
    'VR_OCL_Decay':    '#C44E52',
    'VR_OCL_Adaptive': '#9467BD',
}
SHORT = {
    'ER': 'ER', 'EWC': 'EWC', 'VR_OCL': 'VR fixed',
    'VR_OCL_Decay': 'VR growing', 'VR_OCL_Adaptive': 'VR adaptive',
}


def load():
    f = TABLES / 'summary.csv'
    if not f.exists():
        sys.exit('results/tables/summary.csv not found: run scripts/make_report_tables.py first')
    return pd.read_csv(f)


def label(row):
    cfg = '' if row.config == '-' else '\n' + row.config.replace(', ', '\n')
    return SHORT[row.method] + cfg


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', dpi=150)
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {name}')


def paired_bars(summary, first, second, ylabel, title, name):
    """Two bars per configuration: the incremental metric and the final metric."""
    fig, axes = plt.subplots(1, len(ENVS), figsize=(9 * len(ENVS), 5.5), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, env in zip(axes, ENVS):
        s = summary[summary.environment == env].reset_index(drop=True)
        x = np.arange(len(s))
        w = 0.38
        colors = [C[m] for m in s.method]
        ax.bar(x - w / 2, s[f'{first}_mean'], w, yerr=s[f'{first}_std'], capsize=3,
               color=colors, alpha=0.9, label='average incremental')
        ax.bar(x + w / 2, s[f'{second}_mean'], w, yerr=s[f'{second}_std'], capsize=3,
               color=colors, alpha=0.45, hatch='//', label='final (after last task)')
        ax.set_xticks(x)
        ax.set_xticklabels([label(r) for r in s.itertuples()], fontsize=8)
        ax.set_title(f'{env} runs')
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save(fig, name)


def fig_scatter(summary):
    fig, axes = plt.subplots(1, len(ENVS), figsize=(9 * len(ENVS), 6), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, env in zip(axes, ENVS):
        s = summary[summary.environment == env]
        for r in s.itertuples():
            marker = 'D' if r.method in ('ER', 'EWC') else 'o'
            ax.errorbar(r.avg_fgt_mean, r.aia_mean, xerr=r.avg_fgt_std, yerr=r.aia_std,
                        fmt=marker, color=C[r.method], ms=9, capsize=3, alpha=0.85,
                        mec='black' if r.method == 'ER' else 'none')
            ax.annotate(label(r).replace('\n', ' '), (r.avg_fgt_mean, r.aia_mean),
                        textcoords='offset points', fontsize=8,
                        xytext=(7, -14) if r.method in ('ER', 'EWC') else (7, 5))
        ax.set_title(f'{env} runs')
        ax.set_xlabel('Average incremental forgetting (%)  (lower is better)')
    axes[0].set_ylabel('Average incremental accuracy (%)  (higher is better)')
    fig.suptitle('Accuracy vs forgetting, mean ± std over 3 seeds', y=1.01)
    fig.tight_layout()
    save(fig, 'fig3_scatter')


def fig_ablation(summary):
    fig, axes = plt.subplots(len(ENVS), 2, figsize=(13, 4.5 * len(ENVS)), squeeze=False)
    for row, env in enumerate(ENVS):
        s = summary[summary.environment == env]
        base = s.set_index('method')
        vr = s[s.method.isin(['VR_OCL', 'VR_OCL_Decay'])].copy()
        vr['mu'] = vr.config.str.replace('mu=', '').astype(float)
        for col, (metric, ylabel) in enumerate([('aia', 'Average incremental accuracy (%)'),
                                                ('avg_fgt', 'Average incremental forgetting (%)')]):
            ax = axes[row, col]
            for method, marker in [('VR_OCL', 'o'), ('VR_OCL_Decay', 's')]:
                v = vr[vr.method == method].sort_values('mu')
                ax.errorbar(v.mu, v[f'{metric}_mean'], yerr=v[f'{metric}_std'], fmt=marker + '-',
                            color=C[method], capsize=3, lw=2, ms=7, label=SHORT[method])
            for method in ['ER', 'EWC']:
                ax.axhline(base.loc[method, f'{metric}_mean'], color=C[method], ls='--', lw=1.5,
                           label=f"{method} ({base.loc[method, f'{metric}_mean']:.2f})")
            ax.set_xscale('log')
            ax.set_xticks(sorted(vr.mu.unique()))
            ax.set_xticklabels([f'{m:g}' for m in sorted(vr.mu.unique())])
            ax.minorticks_off()
            ax.set_xlabel('mu')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{env} runs')
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Effect of mu (only mu values with saved result files)', y=1.01)
    fig.tight_layout()
    save(fig, 'fig4_ablation')


def fig_ewc_heatmap(summary):
    row = summary[(summary.environment == 'local') & (summary.method == 'EWC')].iloc[0]
    mats = [pd.read_csv(f).to_numpy(dtype=float)
            for f in sorted((RESULTS / row.folder).glob('EWC,*/run*/acc.csv'))]
    m = np.nanmean(mats, axis=0)
    n = m.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.ma.masked_invalid(m), cmap='viridis', vmin=0, vmax=1, aspect='auto')
    fig.colorbar(im, ax=ax).set_label('Accuracy')
    ax.set_xticks(range(n), [f'T{i + 1}' for i in range(n)])
    ax.set_yticks(range(n), [f'After T{i + 1}' for i in range(n)])
    ax.set_xlabel('Evaluated task')
    ax.set_ylabel('Training progress')
    ax.set_title(f'EWC per-task accuracy, local runs (mean of {len(mats)} seeds)')
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f'{m[i, j]:.2f}', ha='center', va='center', fontsize=10,
                        color='white' if m[i, j] < 0.65 else 'black')
    fig.tight_layout()
    save(fig, 'fig5_ewc_heatmap')


def main():
    summary = load()
    print('Generating figures...\n')
    paired_bars(summary, 'aia', 'final_acc', 'Accuracy (%)',
                'Accuracy on Split CIFAR-10 (mem=500), mean ± std over 3 seeds', 'fig1_accuracy')
    paired_bars(summary, 'avg_fgt', 'final_fgt', 'Forgetting (%)',
                'Forgetting on Split CIFAR-10 (mem=500), mean ± std over 3 seeds', 'fig2_forgetting')
    fig_scatter(summary)
    fig_ablation(summary)
    fig_ewc_heatmap(summary)
    print(f'\nAll done. Figures in {OUT}')


if __name__ == '__main__':
    main()
