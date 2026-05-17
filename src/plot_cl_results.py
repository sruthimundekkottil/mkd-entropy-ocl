"""
plot_cl_results.py
==================
Loads results/all_results.csv and produces:
  Figure 1 — Bar chart: Average Accuracy per method
  Figure 2 — Bar chart: Forgetting per method
  Figure 3 — Scatter: Accuracy vs Forgetting trade-off
  Console  — LaTeX-ready table

Run after run_experiments.py has finished.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_CSV = './results/all_results.csv'
FIGURES_DIR = './results/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────
COLORS = {
    'ER':               '#4C72B0',
    'ER_Entropy':       '#DD8452',
    'ER_EMA':           '#55A868',
    'ER_EMA_Entropy':   '#C44E52',
}
LABELS = {
    'ER':               'ER (baseline)',
    'ER_Entropy':       'ER + Entropy Replay',
    'ER_EMA':           'ER + MKD (Michel et al.)',
    'ER_EMA_Entropy':   'ER + MKD + Entropy (ours)',
}

plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


def load():
    df = pd.read_csv(RESULTS_CSV)
    summary = df.groupby('method').agg(
        acc_mean=('acc', 'mean'),
        acc_std=('acc', 'std'),
        fgt_mean=('fgt', 'mean'),
        fgt_std=('fgt', 'std'),
    ).reset_index()
    return df, summary


def plot_bar(summary, metric_mean, metric_std, title, ylabel, fname, higher_better=True):
    methods = summary['method'].tolist()
    means   = summary[metric_mean].tolist()
    stds    = summary[metric_std].tolist()
    colors  = [COLORS.get(m, '#888888') for m in methods]
    labels  = [LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, width=0.55,
                  error_kw=dict(elinewidth=1.5))

    for bar, mean, std in zip(bars, means, stds):
        offset = std + 0.003
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + offset,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)

    y_vals = [m - s for m, s in zip(means, stds)] + \
             [m + s for m, s in zip(means, stds)]
    pad = (max(y_vals) - min(y_vals)) * 0.15
    ax.set_ylim(max(0, min(y_vals) - pad), min(1, max(y_vals) + pad * 2))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, fname)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_scatter(summary):
    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in summary.iterrows():
        m = row['method']
        ax.scatter(row['fgt_mean'], row['acc_mean'],
                   color=COLORS.get(m, '#888'),
                   s=120, zorder=3, label=LABELS.get(m, m))
        ax.errorbar(row['fgt_mean'], row['acc_mean'],
                    xerr=row['fgt_std'], yerr=row['acc_std'],
                    color=COLORS.get(m, '#888'), alpha=0.4, zorder=2)

    ax.set_xlabel('Forgetting (lower is better →)')
    ax.set_ylabel('Average Accuracy (higher is better ↑)')
    ax.set_title('Accuracy vs Forgetting Trade-off')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_scatter.pdf')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def print_latex(summary):
    print('\n' + '='*60)
    print('LaTeX Table:')
    print('='*60)
    print(r'\begin{table}[h]')
    print(r'\centering')
    print(r'\caption{Results on Split CIFAR-10 (mean $\pm$ std over 3 seeds).}')
    print(r'\begin{tabular}{lcc}')
    print(r'\toprule')
    print(r'Method & Avg. Accuracy (\%) & Forgetting (\%) \\')
    print(r'\midrule')
    for _, row in summary.iterrows():
        label = LABELS.get(row['method'], row['method'])
        print(
            f"{label} & "
            f"{row['acc_mean']*100:.1f} $\\pm$ {row['acc_std']*100:.1f} & "
            f"{row['fgt_mean']*100:.1f} $\\pm$ {row['fgt_std']*100:.1f} \\\\"
        )
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


def main():
    df, summary = load()
    print(summary.to_string())

    plot_bar(summary,
             'acc_mean', 'acc_std',
             'Average Accuracy — Split CIFAR-10',
             'Average Accuracy', 'fig1_accuracy.pdf')

    plot_bar(summary,
             'fgt_mean', 'fgt_std',
             'Forgetting — Split CIFAR-10',
             'Forgetting', 'fig2_forgetting.pdf',
             higher_better=False)

    plot_scatter(summary)
    print_latex(summary)

    print(f'\nAll figures saved to: {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
