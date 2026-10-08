"""
generate_figures.py
===================
Generates all figures for the VR-OCL project report.
Uses complete and correct results from all experiments.

Run from repo root with vrocl environment active:
    python generate_figures.py

Saves to: ./report_figures/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs('./report_figures', exist_ok=True)

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         12,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
    'axes.grid':         True,
    'grid.linestyle':    ':',
    'grid.alpha':        0.4,
})

C = {
    'ER':       '#4C72B0',
    'EWC':      '#55A868',
    'VR_OCL':   '#DD8452',
    'VR_Decay': '#C44E52',
    'VR_Adapt': '#9467BD',
}

# ── All confirmed results ──────────────────────────────────────────
# VR methods: best result at mu=0.0005
# Adaptive:   softer settings mu_max=0.02 tau=0.5
RESULTS = {
    'ER':       {'acc': 0.7038, 'acc_std': 0.0272, 'fgt': 0.2479, 'fgt_std': 0.0350},
    'EWC':      {'acc': 0.7069, 'acc_std': 0.0226, 'fgt': 0.2212, 'fgt_std': 0.0152},
    'VR_OCL':   {'acc': 0.7071, 'acc_std': 0.0257, 'fgt': 0.2022, 'fgt_std': 0.0475},
    'VR_Decay': {'acc': 0.6994, 'acc_std': 0.0180, 'fgt': 0.1889, 'fgt_std': 0.0618},
    'VR_Adapt': {'acc': 0.6214, 'acc_std': 0.0125, 'fgt': 0.1902, 'fgt_std': 0.0820},
}
ORDER  = ['ER', 'EWC', 'VR_OCL', 'VR_Decay', 'VR_Adapt']
LABELS = {
    'ER':       'ER\n(baseline)',
    'EWC':      'EWC\n(Kirkpatrick et al.)',
    'VR_OCL':   'VR-OCL\nfixed mu',
    'VR_Decay': 'VR-OCL\ndecay mu',
    'VR_Adapt': 'VR-OCL\nadaptive mu',
}

def save(name):
    plt.savefig(f'./report_figures/{name}.png', bbox_inches='tight', dpi=150)
    plt.savefig(f'./report_figures/{name}.pdf', bbox_inches='tight')
    print(f"Saved: {name}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# FIG 1 — Accuracy bar chart
# ═══════════════════════════════════════════════════════════════════
def fig1():
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ORDER))
    means  = [RESULTS[m]['acc']     for m in ORDER]
    stds   = [RESULTS[m]['acc_std'] for m in ORDER]
    colors = [C[m] for m in ORDER]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  alpha=0.85, width=0.55,
                  error_kw=dict(elinewidth=1.8, capthick=1.8))

    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.003,
                f'{mean:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
        if ORDER[i] in ['VR_OCL', 'VR_Decay', 'VR_Adapt']:
            bar.set_edgecolor('black')
            bar.set_linewidth(1.8)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in ORDER], fontsize=10)
    ax.set_ylabel('Average Accuracy')
    ax.set_title('Average Accuracy — Split CIFAR-10  (mem=500, mu=0.0005 for VR methods)')
    ax.set_ylim(0.55, 0.78)
    plt.tight_layout()
    save('fig1_accuracy')


# ═══════════════════════════════════════════════════════════════════
# FIG 2 — Forgetting bar chart
# ═══════════════════════════════════════════════════════════════════
def fig2():
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ORDER))
    means  = [RESULTS[m]['fgt']     for m in ORDER]
    stds   = [RESULTS[m]['fgt_std'] for m in ORDER]
    colors = [C[m] for m in ORDER]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  alpha=0.85, width=0.55,
                  error_kw=dict(elinewidth=1.8, capthick=1.8))

    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.003,
                f'{mean:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
        if ORDER[i] in ['VR_OCL', 'VR_Decay', 'VR_Adapt']:
            bar.set_edgecolor('black')
            bar.set_linewidth(1.8)

    
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in ORDER], fontsize=10)
    ax.set_ylabel('Forgetting')
    ax.set_title('Forgetting — Split CIFAR-10  (mem=500, mu=0.0005 for VR methods)')
    ax.set_ylim(0, 0.35)
    plt.tight_layout()
    save('fig2_forgetting')


# ═══════════════════════════════════════════════════════════════════
# FIG 3 — Scatter: Accuracy vs Forgetting
# ═══════════════════════════════════════════════════════════════════
def fig3():
    fig, ax = plt.subplots(figsize=(9, 7))
    offsets = {
        'ER': (10, 8), 'EWC': (10, 8),
        'VR_OCL': (10, -16), 'VR_Decay': (-85, 8), 'VR_Adapt': (10, 8)
    }
    for m in ORDER:
        r = RESULTS[m]
        ax.scatter(r['fgt'], r['acc'], color=C[m], s=200, zorder=3)
        ax.errorbar(r['fgt'], r['acc'], xerr=r['fgt_std'], yerr=r['acc_std'],
                    color=C[m], alpha=0.3, zorder=2, capsize=3)
        ox, oy = offsets[m]
        ax.annotate(LABELS[m].replace('\n', ' '), (r['fgt'], r['acc']),
                    textcoords='offset points', xytext=(ox, oy), fontsize=9)

    ax.annotate('', xy=(0.17, 0.725), xytext=(0.245, 0.670),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(0.197, 0.700, 'ideal\ndirection', fontsize=9,
            color='gray', ha='center', style='italic')
    ax.set_xlabel('Forgetting ')
    ax.set_ylabel('Average Accuracy  (higher is better  ↑)')
    ax.set_title('Accuracy vs Forgetting Trade-off — Split CIFAR-10')
    plt.tight_layout()
    save('fig3_scatter')


# ═══════════════════════════════════════════════════════════════════
# FIG 4 — Ablation: Effect of mu
# ═══════════════════════════════════════════════════════════════════
def fig4():
    mu_values  = [0.010,  0.005,  0.001,  0.0005]
    vr_accs    = [0.6670, 0.6755, 0.6883, 0.7071]
    vr_fgts    = [0.2080, 0.1859, 0.1815, 0.2022]
    decay_accs = [0.6570, 0.6890, 0.6927, 0.6994]
    decay_fgts = [0.1864, 0.2036, 0.1680, 0.1889]
    er_acc, er_fgt   = 0.7038, 0.2479
    ewc_acc, ewc_fgt = 0.7069, 0.2212

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(mu_values, vr_accs,    'o-', color=C['VR_OCL'],
             lw=2, ms=8, label='VR-OCL fixed mu')
    ax1.plot(mu_values, decay_accs, 's-', color=C['VR_Decay'],
             lw=2, ms=8, label='VR-OCL decay mu')
    ax1.axhline(er_acc,  color=C['ER'],  ls='--', lw=1.5,
                label=f'ER  ({er_acc:.3f})')
    ax1.axhline(ewc_acc, color=C['EWC'], ls='--', lw=1.5,
                label=f'EWC ({ewc_acc:.3f})')
    ax1.scatter([0.0005], [0.7071], color=C['VR_OCL'],
                s=150, zorder=5, edgecolors='black', lw=2)
    ax1.annotate('best', (0.0005, 0.7071),
                 textcoords='offset points', xytext=(-30, 8), fontsize=9)
    ax1.set_xlabel('mu')
    ax1.set_ylabel('Average Accuracy')
    ax1.set_title('Accuracy vs mu')
    ax1.set_xscale('log')
    ax1.invert_xaxis()
    ax1.legend(frameon=False, fontsize=9)

    ax2.plot(mu_values, vr_fgts,    'o-', color=C['VR_OCL'],
             lw=2, ms=8, label='VR-OCL fixed mu')
    ax2.plot(mu_values, decay_fgts, 's-', color=C['VR_Decay'],
             lw=2, ms=8, label='VR-OCL decay mu')
    ax2.axhline(er_fgt,  color=C['ER'],  ls='--', lw=1.5,
                label=f'ER  ({er_fgt:.3f})')
    ax2.axhline(ewc_fgt, color=C['EWC'], ls='--', lw=1.5,
                label=f'EWC ({ewc_fgt:.3f})')
    ax2.scatter([0.001],  [0.1680], color=C['VR_Decay'],
                s=150, zorder=5, edgecolors='black', lw=2)
    ax2.annotate('lowest forgetting', (0.001, 0.1680),
                 textcoords='offset points', xytext=(8, -22), fontsize=9)
    ax2.set_xlabel('mu')
    ax2.set_ylabel('Forgetting  (lower is better)')
    ax2.set_title('Forgetting vs mu')
    ax2.set_xscale('log')
    ax2.invert_xaxis()
    ax2.legend(frameon=False, fontsize=9)

    plt.suptitle('Ablation: Effect of mu — Split CIFAR-10 (mem=500)', fontsize=12, y=1.01)
    plt.tight_layout()
    save('fig4_ablation')


# ═══════════════════════════════════════════════════════════════════
# FIG 5 — EWC Heatmap
# ═══════════════════════════════════════════════════════════════════
def fig5():
    ewc_matrix = np.array([
        [0.92, np.nan, np.nan, np.nan, np.nan],
        [0.72, 0.84,   np.nan, np.nan, np.nan],
        [0.60, 0.47,   0.90,   np.nan, np.nan],
        [0.62, 0.63,   0.77,   0.62,   np.nan],
        [0.49, 0.68,   0.47,   0.67,   0.55  ],
    ])
    fig, ax = plt.subplots(figsize=(7, 6))
    masked  = np.ma.masked_where(np.isnan(ewc_matrix), ewc_matrix)
    im      = ax.imshow(masked, cmap='viridis', vmin=0.4, vmax=0.95, aspect='auto')
    cbar    = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f'T{i+1}' for i in range(5)])
    ax.set_yticklabels([f'After T{i+1}' for i in range(5)])
    ax.set_xlabel('Evaluated Task')
    ax.set_ylabel('Training Progress')
    ax.set_title('EWC — Per-Task Accuracy Over Training')
    for i in range(5):
        for j in range(5):
            v = ewc_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='white' if v < 0.65 else 'black')
    plt.tight_layout()
    save('fig5_ewc_heatmap')


# ═══════════════════════════════════════════════════════════════════
# FIG 6 — Full Tradeoff Summary (all mu values on one plot)
# ═══════════════════════════════════════════════════════════════════
def fig6():
    fig, ax = plt.subplots(figsize=(10, 7))

    # Trajectory points
    vr_pts    = [(0.2080,0.6670),(0.1859,0.6755),(0.1815,0.6883),(0.2022,0.7071)]
    decay_pts = [(0.1864,0.6570),(0.2036,0.6890),(0.1680,0.6927),(0.1889,0.6994)]

    ax.plot([p[0] for p in vr_pts],    [p[1] for p in vr_pts],
            '-', color=C['VR_OCL'],   alpha=0.35, lw=1.5)
    ax.plot([p[0] for p in decay_pts], [p[1] for p in decay_pts],
            '--', color=C['VR_Decay'], alpha=0.35, lw=1.5)

    mu_labels = ['mu=0.01', 'mu=0.005', 'mu=0.001', 'mu=0.0005']
    for i, (fgt, acc) in enumerate(vr_pts):
        sz = 160 if i == 3 else 70
        ec = 'black' if i == 3 else 'none'
        ax.scatter(fgt, acc, color=C['VR_OCL'], s=sz, zorder=3,
                   edgecolors=ec, linewidth=1.5)
        if i == 3:
            ax.annotate(mu_labels[i], (fgt, acc),
                        textcoords='offset points', xytext=(8, 6), fontsize=8)

    for i, (fgt, acc) in enumerate(decay_pts):
        sz = 160 if i == 3 else 70
        ec = 'black' if i == 3 else 'none'
        ax.scatter(fgt, acc, color=C['VR_Decay'], marker='s', s=sz,
                   zorder=3, edgecolors=ec, linewidth=1.5)
        if i == 3:
            ax.annotate(mu_labels[i], (fgt, acc),
                        textcoords='offset points', xytext=(8, -14), fontsize=8)

    # Baselines and adaptive
    for m, marker in [('ER','D'),('EWC','D'),('VR_Adapt','^')]:
        r = RESULTS[m]
        ax.scatter(r['fgt'], r['acc'], color=C[m], marker=marker,
                   s=180, zorder=4, edgecolors='black', linewidth=1.5,
                   label=LABELS[m].replace('\n',' '))
        ax.errorbar(r['fgt'], r['acc'],
                    xerr=r['fgt_std'], yerr=r['acc_std'],
                    color=C[m], alpha=0.3, zorder=2, capsize=3)

    legend_els = [
        plt.scatter([],[],color=C['ER'],      marker='D',s=100,edgecolors='black',lw=1.5,label='ER'),
        plt.scatter([],[],color=C['EWC'],     marker='D',s=100,edgecolors='black',lw=1.5,label='EWC'),
        plt.scatter([],[],color=C['VR_OCL'],  marker='o',s=100,edgecolors='black',lw=1.5,label='VR-OCL fixed (best)'),
        plt.scatter([],[],color=C['VR_Decay'],marker='s',s=100,edgecolors='black',lw=1.5,label='VR-OCL decay (best)'),
        plt.scatter([],[],color=C['VR_Adapt'],marker='^',s=100,edgecolors='black',lw=1.5,label='VR-OCL adaptive'),
        plt.Line2D([0],[0],color=C['VR_OCL'], lw=1.5,alpha=0.5,label='VR-OCL fixed trajectory'),
        plt.Line2D([0],[0],color=C['VR_Decay'],lw=1.5,ls='--',alpha=0.5,label='VR-OCL decay trajectory'),
    ]
    ax.legend(handles=legend_els, frameon=False, fontsize=8, loc='lower left')
    ax.annotate('', xy=(0.16,0.725), xytext=(0.245,0.660),
                arrowprops=dict(arrowstyle='->',color='gray',lw=2))
    ax.text(0.195,0.695,'ideal', fontsize=9,color='gray',style='italic')
    ax.set_xlabel('Forgetting  (lower is better  ←)')
    ax.set_ylabel('Average Accuracy  (higher is better  ↑)')
    ax.set_title('Full Stability-Plasticity Tradeoff — All VR-OCL Variants\n(Large markers = best per method, small = other mu values)')
    plt.tight_layout()
    save('fig6_tradeoff_summary')


# ── Run all ────────────────────────────────────────────────────────
print("Generating all figures...\n")
fig1()
fig2()
fig3()
fig4()
fig5()
fig6()
print(f"\nAll done. Figures in ./report_figures/")