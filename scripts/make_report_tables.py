"""
make_report_tables.py
=====================
Builds every results table from the saved per-run files listed in
results/manifest.csv. No number is typed in by hand.

Headline tables use only folders with status `valid` (the local runs).
Folders with status `archive` (the Kaggle runs, code version
unconfirmed) get a separate, clearly labelled archive table so the
numbers quoted from them in docs/report_errata.md stay traceable.

Metrics (all in %, computed from each run's avg.csv / avg_forgetting.csv):
  aia        average incremental accuracy: mean over the 5 task checkpoints
             of the average accuracy on the tasks seen so far
             (what the report's tables call "Avg Accuracy")
  avg_fgt    average incremental forgetting: same averaging for forgetting
             (what the report's tables call "Forgetting")
  final_acc  average accuracy over all 5 tasks after the last task
  final_fgt  average forgetting after the last task

Each method is compared only with the ER baseline from the same
set of runs (never local vs Kaggle).

Usage (from the repo root):
    python scripts/make_report_tables.py

Writes results/tables/*.csv and docs/results_tables.md (headline),
and results/archive_kaggle/tables/*.csv and
results/archive_kaggle/archive_tables.md (archive).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / 'results'
OUTPUTS = {
    # status: (tables folder, markdown file)
    'valid': (RESULTS / 'tables', REPO_ROOT / 'docs' / 'results_tables.md'),
    'archive': (RESULTS / 'archive_kaggle' / 'tables', RESULTS / 'archive_kaggle' / 'archive_tables.md'),
}

METRICS = ['aia', 'final_acc', 'avg_fgt', 'final_fgt']
METRIC_NAMES = {
    'aia': 'Avg. incremental acc. (%)',
    'final_acc': 'Final acc. (%)',
    'avg_fgt': 'Avg. incremental forgetting (%)',
    'final_fgt': 'Final forgetting (%)',
}
METHOD_ORDER = ['ER', 'EWC', 'VR_OCL', 'VR_OCL_Decay', 'VR_OCL_Adaptive']


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def adaptive_variant(folder, seed):
    """The accuracy-gap code logs best_mem; the older loss-gap code does not."""
    f = folder / f'gap_VR_OCL_Adaptive_seed{seed}' / 'gap_history.csv'
    if not f.exists():
        return 'unknown'
    cols = pd.read_csv(f, nrows=0).columns
    return 'acc-gap' if 'best_mem' in cols else 'loss-gap'


def config_label(method, params, variant):
    if method in ('VR_OCL', 'VR_OCL_Decay'):
        return f"mu={params['vr_mu']:g}"
    if method == 'VR_OCL_Adaptive':
        return f"{variant}, mu_max={params['vr_mu_max']:g}, tau={params['vr_tau']:g}"
    return '-'


def load_runs(status):
    manifest = pd.read_csv(RESULTS / 'manifest.csv')
    selected = manifest[manifest.status == status]
    rows = []
    for _, m in selected.iterrows():
        folder = RESULTS / m.path
        for p in sorted(folder.glob('*/run*/params_used.json')):
            params = json.load(open(p))
            run = p.parent
            avg = pd.read_csv(run / 'avg.csv').iloc[:, 0].to_numpy(dtype=float)
            fgt = pd.read_csv(run / 'avg_forgetting.csv').iloc[:, 0].to_numpy(dtype=float)
            method, seed = params['learner'], params['seed']
            variant = adaptive_variant(folder, seed) if method == 'VR_OCL_Adaptive' else ''
            rows.append({
                'environment': m.environment,
                'folder': m.path,
                'method': method,
                'config': config_label(method, params, variant),
                'mu': params.get('vr_mu') if method in ('VR_OCL', 'VR_OCL_Decay') else np.nan,
                'seed': seed,
                'mem_size': params['mem_size'],
                'lr': params['learning_rate'],
                'class_order': ' '.join(map(str, params['labels_order'])),
                'aia': np.nanmean(avg) * 100,
                'avg_fgt': np.nanmean(fgt) * 100,
                'final_acc': avg[-1] * 100,
                'final_fgt': fgt[-1] * 100,
            })
    runs = pd.DataFrame(rows)
    runs['method_rank'] = runs.method.map({m: i for i, m in enumerate(METHOD_ORDER)})
    return runs.sort_values(['environment', 'method_rank', 'config', 'seed']).drop(columns='method_rank')


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------

def summarise(runs):
    g = runs.groupby(['environment', 'method', 'config', 'folder'], sort=False)
    out = g.agg(n_seeds=('seed', 'count'), seeds=('seed', lambda s: ','.join(map(str, sorted(s)))))
    for k in METRICS:
        out[f'{k}_mean'] = g[k].mean()
        out[f'{k}_std'] = g[k].std(ddof=1)  # same std as the original summary.csv files
    return out.reset_index()


def deltas_vs_er(runs):
    """Per-seed difference from the ER run with the same seed in the same environment."""
    er = runs[runs.method == 'ER'].set_index(['environment', 'seed'])[METRICS]
    others = runs[runs.method != 'ER'].copy()
    for k in METRICS:
        others[f'd_{k}'] = [
            row[k] - er.loc[(row.environment, row.seed), k] for _, row in others.iterrows()
        ]
    return others[['environment', 'method', 'config', 'seed'] + [f'd_{k}' for k in METRICS]]


def delta_summary(d):
    g = d.groupby(['environment', 'method', 'config'], sort=False)
    out = g.size().rename('n_seeds').to_frame()
    for k in METRICS:
        out[f'd_{k}_mean'] = g[f'd_{k}'].mean()
    # How many seeds are better than ER (higher accuracy / lower forgetting)
    out['seeds_aia_above_ER'] = g['d_aia'].apply(lambda s: f'{(s > 0).sum()}/{len(s)}')
    out['seeds_final_acc_above_ER'] = g['d_final_acc'].apply(lambda s: f'{(s > 0).sum()}/{len(s)}')
    out['seeds_avg_fgt_below_ER'] = g['d_avg_fgt'].apply(lambda s: f'{(s < 0).sum()}/{len(s)}')
    out['seeds_final_fgt_below_ER'] = g['d_final_fgt'].apply(lambda s: f'{(s < 0).sum()}/{len(s)}')
    return out.reset_index()


def order_check(runs):
    rows = []
    for seed, g in runs.groupby('seed'):
        rows.append({
            'seed': seed,
            'n_runs': len(g),
            'environments': ','.join(sorted(g.environment.unique())),
            'distinct_class_orders': g.class_order.nunique(),
            'class_order': g.class_order.iloc[0] if g.class_order.nunique() == 1 else 'MISMATCH',
        })
    return pd.DataFrame(rows)


def seed_coverage(runs):
    """Every configuration should have been run with the same seeds as its ER baseline."""
    er_seeds = runs[runs.method == 'ER'].groupby('environment').seed.apply(set)
    rows = []
    for (env, method, config), g in runs.groupby(['environment', 'method', 'config'], sort=False):
        rows.append({'environment': env, 'method': method, 'config': config,
                     'seeds': ','.join(map(str, sorted(g.seed))),
                     'same_seeds_as_ER': set(g.seed) == er_seeds[env]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Cross-check against the summary CSVs that were saved with the runs
# ----------------------------------------------------------------------

def cross_check(summary):
    problems, checked = [], 0
    for folder in summary.folder.unique():
        for f in (RESULTS / folder).glob('*summary*.csv'):
            saved = pd.read_csv(f)
            if saved.empty:
                continue
            for _, s in saved.iterrows():
                mine = summary[(summary.folder == folder) & (summary.method == s.method)]
                if len(mine) != 1:
                    continue
                mine = mine.iloc[0]
                checked += 1
                for a, b in [('acc_mean', 'aia_mean'), ('acc_std', 'aia_std'),
                             ('fgt_mean', 'avg_fgt_mean'), ('fgt_std', 'avg_fgt_std')]:
                    if abs(s[a] * 100 - mine[b]) > 0.006:  # saved files are rounded to 4 d.p.
                        problems.append(f'{f.relative_to(REPO_ROOT)} {s.method} {a}: '
                                        f'saved {s[a] * 100:.2f}, recomputed {mine[b]:.2f}')
    return checked, problems


# ----------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------

def pm(mean, std):
    return f'{mean:.2f} ± {std:.2f}'


def md_table(df, cols, headers):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(str(c(r)) if callable(c) else str(r[c]) for c in cols) + ' |')
    return '\n'.join(lines)


def metric_cols():
    return [lambda r, k=k: pm(r[f'{k}_mean'], r[f'{k}_std']) for k in METRICS]


INTROS = {
    'valid': [
        '# Results tables (local runs)',
        '',
        '_Generated by `scripts/make_report_tables.py` from the folders with status `valid` in '
        '`results/manifest.csv`. Do not edit by hand. Provenance for each folder is in '
        '`results/PROVENANCE.md`._',
        '',
        "These are the headline results and use **only the runs from the author's own machine**. "
        'Earlier Kaggle runs are archived and not used here (their code version cannot be '
        'confirmed): see `results/archive_kaggle/README.md`.',
    ],
    'archive': [
        '# ARCHIVE: Kaggle runs (not headline results)',
        '',
        '_Generated by `scripts/make_report_tables.py` from the folders with status `archive` in '
        '`results/manifest.csv`. Do not edit by hand._',
        '',
        '**These runs are kept for the record only.** They ran on notebook copies of the code that '
        'were edited between runs, so the code version behind each folder cannot be confirmed, and '
        'their ER baseline differs from the local one. Do not compare them with the local tables '
        'in `docs/results_tables.md`.',
    ],
}
ABLATION_NOTES = {
    'valid': 'Only mu values with saved result files are listed.',
    'archive': 'Kaggle mu = 0.001 is not included: it exists only as printed output in the '
               "author's private notebook (unverified, see results/PROVENANCE.md).",
}


def write_markdown(status, doc, summary, ablation, deltas, dsum, order, coverage, checked, problems):
    envs = list(dict.fromkeys(summary.environment))
    out = INTROS[status] + [
        '',
        'All values are mean ± sample std over seeds 0, 1, 2, in %. '
        '**Avg. incremental acc.** is the metric the report labels "Avg Accuracy"; '
        '**Avg. incremental forgetting** is the one it labels "Forgetting". '
        'Final acc./forgetting are measured once, after the last task. '
        'Each method is compared only with the ER run from the same set of runs.',
        '',
    ]
    for env in envs:
        s = summary[summary.environment == env]
        out += [f'## All runs: {env}', '',
                md_table(s, ['method', 'config', 'seeds'] + metric_cols() + ['folder'],
                         ['Method', 'Config', 'Seeds'] + [METRIC_NAMES[k] for k in METRICS] + ['Folder']),
                '']

    out += ['## mu ablation', '', ABLATION_NOTES[status], '']
    for env in envs:
        a = ablation[ablation.environment == env]
        er = summary[(summary.environment == env) & (summary.method == 'ER')].iloc[0]
        out += [f'### {env} (ER: avg. incremental acc. '
                f'{pm(er.aia_mean, er.aia_std)}, avg. incremental forgetting '
                f'{pm(er.avg_fgt_mean, er.avg_fgt_std)}, final acc. '
                f'{pm(er.final_acc_mean, er.final_acc_std)}, final forgetting '
                f'{pm(er.final_fgt_mean, er.final_fgt_std)})', '',
                md_table(a, ['mu', 'method'] + metric_cols(),
                         ['mu', 'Method'] + [METRIC_NAMES[k] for k in METRICS]),
                '']

    out += ['## Difference from ER, per seed', '',
            'Δ = method − ER for the same seed, in percentage points. '
            'Positive Δ accuracy and negative Δ forgetting favour the method.', '',
            md_table(deltas, ['environment', 'method', 'config', 'seed']
                     + [lambda r, k=k: f"{r[f'd_{k}']:+.2f}" for k in METRICS],
                     ['Env', 'Method', 'Config', 'Seed'] + [f'Δ {METRIC_NAMES[k]}' for k in METRICS]),
            '', '### Summary of the per-seed differences', '',
            md_table(dsum, ['environment', 'method', 'config']
                     + [lambda r, k=k: f"{r[f'd_{k}_mean']:+.2f}" for k in METRICS]
                     + ['seeds_aia_above_ER', 'seeds_final_acc_above_ER',
                        'seeds_avg_fgt_below_ER', 'seeds_final_fgt_below_ER'],
                     ['Env', 'Method', 'Config'] + [f'mean Δ {METRIC_NAMES[k]}' for k in METRICS]
                     + ['seeds: avg. inc. acc > ER', 'seeds: final acc > ER',
                        'seeds: avg. inc. fgt < ER', 'seeds: final fgt < ER']),
            '']

    all_match = (order.distinct_class_orders == 1).all()
    out += ['## Seeds and data order', '',
            md_table(order, ['seed', 'n_runs', 'environments', 'distinct_class_orders', 'class_order'],
                     ['Seed', 'Runs', 'Environments', 'Distinct class orders', 'Class order (task split)']),
            '',
            md_table(coverage, ['environment', 'method', 'config', 'seeds', 'same_seeds_as_ER'],
                     ['Env', 'Method', 'Config', 'Seeds', 'Same seeds as ER']),
            '',
            ('For each seed, the class-to-task order recorded in `params_used.json` is identical '
             'across every run in these tables.' if all_match else
             '**The class order differs between runs with the same seed.**') +
            ' The order of samples *within* a task (DataLoader shuffling) is not recorded, so '
            'it cannot be verified from the files; it depends on the torch RNG state, '
            'which is seeded identically but is consumed by training.',
            '',
            '## Cross-check against saved summary files', '',
            f'{checked} method rows in the summary CSVs saved with the runs were compared with the '
            f'values recomputed here.' + (' All match to rounding.' if not problems else ' Mismatches:'),
            ]
    out += [f'- {p}' for p in problems]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text('\n'.join(out) + '\n', encoding='utf-8')


def build(status):
    tables, doc = OUTPUTS[status]
    runs = load_runs(status)
    summary = summarise(runs)
    ablation = summary[summary.method.isin(['VR_OCL', 'VR_OCL_Decay'])].copy()
    ablation['mu'] = ablation.config.str.replace('mu=', '').astype(float)
    ablation = ablation.sort_values(['environment', 'mu', 'method'], ascending=[True, False, True])
    deltas = deltas_vs_er(runs)
    dsum = delta_summary(deltas)
    order = order_check(runs)
    coverage = seed_coverage(runs)
    checked, problems = cross_check(summary)

    tables.mkdir(parents=True, exist_ok=True)
    runs.to_csv(tables / 'runs.csv', index=False)
    summary.to_csv(tables / 'summary.csv', index=False)
    ablation.to_csv(tables / 'mu_ablation.csv', index=False)
    deltas.to_csv(tables / 'deltas_vs_er_per_seed.csv', index=False)
    dsum.to_csv(tables / 'deltas_vs_er_summary.csv', index=False)
    order.to_csv(tables / 'class_order_check.csv', index=False)
    coverage.to_csv(tables / 'seed_coverage.csv', index=False)
    write_markdown(status, doc, summary, ablation, deltas, dsum, order, coverage, checked, problems)

    print(f'[{status}] {len(runs)} runs from {summary.folder.nunique()} folders, '
          f'environments: {sorted(runs.environment.unique())}')
    print(f'[{status}] cross-check: {checked} saved summary rows compared, {len(problems)} mismatches')
    for p in problems:
        print('  ', p)
    print(f'[{status}] wrote {tables.relative_to(REPO_ROOT).as_posix()}/ and '
          f'{doc.relative_to(REPO_ROOT).as_posix()}')
    return problems


def main():
    problems = build('valid') + build('archive')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
