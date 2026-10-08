# Provenance of saved results

This file records where every result folder came from and how it is used. `manifest.csv` is the machine-readable version.

- **Headline results:** the folders with status `valid`. These are the runs on the author's own machine, under `local/`.
- **Archive:** the folders with status `archive`, under `archive_kaggle/`. These are Kaggle runs, kept for the record. `scripts/make_report_tables.py` summarises them separately in `archive_kaggle/archive_tables.md` and never mixes them into the headline tables or figures.
- **Not used at all:** the folders with status `invalid` or `duplicate`.

All runs use Split CIFAR-10, 5 tasks, memory batch 64, stream batch 10, 1 epoch per task, reduced ResNet-18 (nf=20), Adam with **lr = 0.0005**, and seeds 0, 1 and 2. These values come from every `params_used.json`. Memory size is 500 unless stated otherwise.

**Dates.** `params_used.json` does not store a date. The Kaggle folders were copied into the repo on 2026-10-08, so their file times are meaningless. The dates below come from the execution timestamps (UTC) of the author's private Kaggle notebooks, where a notebook output shows the run, or from the folder modification time for local runs. Runs made with `scripts/run_vrocl.py` from now on write the date, git commit and device to `run_config.json`.

**Local vs Kaggle.** "Local" means the author's own machine: the stored `results_root` is a local Windows path or a relative `./results_*` path. "Kaggle" means a Kaggle notebook (the paths were `/kaggle/working/...`). The two give different ER baselines for the same seeds and settings, so they are never compared with each other.

**Code version.** Local runs used the code in this repo. Kaggle runs used copies of the code pasted into notebooks that were edited between runs, so which code version produced each Kaggle folder cannot be confirmed. This is why they are archived.

**Notebooks.** The Kaggle notebooks, together with screenshots and older figure images, are kept privately by the author and are not in this repository. Where they are cited below, the cell and timestamp are given so the author can check them.

## Headline (valid): local runs

| Folder | Contents | Date | Evidence |
|---|---|---|---|
| `local/baselines_vr_mu0p001_adaptive_mumax0p05_tau0p3` | ER, EWC, VR_OCL μ=0.001, VR_OCL_Decay μ=0.001, VR_OCL_Adaptive (accuracy-gap code) μmax=0.05 τ=0.3 | folder last modified 2026-05-30 19:40 (local time) | formerly `results_local/`; its `summary.csv` matches the per-run files |
| `local/vr_mu0p0005` | VR_OCL and VR_OCL_Decay μ=0.0005 | folder last modified 2026-05-30 23:40 | formerly `results_mu0005/` |
| `local/adaptive_mumax0p02_tau0p5` | VR_OCL_Adaptive (accuracy-gap code) μmax=0.02 τ=0.5 | folder last modified 2026-05-31 01:09 | formerly `results_adaptive_v2/` |

## Archive: Kaggle runs (code version unconfirmed)

| Folder | Contents | Date | Evidence |
|---|---|---|---|
| `archive_kaggle/baselines` | ER, EWC | 2026-05-17, 17:56–18:24 UTC | private notebook `variancecl.ipynb`, cells 10–14, print the matching ER summary |
| `archive_kaggle/vr_mu0p01` | VR_OCL and VR_OCL_Decay μ=0.01 | unknown (no saved notebook output) | formerly `results_extra/vrocl_final_results/`; `vr_mu`=0.01 in every params file |
| `archive_kaggle/vr_mu0p005` | VR_OCL and VR_OCL_Decay μ=0.005 | unknown (no saved notebook output) | formerly `results_extra/vrocl_mu0005_results/`. The folder name suggested 0.0005, but `vr_mu`=0.005 in every params file |
| `archive_kaggle/adaptive_v1lossgap_mumax0p05_tau0p5` | VR_OCL_Adaptive, **older loss-gap code** (its `gap_history.csv` has no `best_mem` column and the gap can be negative), μmax=0.05 τ=0.5 | 2026-05-19 09:18 UTC | private notebook `variancecl (6).ipynb`, cell 14 |

## Invalid (kept for the record, excluded from all tables)

| Folder | Why |
|---|---|
| `archive_kaggle/invalid/kaggle_results_adaptive_final_er_vr_decay` | For every seed, the VR_OCL and VR_OCL_Decay `acc.csv` files are **byte-identical to ER's**, so the regularizer had no effect in this run. Its ER and EWC are copies of `archive_kaggle/baselines`, and its `summary.csv` therefore reports VR = ER, which is wrong. |
| `archive_kaggle/invalid/kaggle_vr_mu0p1_backup` | Early μ=0.1 run with `vr_mu_cap`=10, superseded and not used in the report. |
| `archive_kaggle/invalid/kaggle_mem1000_mu0p001_incomplete` | mem=1000 run. `all_results.csv` and `final_summary.csv` are empty because the run did not finish its summary step. Not used in the report. |

## Duplicates (excluded)

| Folder | Duplicate of |
|---|---|
| `archive_kaggle/duplicates/kaggle_vr_mu0p01_copy1` | Byte-identical to `archive_kaggle/vr_mu0p01` (formerly `vrocl_mu001_decayfix_results/.../results_vrocl`). Despite the name "decayfix", the results did not change. |
| `archive_kaggle/duplicates/kaggle_vr_mu0p01_copy2` | Same as `archive_kaggle/vr_mu0p01`, minus its two summary CSVs (formerly `.../tuned_results`). |

## Unverified (notebook output only, no result files)

These values were printed in a notebook, but their per-run files were not saved: they were written to `/kaggle/working/results_vrocl` and later overwritten. They are **not** in any table.

| Values | Source | Notes |
|---|---|---|
| VR_OCL μ=0.001: 68.83 ± 2.32 / 18.15 ± 4.47; VR_OCL_Decay μ=0.001: 69.27 ± 3.93 / 16.80 ± 2.46 (Kaggle, mem 500) | author's private notebook `variancecl (4).ipynb`, cell 61, 2026-05-17 20:48 UTC (not in this repository) | These are the report's Table 4.2 μ=0.001 row. The local run at μ=0.001 gives different values (see `docs/results_tables.md`). |
