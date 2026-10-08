# Experimental: entropy-guided replay (unfinished)

This is an earlier, **unfinished** idea from this project. It is **not** part of the VR-OCL report and has no saved results.

The idea was to replay the buffer samples the model is most uncertain about, meaning those with the highest softmax entropy, instead of uniformly random ones. It was tried both with plain ER and with MKD (EMA teacher).

| File | Contents |
|---|---|
| `entropy_reservoir.py` | Reservoir buffer with entropy-ranked and mixed random/entropy retrieval |
| `er_entropy.py` | ER with the mixed retrieval (`ER_Entropy`) |
| `er_ema_entropy.py` | ER + MKD with entropy retrieval and optional random EMA alpha (`ER_EMA_Entropy`) |
| `run_experiments.py` | Runs ER / ER_Entropy / ER_EMA / ER_EMA_Entropy, writing to `results/experimental_entropy/` |
| `plot_cl_results.py` | Plots that folder's `all_results.csv` |

Run from the repo root with `python experimental/run_experiments.py`. Nothing has been verified, so treat any output as preliminary.
