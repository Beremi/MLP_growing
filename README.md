# MLP Growing Experiments

Python experiments for growing multilayer perceptrons that act as surrogates inside delayed-acceptance MCMC. The repo contains two main experiment families:
- Dense baselines that retrain or fine-tune fixed-width MLPs as more chain data arrives.
- Low-rank MLPs whose rank doubles and is optionally compressed on-the-fly to increase capacity without full retrains.

The code expects a local HDF5 dataset and writes CSV logs of every training window so sweeps can be resumed or plotted later.

## Repository Layout
- `test_mlp.py` — core dense MLP training/eval with Adam + LBFGS, optional standardization, log-posterior loss, and DA-MH diagnostics.
- `rank_mlp_double.py` — low-rank MLP definition, rank expansion/compression utilities, and the doubling loop used in the paper/poster experiments.
- `run_all_mlp.py` — spawns a grid of dense baselines (hidden layer counts/sizes) in parallel via `test_mlp.py`.
- `run_fixed_rank_lowrank.py` — compares fixed low ranks; runs `rank_mlp_double.py` with `--disable-rank-growth`.
- `run_rank_mlp_grid.py` — sweeps hidden-layer counts/widths for the low-rank model with growth enabled.
- `adaptive_rank_1024/`, `results_fixed_rank_lowrank/`, `results_tanh_L1/`, `mlp_grow_progress/`, `old_results/`, `results_tanh_*`, `mlp_grow_progress*` — CSV logs from previous sweeps. The filenames encode the configs; columns mirror the fields written in the scripts (train sizes, validation errors, ranks before/after growth, etc.).
- Notebooks: `mlp_growth_experiment.ipynb`, `regrowing_rank_MLP.ipynb`, `rank_MLP.ipynb`, `surrogate_mlp_dropout_test.ipynb`, `final_chain_analysis.ipynb`, `mlp_results_visualization.ipynb`, `test_NN.ipynb`, `test_MLP.ipynb` for exploratory analysis and plotting.
- Data: `data1.h5` (main) and `data.h5` (shorter chain) with datasets:
  - `par` `(28324, 30)` parameter vectors
  - `obs` `(28324, 52)` simulator outputs
  - `y_obs` `(52,)` observation used in the log posterior
  - `chain`, `props` proposal indices through time (lengths 56646 for `data1`, 20000 for `data`)
  - `logpi` `(28324,)` precomputed unnormalized log posterior (computed on the fly if missing)

## Quick Start
Create an environment with PyTorch + common numerics:
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas h5py
```
Optional: add `matplotlib`, `ipykernel`, `jupyter` if you want to run the notebooks.

## Dense Baselines (`test_mlp.py`)
- Runs a single MLP on progressively larger training windows drawn from the chain/proposal indices.
- Default training: Adam (patience-based LR drops) followed by LBFGS polishing; optional mixed L1/MSE loops; optional batch growth per outer loop.
- Outputs a CSV per configuration containing train/val errors and delayed-acceptance metrics:
  - `val_mse_obs`, `val_logpi_l1_error`, `mean_da_reject`, `mean_da_accept`, `mean_a1`, `mean_a2_reject`, etc.
- Example (3×64 tanh MLP, L1 loss, training on `data1.h5`):
```bash
python test_mlp.py --hidden-sizes 64 64 64 --data data1.h5 --activation tanh \
  --sigma-prior 1.0 --sigma-lik 0.3 --train-step 2000 --train-cutoff 50000 \
  --val-size 6646 --adam-epochs 2000 --adam-lr 1e-3 --adam-patience 200 \
  --lbfgs-steps 20 --train-loops 80 --retrain --train-loss l1 \
  --batch-size 32 --batch-growth 1.2 --output-dir results_tanh_Retrain_new
```
- Launch the full grid used in the repo with `python run_all_mlp.py` (spawns one process per layer-count/width combo).

## Low-Rank Rank-Growth Experiments (`rank_mlp_double.py`)
- Implements `LowRankMLP` where every hidden→hidden transition uses a factored `U @ V` weight. Ranks can grow (duplicate columns with small noise), then be SVD-compressed by a singular-value ratio.
- Training windows: sliding windows over the chain (train window + adjacent validation window), with optional warm starts.
- Each growth trial logs before/after ranks, singular-value ratios, expand/compress losses, and validation errors; checkpoints are written incrementally to the configured CSV.
- Typical run (3 hidden layers of width 1024, starting rank 2, doubling ranks until no benefit):
```bash
python rank_mlp_double.py --data data1.h5 --hidden-dim 1024 --num-hidden-layers 3 \
  --initial-rank 2 --train-start-step 2000 --window-size 2000 \
  --max-total-train-steps 20000 --master-val-start 50000 \
  --growth-compression-ratio 0.05 --improvement-tol 1e-5 \
  --loss-name l1 --results-csv mlp_grow_progress/mlp_growth_progress.csv
```
- Sweep helpers:
  - `python run_rank_mlp_grid.py --hidden-layer-counts 3 4 5 --hidden-dims 1024 --initial-rank 2` to vary depth/width.
  - `python run_fixed_rank_lowrank.py` to benchmark fixed ranks (no growth) and drop CSVs in `results_fixed_rank_lowrank/`.

## Notebooks and Outputs
- Exploration/plotting notebooks read the CSVs above; `mlp_results_visualization.ipynb` is handy for quick sanity plots.
- Historical CSVs are left intact so you can compare against new runs; PDFs/PNGs in the root are snapshot figures for dense vs low-rank runs.

## Future Work and Growth Triggers
- The intended strategy for online growth without full retrains:
  - Triggers: validation-EMA plateau at high error, diminishing improvement from new data, high/flat train vs val, residual structure, and dataset-to-parameter ratio becoming too large.
  - Growth operators: rank increases of factored layers (or LoRA-style adapters), Net2Wider/Net2Deeper morphisms that preserve the current function, neuron splitting for stressed units, and slimmable/width-switchable layers for instant capacity scaling.
  - Policy sketch: require two triggers before scheduling growth, cap growth frequency, and fine-tune new parameters first with a higher LR while keeping Stage-1 DA surrogate deterministic.
- These ideas map cleanly onto `LowRankMLP`: keep `rank_mlp_double.py` for rank-based growth, then add Net2Net-style widen/deepen helpers if you need neuron-level expansion without downtime.

## Tips
- The HDF5 loader computes `logpi` if absent; adjust `sigma_prior`/`sigma_lik` to match your target.
- `--standardize` in `test_mlp.py` applies per-feature mean/std computed on the current training window.
- Many scripts spawn parallel processes; monitor GPU/CPU load before running the full grids.
- CSVs are overwritten if paths collide—keep per-run folders to avoid confusion.
