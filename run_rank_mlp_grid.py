#!/usr/bin/env python3
"""Launch a sweep of rank_mlp_double.py with multiple layer/width combos."""
from __future__ import annotations

import argparse
import shlex
import subprocess
from itertools import product
from pathlib import Path
from typing import Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spawn multiple rank_mlp_double runs in parallel")
    parser.add_argument("--script", default="rank_mlp_double.py", help="Path to the single-run script")
    parser.add_argument(
        "--hidden-layer-counts",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="Hidden layer counts to sweep",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[1024],
        help="Hidden layer widths to sweep",
    )
    parser.add_argument("--initial-rank", type=int, default=2, help="Initial rank for every low-rank layer")
    parser.add_argument("--data", default="data1.h5")
    parser.add_argument("--sigma-prior", type=float, default=1.0)
    parser.add_argument("--sigma-lik", type=float, default=0.3)
    parser.add_argument("--train-start-step", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--num-train-windows", type=int, default=30)
    parser.add_argument("--max-total-train-steps", type=int, default=40000)
    parser.add_argument("--master-val-start", type=int, default=50000)
    parser.add_argument("--master-val-length", type=int, default=None)
    parser.add_argument("--growth-compression-ratio", type=float, default=0.01)
    parser.add_argument("--improvement-tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--loss-name", choices=["l1", "mse", "mixed"], default="l1")
    parser.add_argument("--warm-start", action="store_true", help="Enable warm start in child runs")
    parser.add_argument("--results-dir", default="rank_mlp_grid_results")
    parser.add_argument(
        "--extra-args",
        default="",
        help="Additional CLI arguments to append to each run (e.g. '--train-verbose 1').",
    )
    return parser.parse_args()


def build_command(
    args: argparse.Namespace,
    hidden_layers: int,
    hidden_dim: int,
    results_path: Path,
) -> list[str]:
    cmd = [
        "python",
        args.script,
        "--data",
        args.data,
        "--sigma-prior",
        str(args.sigma_prior),
        "--sigma-lik",
        str(args.sigma_lik),
        "--train-start-step",
        str(args.train_start_step),
        "--window-size",
        str(args.window_size),
        "--num-train-windows",
        str(args.num_train_windows),
        "--max-total-train-steps",
        str(args.max_total_train_steps),
        "--master-val-start",
        str(args.master_val_start),
        "--growth-compression-ratio",
        str(args.growth_compression_ratio),
        "--improvement-tol",
        str(args.improvement_tol),
        "--loss-name",
        args.loss_name,
        "--seed",
        str(args.seed),
        "--hidden-dim",
        str(hidden_dim),
        "--num-hidden-layers",
        str(hidden_layers),
        "--initial-rank",
        str(args.initial_rank),
        "--results-csv",
        str(results_path),
    ]
    if args.master_val_length is not None:
        cmd.extend(["--master-val-length", str(args.master_val_length)])
    if args.warm_start:
        cmd.append("--warm-start")
    extra = shlex.split(args.extra_args)
    cmd.extend(extra)
    return cmd


def ensure_sequence(values: Sequence[int], label: str) -> Sequence[int]:
    if not values:
        raise ValueError(f"{label} list is empty")
    return values


def main() -> int:
    args = parse_args()
    counts = ensure_sequence(args.hidden_layer_counts, "hidden-layer-counts")
    dims = ensure_sequence(args.hidden_dims, "hidden-dims")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent

    processes: list[tuple[tuple[int, int], subprocess.Popen]] = []

    for hidden_layers, hidden_dim in product(counts, dims):
        results_path = results_dir / f"rank_mlp_layers{hidden_layers}_width{hidden_dim}.csv"
        cmd = build_command(args, hidden_layers, hidden_dim, results_path)
        print(f"Launching L={hidden_layers}, W={hidden_dim}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, cwd=repo_root)
        processes.append(((hidden_layers, hidden_dim), proc))

    failed = False
    try:
        for (hidden_layers, hidden_dim), proc in processes:
            ret = proc.wait()
            if ret != 0:
                failed = True
                print(
                    f"Run failed for L={hidden_layers}, W={hidden_dim} with exit code {ret}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("KeyboardInterrupt received; terminating all child processes...", flush=True)
        for _, proc in processes:
            proc.terminate()
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
