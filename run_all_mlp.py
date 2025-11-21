#!/usr/bin/env python3
"""Launches test_mlp.py with multiple layer/size combos in parallel."""
from itertools import product
from pathlib import Path
import subprocess
import sys

HIDDEN_LAYER_COUNTS = (1, 2, 3)
HIDDEN_LAYER_SIZES = (8, 16, 32, 64, 128)

BASE_COMMAND = ["python", "test_mlp.py"]


def build_command(layer_count: int, layer_size: int) -> list[str]:
    hidden_sizes = [str(layer_size)] * layer_count
    return [
        *BASE_COMMAND,
        "--hidden-sizes",
        *hidden_sizes,
        "--activation",
        "tanh",
        "--data",
        "data1.h5",
        "--sigma-prior",
        "1.0",
        "--sigma-lik",
        "0.3",
        "--train-step",
        "2000",
        "--train-cutoff",
        "50000",
        "--train-window",
        "20000",
        # "--progress-train-step",
        # "--train-step-multiplier",
        # "1",
        "--val-size",
        "6646",
        "--adam-epochs",
        "2000",
        "--adam-lr",
        "1e-3",
        "--adam-patience",
        "200",
        "--lbfgs-steps",
        "20",
        "--train-loops",
        "80",
        "--retrain",
        "--train-loss",
        "l1",
        "--loop-improvement-pct",
        "0.1",
        "--loss-domain",
        "obs",
        "--batch-size",
        "32",
        "--batch-growth",
        "1.2",
        "--output-dir",
        "results_tanh_Retrain_new",
    ]


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    processes: list[tuple[tuple[int, int], subprocess.Popen]] = []

    for layer_count, layer_size in product(HIDDEN_LAYER_COUNTS, HIDDEN_LAYER_SIZES):
        cmd = build_command(layer_count, layer_size)
        print(
            f"Launching layers={layer_count}, size={layer_size}: {' '.join(cmd)}",
            flush=True,
        )
        proc = subprocess.Popen(cmd, cwd=repo_root)
        processes.append(((layer_count, layer_size), proc))

    failed = False
    try:
        for (layer_count, layer_size), proc in processes:
            ret = proc.wait()
            if ret != 0:
                failed = True
                print(
                    f"Run failed for layers={layer_count}, size={layer_size} with exit code {ret}",
                    file=sys.stderr,
                    flush=True,
                )
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, terminating all child processes...", flush=True)
        for _, proc in processes:
            proc.terminate()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
