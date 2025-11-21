#!/usr/bin/env python3
"""Launch fixed-rank low-rank MLP sweeps across several ranks."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Sequence

RANKS: Sequence[int] = (4, 256)
BASE_COMMAND: Sequence[str] = ("python", "rank_mlp_double.py")
RESULTS_DIR = "results_fixed_rank_lowrank"


def build_command(rank: int, results_path: Path) -> list[str]:
    return [
        *BASE_COMMAND,
        "--data",
        "data1.h5",
        "--hidden-dim",
        "1024",
        "--num-hidden-layers",
        "3",
        "--initial-rank",
        str(rank),
        "--disable-rank-growth",
        "--train-start-step",
        "2000",
        "--window-size",
        "2000",
        "--max-total-train-steps",
        "20000",
        "--results-csv",
        str(results_path),
    ]


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    results_dir = repo_root / RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[int, subprocess.Popen]] = []
    for rank in RANKS:
        results_csv = results_dir / f"rank_mlp_rank{rank}.csv"
        cmd = build_command(rank, results_csv)
        print(f"Launching rank={rank}: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, cwd=repo_root)
        processes.append((rank, proc))

    failed = False
    try:
        for rank, proc in processes:
            ret = proc.wait()
            if ret != 0:
                failed = True
                print(
                    f"Run failed for rank={rank} with exit code {ret}",
                    file=sys.stderr,
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
