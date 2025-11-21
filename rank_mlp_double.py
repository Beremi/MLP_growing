#!/usr/bin/env python3
"""Run the low-rank doubling experiment from rank_MLP_2_layers_double.ipynb via CLI."""
from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from test_mlp import (
    load_data,
    train_mlp,
    standardize_features,
    apply_standardization,
    unique_preserve_order,
    log_posterior_unnorm_numpy,
)


class LowRankLinear(nn.Module):
    """Low-rank linear layer with factored weight W = U @ V."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)

        self.U = nn.Parameter(torch.empty(self.out_features, self.rank))
        self.V = nn.Parameter(torch.empty(self.rank, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(max(1, self.in_features))
        nn.init.uniform_(self.U, -bound, bound)
        nn.init.uniform_(self.V, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> Tensor:
        return self.U @ self.V

    def forward(self, input: Tensor) -> Tensor:  # type: ignore[override]
        return F.linear(input, self.weight, self.bias)

    def _set_uv(self, U_new: Tensor, V_new: Tensor) -> None:
        if U_new.shape[0] != self.out_features or V_new.shape[1] != self.in_features:
            raise ValueError("U_new/V_new shapes do not match layer dimensions")
        if U_new.shape[1] != V_new.shape[0]:
            raise ValueError("U_new and V_new shapes are inconsistent")
        U_new = U_new.to(dtype=self.U.dtype, device=self.U.device)
        V_new = V_new.to(dtype=self.V.dtype, device=self.V.device)
        self.rank = int(U_new.shape[1])
        self.U = nn.Parameter(U_new)
        self.V = nn.Parameter(V_new)

    def expand_rank(self, new_rank: int, noise_std: float = 0.01) -> None:
        if new_rank < self.rank:
            raise ValueError("new_rank must be >= current rank")
        if new_rank == self.rank:
            return
        U_new = self.U.new_zeros(self.out_features, new_rank)
        V_new = self.V.new_zeros(new_rank, self.in_features)
        with torch.no_grad():
            U_new[:, : self.rank].copy_(self.U)
            V_new[: self.rank, :].copy_(self.V)
            if noise_std > 0:
                U_new[:, self.rank :] = torch.randn_like(U_new[:, self.rank :]) * noise_std
                V_new[self.rank :, :] = torch.randn_like(V_new[self.rank :, :]) * noise_std
        self._set_uv(U_new, V_new)

    def contract_by_ratio(self, ratio: float) -> int:
        W = self.weight.detach()
        U_svd, S, Vh = torch.linalg.svd(W, full_matrices=False)
        if S.numel() == 0:
            self.contract_to_rank(1)
            return self.rank
        s_max = S.max()
        if s_max <= 0:
            target_rank = 1
        else:
            mask = S >= ratio * s_max
            k = int(mask.sum().item())
            target_rank = k or 1
        max_rank = int(min(self.out_features, self.in_features))
        target_rank = max(1, min(target_rank, max_rank))
        self._svd_truncate_to_rank(U_svd, S, Vh, target_rank)
        return self.rank

    def contract_to_rank(self, target_rank: int) -> int:
        max_rank = int(min(self.out_features, self.in_features))
        target_rank = max(1, min(int(target_rank), max_rank))
        if target_rank == self.rank:
            return self.rank
        if target_rank > self.rank:
            self.expand_rank(target_rank)
            return self.rank
        W = self.weight.detach()
        U_svd, S, Vh = torch.linalg.svd(W, full_matrices=False)
        self._svd_truncate_to_rank(U_svd, S, Vh, target_rank)
        return self.rank

    def _svd_truncate_to_rank(self, U_svd: Tensor, S: Tensor, Vh: Tensor, k: int) -> None:
        k = max(1, min(int(k), S.numel()))
        P = U_svd[:, :k]
        Q_T = Vh[:k, :]
        s_k = S[:k]
        sqrt_s = torch.sqrt(torch.clamp(s_k, min=0.0))
        U_new = P * sqrt_s.unsqueeze(0)
        V_new = sqrt_s.unsqueeze(1) * Q_T
        self._set_uv(U_new, V_new)


class LowRankMLP(nn.Module):
    """Fully-connected MLP with low-rank factored hidden→hidden layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int,
        ranks: Sequence[int],
        activation: Optional[nn.Module] = None,
        noise_std: float = 0.01,
        apply_final_activation: bool = True,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        expected = max(num_hidden_layers - 1, 0)
        if len(ranks) != expected:
            raise ValueError(f"Expected {expected} ranks, got {len(ranks)}")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.activation = activation if activation is not None else nn.Tanh()
        self.noise_std = float(noise_std)
        self.apply_final_activation = bool(apply_final_activation)

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

        self.low_rank_layers = nn.ModuleList()
        for rank in ranks:
            self.low_rank_layers.append(
                LowRankLinear(
                    in_features=self.hidden_dim,
                    out_features=self.hidden_dim,
                    rank=int(rank),
                    bias=True,
                )
            )
        self.ranks: List[int] = [int(r) for r in ranks]

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = self.input_layer(x)
        for layer in self.low_rank_layers:
            x = self.activation(x)
            x = layer(x)
        if self.apply_final_activation:
            x = self.activation(x)
        return self.output_layer(x)

    def expand_ranks(self, new_ranks: Sequence[int], noise_std: Optional[float] = None) -> None:
        if len(new_ranks) != len(self.low_rank_layers):
            raise ValueError("Length mismatch for new_ranks")
        std = self.noise_std if noise_std is None else float(noise_std)
        for idx, (layer, new_rank) in enumerate(zip(self.low_rank_layers, new_ranks)):
            if new_rank < layer.rank:
                raise ValueError("Cannot shrink rank via expand_ranks")
            if new_rank > layer.rank:
                layer.expand_rank(int(new_rank), noise_std=std)
                self.ranks[idx] = layer.rank

    def contract_ranks_by_ratio(self, ratio: float) -> None:
        for idx, layer in enumerate(self.low_rank_layers):
            self.ranks[idx] = layer.contract_by_ratio(ratio)

    def contract_ranks_by_amount(self, target_ranks: Sequence[int]) -> None:
        if len(target_ranks) != len(self.low_rank_layers):
            raise ValueError("Length mismatch for target_ranks")
        for idx, (layer, target_rank) in enumerate(zip(self.low_rank_layers, target_ranks)):
            target_rank = max(1, int(target_rank))
            if target_rank == layer.rank:
                continue
            if target_rank > layer.rank:
                layer.expand_rank(target_rank, noise_std=self.noise_std)
            else:
                layer.contract_to_rank(target_rank)
            self.ranks[idx] = layer.rank

    def singular_values_by_layer(self, max_rank: Optional[int] = None) -> List[Tensor]:
        if not self.low_rank_layers:
            return []
        cap = None if max_rank is None else max(1, int(max_rank))
        values: List[Tensor] = []
        for layer in self.low_rank_layers:
            s = torch.linalg.svdvals(layer.weight.detach())
            keep = min(layer.rank, s.numel())
            if cap is not None:
                keep = min(keep, cap)
            values.append(s[:keep].cpu())
        return values


def activation_from_name(name: str) -> nn.Module:
    lower = name.lower()
    if lower == "relu":
        return nn.ReLU()
    if lower == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'")


def build_train_sizes(
    total_chain: int,
    train_start: int,
    window: int,
    max_total_steps: Optional[int],
    num_windows: Optional[int],
) -> list[int]:
    if train_start >= total_chain:
        raise ValueError("train-start-step must be < chain length")
    if window <= 0:
        raise ValueError("window-size must be positive")
    max_trainable = max(0, total_chain - train_start - window)
    if max_trainable < window:
        raise ValueError("Not enough space for train/validation windows")
    limit = max_trainable if max_total_steps is None else min(max_trainable, int(max_total_steps))
    sizes: list[int] = []
    current = window
    while current <= limit:
        train_end = train_start + current
        val_end = train_end + window
        if val_end > total_chain:
            break
        sizes.append(current)
        if num_windows is not None and len(sizes) >= num_windows:
            break
        current += window
    if not sizes:
        raise ValueError("Training schedule is empty; adjust window/limits")
    return sizes


def collect_training_range(chain: np.ndarray, props: np.ndarray, start: int, steps: int) -> np.ndarray:
    end = min(chain.shape[0], start + steps)
    if start >= end:
        raise ValueError("Requested training window is empty")
    idx_chain = chain[start:end]
    idx_props = props[start:end]
    combined = np.concatenate([idx_chain, idx_props])
    return unique_preserve_order(combined)


def prepare_training_arrays(
    par: np.ndarray,
    obs: np.ndarray,
    logpi_true: np.ndarray,
    indices: np.ndarray,
    use_standardization: bool,
):
    X_raw = par[indices]
    y_block = obs[indices]
    logpi_block = logpi_true[indices]
    if use_standardization:
        X_proc, mean, std = standardize_features(X_raw)
    else:
        X_proc = X_raw
        mean = np.zeros(X_raw.shape[1], dtype=X_raw.dtype)
        std = np.ones(X_raw.shape[1], dtype=X_raw.dtype)
    return X_raw, X_proc, y_block, logpi_block, mean, std


def apply_standardization_block(
    par_block: np.ndarray,
    use_standardization: bool,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return apply_standardization(par_block, mean, std) if use_standardization else par_block


def logpi_l1_error(
    model: LowRankMLP,
    par: np.ndarray,
    obs: np.ndarray,
    logpi_true: np.ndarray,
    y_obs: np.ndarray,
    chain: np.ndarray,
    props: np.ndarray,
    start: int,
    length: Optional[int],
    use_standardization: bool,
    mean: np.ndarray,
    std: np.ndarray,
    sigma_prior: float,
    sigma_lik: float,
    device: torch.device,
) -> float:
    if length is None:
        end = chain.shape[0]
    else:
        end = min(chain.shape[0], start + length)
    if start >= end:
        return float("nan")
    idx_chain = chain[start:end]
    idx_props = props[start:end]
    candidate_idx = np.concatenate([idx_chain, idx_props])
    unique_idx = unique_preserve_order(candidate_idx)
    if unique_idx.size == 0:
        return float("nan")
    par_block = par[unique_idx]
    X_block = apply_standardization_block(par_block, use_standardization, mean, std)
    with torch.no_grad():
        preds = model(torch.from_numpy(X_block.astype(np.float32)).to(device)).cpu().numpy()
    logpi_pred = log_posterior_unnorm_numpy(par_block, preds, y_obs, sigma_prior, sigma_lik)
    return float(np.mean(np.abs(logpi_true[unique_idx] - logpi_pred)))


def run_training_cycle(
    model: LowRankMLP,
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    X_train_raw: np.ndarray,
    logpi_train: np.ndarray,
    device: torch.device,
    train_cfg: argparse.Namespace,
    sigma_prior: float,
    sigma_lik: float,
    y_obs: np.ndarray,
) -> float:
    return train_mlp(
        model,
        X_train=X_train_proc.astype(np.float32),
        y_train=y_train.astype(np.float32),
        device=device,
        max_adam_epochs=train_cfg.max_adam_epochs,
        adam_lr=train_cfg.adam_lr,
        adam_patience=train_cfg.adam_patience,
        tol=train_cfg.tol,
        max_lbfgs_iter=train_cfg.max_lbfgs_iter,
        loss_name=train_cfg.loss_name,
        train_loops=train_cfg.train_loops,
        batch_size=train_cfg.batch_size,
        loss_domain=train_cfg.loss_domain,
        par_train_raw=X_train_raw,
        logpi_targets=logpi_train,
        y_obs=y_obs,
        sigma_prior=sigma_prior,
        sigma_lik=sigma_lik,
        batch_growth=train_cfg.batch_growth,
        verbose=train_cfg.verbose,
        loop_improvement_pct=train_cfg.loop_improvement_pct,
    )


def layer_singular_value_ratios(model: LowRankMLP) -> list[float]:
    ratios: list[float] = []
    for sv in model.singular_values_by_layer():
        if sv.numel() == 0:
            ratios.append(float("nan"))
            continue
        max_sv = torch.max(sv).item()
        if max_sv <= 0:
            ratios.append(float("nan"))
            continue
        min_sv = torch.min(sv).item()
        ratios.append(min_sv / max_sv)
    return ratios


def doubled_rank_targets(model: LowRankMLP) -> tuple[list[int], bool]:
    targets: list[int] = []
    changed = False
    max_rank_allowed = model.hidden_dim
    for current in model.ranks:
        new_rank = min(max_rank_allowed, max(1, int(current) * 2))
        if new_rank != int(current):
            changed = True
        targets.append(new_rank)
    return targets, changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-rank MLP doubling experiment")
    parser.add_argument("--data", default="data1.h5", help="Path to HDF5 dataset")
    parser.add_argument("--sigma-prior", type=float, default=1.0)
    parser.add_argument("--sigma-lik", type=float, default=0.3)
    parser.add_argument("--train-start-step", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--num-train-windows", type=int, default=30)
    parser.add_argument("--max-total-train-steps", type=int, default=40000)
    parser.add_argument("--master-val-start", type=int, default=50000)
    parser.add_argument("--master-val-length", type=int, default=None)
    parser.add_argument("--growth-compression-ratio", type=float, default=0.05)
    parser.add_argument("--improvement-tol", type=float, default=1e-5)
    parser.add_argument("--use-standardization", action="store_true")
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--skip-pre-contract", action="store_true", help="Skip contracting ranks before base training")
    parser.add_argument(
        "--disable-rank-growth",
        action="store_true",
        help="Keep ranks fixed (no pre-contract, expansion, or compression).",
    )
    parser.add_argument("--results-csv", default="rank_mlp_progress.csv")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=5)
    parser.add_argument("--initial-rank", type=int, default=2)
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--no-final-activation", action="store_true")

    parser.add_argument("--max-adam-epochs", type=int, default=1000)
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--adam-patience", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--max-lbfgs-iter", type=int, default=50)
    parser.add_argument("--loss-name", choices=["l1", "mse", "mixed"], default="l1")
    parser.add_argument("--train-loops", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--loss-domain", choices=["obs", "logpi"], default="obs")
    parser.add_argument("--batch-growth", type=float, default=1.2)
    parser.add_argument("--train-verbose", type=int, default=2, help="Verbosity passed to train_mlp")
    parser.add_argument("--loop-improvement-pct", type=float, default=0.1)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    par, obs, y_obs, chain, props, logpi_true = load_data(args.data, args.sigma_prior, args.sigma_lik)
    train_sizes = build_train_sizes(
        total_chain=chain.shape[0],
        train_start=args.train_start_step,
        window=args.window_size,
        max_total_steps=args.max_total_train_steps,
        num_windows=args.num_train_windows,
    )
    if args.master_val_length is None:
        master_val_length = None
    else:
        master_val_length = min(args.master_val_length, max(0, chain.shape[0] - args.master_val_start))

    ranks = [int(args.initial_rank)] * max(args.num_hidden_layers - 1, 0)

    def make_model() -> LowRankMLP:
        model = LowRankMLP(
            input_dim=par.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=obs.shape[1],
            num_hidden_layers=args.num_hidden_layers,
            ranks=ranks,
            activation=activation_from_name(args.activation),
            noise_std=args.noise_std,
            apply_final_activation=not args.no_final_activation,
        )
        return model

    train_cfg = argparse.Namespace(
        max_adam_epochs=args.max_adam_epochs,
        adam_lr=args.adam_lr,
        adam_patience=args.adam_patience,
        tol=args.tol,
        max_lbfgs_iter=args.max_lbfgs_iter,
        loss_name=args.loss_name,
        train_loops=args.train_loops,
        batch_size=args.batch_size,
        loss_domain=args.loss_domain,
        batch_growth=args.batch_growth,
        verbose=args.train_verbose,
        loop_improvement_pct=args.loop_improvement_pct,
    )

    print(f"Device: {device}")
    print(f"Train checkpoints: {train_sizes}")

    checkpoint_records: list[dict] = []
    previous_model: Optional[LowRankMLP] = None
    pre_contract = not args.skip_pre_contract and not args.disable_rank_growth

    for step_idx, train_steps in enumerate(train_sizes, start=1):
        print(f"=== Step {step_idx}: train on first {train_steps} steps (window={args.window_size}) ===")
        train_indices = collect_training_range(chain, props, args.train_start_step, train_steps)
        X_train_raw, X_train_proc, y_train, logpi_train, x_mean, x_std = prepare_training_arrays(
            par, obs, logpi_true, train_indices, args.use_standardization
        )

        if args.warm_start and previous_model is not None:
            model = deepcopy(previous_model)
        else:
            model = make_model()
        model.to(device)

        if pre_contract and model.low_rank_layers:
            model.contract_ranks_by_ratio(args.growth_compression_ratio)

        base_train_loss = run_training_cycle(
            model,
            X_train_proc,
            y_train,
            X_train_raw,
            logpi_train,
            device,
            train_cfg,
            args.sigma_prior,
            args.sigma_lik,
            y_obs,
        )
        val_start = args.train_start_step + train_steps
        val_error = logpi_l1_error(
            model,
            par,
            obs,
            logpi_true,
            y_obs,
            chain,
            props,
            val_start,
            args.window_size,
            args.use_standardization,
            x_mean,
            x_std,
            args.sigma_prior,
            args.sigma_lik,
            device,
        )
        master_error = logpi_l1_error(
            model,
            par,
            obs,
            logpi_true,
            y_obs,
            chain,
            props,
            args.master_val_start,
            master_val_length,
            args.use_standardization,
            x_mean,
            x_std,
            args.sigma_prior,
            args.sigma_lik,
            device,
        )
        base_ranks = tuple(int(r) for r in getattr(model, "ranks", []))

        best_model = deepcopy(model)
        best_error = val_error
        best_master_error = master_error
        best_train_loss = base_train_loss

        growth_trials: list[dict] = []
        if not args.disable_rank_growth:
            trial_idx = 0

            while True:
                ratios = layer_singular_value_ratios(best_model)
                target_ranks, changed = doubled_rank_targets(best_model)
                if not changed:
                    print("  All low-rank layers already at maximum rank; stopping growth loop.")
                    break

                candidate = deepcopy(best_model)
                ranks_before = tuple(int(r) for r in candidate.ranks)
                candidate.contract_ranks_by_amount(target_ranks)
                ranks_after_expand = tuple(int(r) for r in candidate.ranks)
                trial_idx += 1

                ratio_str = (
                    ", ".join("nan" if not np.isfinite(r) else f"{r:.3e}" for r in ratios)
                    if ratios
                    else "(no ratios)"
                )
                print(
                    f"  [growth][train={train_steps}] trial {trial_idx}: doubling all layers {ranks_before} -> {ranks_after_expand}"
                )
                if ratios:
                    print(f"      singular ratios per layer: [{ratio_str}]")

                expand_loss = run_training_cycle(
                    candidate,
                    X_train_proc,
                    y_train,
                    X_train_raw,
                    logpi_train,
                    device,
                    train_cfg,
                    args.sigma_prior,
                    args.sigma_lik,
                    y_obs,
                )

                candidate.contract_ranks_by_ratio(args.growth_compression_ratio)
                ranks_after_compress = tuple(int(r) for r in candidate.ranks)

                compress_loss = run_training_cycle(
                    candidate,
                    X_train_proc,
                    y_train,
                    X_train_raw,
                    logpi_train,
                    device,
                    train_cfg,
                    args.sigma_prior,
                    args.sigma_lik,
                    y_obs,
                )

                candidate_val_error = logpi_l1_error(
                    candidate,
                    par,
                    obs,
                    logpi_true,
                    y_obs,
                    chain,
                    props,
                    val_start,
                    args.window_size,
                    args.use_standardization,
                    x_mean,
                    x_std,
                    args.sigma_prior,
                    args.sigma_lik,
                    device,
                )
                candidate_master_error = logpi_l1_error(
                    candidate,
                    par,
                    obs,
                    logpi_true,
                    y_obs,
                    chain,
                    props,
                    args.master_val_start,
                    master_val_length,
                    args.use_standardization,
                    x_mean,
                    x_std,
                    args.sigma_prior,
                    args.sigma_lik,
                    device,
                )
                improved = candidate_val_error < best_error - args.improvement_tol

                growth_trials.append(
                    {
                        "trial": trial_idx,
                        "strategy": "double_all",
                        "sv_ratios": [float(r) for r in ratios],
                        "ranks_before": ranks_before,
                        "ranks_target": tuple(target_ranks),
                        "ranks_after_expand": ranks_after_expand,
                        "ranks_after_compress": ranks_after_compress,
                        "expand_loss": float(expand_loss),
                        "compress_loss": float(compress_loss),
                        "prev_best_val": float(best_error),
                        "candidate_val": float(candidate_val_error),
                        "improved": bool(improved),
                    }
                )

                print(
                    f"      ranks: before {ranks_before} | expand {ranks_after_expand} | compress {ranks_after_compress}"
                )
                print(
                    f"      val logpi: {best_error:.4e} -> {candidate_val_error:.4e} | improved={improved}"
                )

                if improved:
                    best_model = deepcopy(candidate)
                    best_error = candidate_val_error
                    best_master_error = candidate_master_error
                    best_train_loss = compress_loss
                else:
                    print("      Improvement threshold not met; ending growth loop.")
                    break
        else:
            print("  Rank growth disabled; keeping ranks fixed through all windows.")

        final_model = best_model
        if args.warm_start:
            previous_model = deepcopy(final_model)
        else:
            previous_model = None

        record = {
            "train_steps": int(train_steps),
            "unique_train_samples": int(train_indices.size),
            "val_window_start": val_start,
            "val_window_length": args.window_size,
            "base_train_loss": float(base_train_loss),
            "final_train_loss": float(best_train_loss),
            "base_val_logpi_l1": float(val_error),
            "final_val_logpi_l1": float(best_error),
            "master_logpi_l1": float(best_master_error),
            "base_ranks": base_ranks,
            "final_ranks": tuple(int(r) for r in final_model.ranks),
            "num_growth_trials": len(growth_trials),
            "growth_trials": growth_trials,
        }
        checkpoint_records.append(record)

        if args.results_csv:
            out_path = Path(args.results_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(checkpoint_records).to_csv(out_path, index=False)
            print(f"  Saved progress to {out_path}")

    print("Completed steps:", len(checkpoint_records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
