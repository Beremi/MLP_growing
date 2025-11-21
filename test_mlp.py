#!/usr/bin/env python3
import argparse
import math
import os
from typing import Sequence, Tuple, Dict, Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


class MLP(nn.Module):
    """
    Simple fully-connected MLP with configurable activations.

    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    hidden_sizes : Sequence[int]
        Sizes of hidden layers.
    output_dim : int
        Dimension of output vector.
    activation : str
        Non-linearity to use between hidden layers ('relu' or 'tanh').
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        output_dim: int,
        activation: str = "tanh",
    ):
        super().__init__()
        activation_lower = activation.lower()
        if activation_lower not in {"relu", "tanh"}:
            raise ValueError("activation must be either 'relu' or 'tanh'.")
        self.activation_name = activation_lower

        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            if activation_lower == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

    def _init_weights(self) -> None:
        """
        Initialize all linear layers with Xavier uniform weights and zero bias.
        """
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def reinitialize(self) -> None:
        """
        Reinitialize network weights (Xavier uniform) to a fresh random state.
        """
        self._init_weights()


# def log_posterior_unnorm_numpy(
#     par: np.ndarray,
#     obs: np.ndarray,
#     y_obs: np.ndarray,
#     sigma_prior: float,
#     sigma_lik: float,
# ) -> np.ndarray:
#     """
#     Unnormalised Gaussian log posterior used for DA-MH quality metric.

#     Assumes:
#     - Prior: par ~ N(0, sigma_prior^2 I)
#     - Likelihood: obs | par ~ N(y_obs, sigma_lik^2 I) with independent components.

#     Adjust this if your actual posterior is different.
#     """
#     par = np.asarray(par)
#     obs = np.asarray(obs)
#     y_obs = np.asarray(y_obs)

#     # Shape checks (not strict; broadcast y_obs over obs samples)
#     assert par.ndim == 2
#     assert obs.ndim == 2
#     assert y_obs.ndim == 1
#     assert par.shape[0] == obs.shape[0]
#     assert obs.shape[1] == y_obs.shape[0]

#     prior_term = -0.5 * np.sum(par ** 2, axis=1) / (sigma_prior ** 2)
#     diff = obs - y_obs[None, :]
#     lik_term = -0.5 * np.sum(diff ** 2, axis=1) / (sigma_lik ** 2)
#     return prior_term + lik_term


def log_posterior_unnorm_numpy(
    par: np.ndarray,
    obs: np.ndarray,
    y_obs: np.ndarray,
    sigma_prior: float = 1.0,
    sigma_lik: float = 1.0,
    batch_growth: float | None = None,
) -> np.ndarray:
    """
    Independent standard-normal prior on 'par' (each dimension),
    independent Gaussian likelihood on 'obs' vs y_obs with sigma_lik.

    Returns unnormalized log posterior for each snapshot i:
      logpi[i] = -0.5 * ||par[i]||^2 / sigma_prior^2
                 -0.5 * ||obs[i] - y_obs||^2 / sigma_lik^2
    (constants dropped; fine for MH ratios)
    """
    par = np.asarray(par, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    if obs.shape[1] != y_obs.shape[0]:
        raise ValueError(f"y_obs has length {y_obs.shape[0]}, expected {obs.shape[1]}")

    lp_prior = -0.5 * np.sum((par / sigma_prior) ** 2, axis=1)
    resid = obs - y_obs[None, :]
    lp_lik = -0.5 * np.sum((resid / sigma_lik) ** 2, axis=1)
    return lp_prior + lp_lik


def _logpi_from_preds(
    par_batch: torch.Tensor,
    obs_pred: torch.Tensor,
    y_obs_tensor: torch.Tensor,
    sigma_prior: float,
    sigma_lik: float,
) -> torch.Tensor:
    """
    Torch helper mirroring log_posterior_unnorm_numpy for use during training.
    """
    prior_term = -0.5 * torch.sum((par_batch / sigma_prior) ** 2, dim=1)
    resid = (obs_pred - y_obs_tensor) / sigma_lik
    lik_term = -0.5 * torch.sum(resid ** 2, dim=1)
    return prior_term + lik_term


def _ensure_grad_contiguous(model: nn.Module) -> None:
    """Make any existing gradients contiguous to satisfy optimizers like LBFGS."""
    for param in model.parameters():
        if param.grad is not None and not param.grad.is_contiguous():
            param.grad = param.grad.contiguous()


def standardize_features(
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to zero mean / unit variance per dimension.

    Returns standardized X, mean, std (with std clipped to minimum 1e-8).
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    X_std = (X - mean) / std
    return X_std, mean, std


def apply_standardization(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply precomputed standardization parameters.
    """
    return (X - mean) / std


def unique_preserve_order(idx: np.ndarray) -> np.ndarray:
    """
    Return unique values from idx while preserving the first occurrence order.
    """
    idx = np.asarray(idx)
    unique_vals, first_pos = np.unique(idx, return_index=True)
    order = np.argsort(first_pos)
    return unique_vals[order]


def collect_training_indices(chain: np.ndarray, props: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Use the first n_steps entries from chain/props and return unique indices for training.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if n_steps > chain.shape[0]:
        raise ValueError("Requested n_steps exceeds available chain length.")
    idx_chain = chain[:n_steps]
    idx_props = props[:n_steps]
    combined = np.concatenate([idx_chain, idx_props])
    return unique_preserve_order(combined)


def train_mlp(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    max_adam_epochs: int = 200,
    adam_lr: float = 1e-3,
    adam_patience: int = 30,
    tol: float = 1e-5,
    max_lbfgs_iter: int = 50,
    loss_name: str = "l1",
    train_loops: int = 1,
    batch_size: int | None = None,
    loss_domain: str = "obs",
    par_train_raw: np.ndarray | None = None,
    logpi_targets: np.ndarray | None = None,
    y_obs: np.ndarray | None = None,
    sigma_prior: float = 1.0,
    sigma_lik: float = 1.0,
    batch_growth: float | None = None,
    verbose: int = 1,
    loop_improvement_pct: float = 1.0,
    fine_tune: bool = False,
    fine_tune_adam_lr: float = 1e-4,
    fine_tune_adam_epochs: int = 200,
    fine_tune_lbfgs_steps: int = 20,
    fine_tune_loops: int = 2,
) -> float:
    """Train the MLP using Adam + LBFGS with optional log-posterior loss."""
    model.to(device)
    model.train()

    def log_msg(msg: str, level: int = 2) -> None:
        if verbose >= level:
            print(msg)

    def log_loop_summary(loop_idx: int, loss_value: float, next_lr: float, reason: str) -> None:
        if verbose >= 1:
            print(
                f"[train][loop {loop_idx + 1}/{train_loops}] loss={loss_value:.6e} "
                f"next_lr={next_lr:.3e} stop={reason}"
            )

    if fine_tune:
        max_adam_epochs = fine_tune_adam_epochs
        adam_lr = fine_tune_adam_lr
        max_lbfgs_iter = fine_tune_lbfgs_steps
        train_loops = fine_tune_loops

    X_tensor = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_tensor = torch.from_numpy(y_train.astype(np.float32)).to(device)
    par_tensor_raw = (
        torch.from_numpy(par_train_raw.astype(np.float32)).to(device)
        if par_train_raw is not None
        else None
    )
    logpi_tensor = (
        torch.from_numpy(logpi_targets.astype(np.float32)).to(device)
        if logpi_targets is not None
        else None
    )
    y_obs_tensor = (
        torch.from_numpy(y_obs.astype(np.float32)).to(device)
        if y_obs is not None
        else None
    )

    n_samples = X_tensor.shape[0]
    if n_samples == 0:
        raise ValueError("Training set is empty.")

    if batch_size is None or batch_size <= 0 or batch_size > n_samples:
        effective_batch = n_samples
    else:
        effective_batch = batch_size

    loss_name_lower = loss_name.lower()
    if loss_name_lower not in {"l1", "mse", "mixed"}:
        raise ValueError(
            f"Unsupported loss_name '{loss_name}'. Expected 'l1', 'mse', or 'mixed'."
        )

    def make_criterion(name: str) -> nn.Module:
        return nn.L1Loss() if name == "l1" else nn.MSELoss()

    loss_domain_lower = loss_domain.lower()
    if loss_domain_lower not in {"obs", "logpi"}:
        raise ValueError("loss_domain must be 'obs' or 'logpi'.")
    use_logpi_loss = loss_domain_lower == "logpi"
    if use_logpi_loss and (
        par_tensor_raw is None or logpi_tensor is None or y_obs_tensor is None
    ):
        raise ValueError(
            "logpi loss requires par_train_raw, logpi_targets, and y_obs inputs."
        )

    if train_loops <= 0:
        raise ValueError("train_loops must be a positive integer.")

    best_loss = float("inf")
    current_adam_lr = adam_lr
    indices = torch.arange(n_samples, dtype=torch.long)
    growth_factor = (
        None if batch_growth is None or batch_growth <= 1.0 else float(batch_growth)
    )
    prev_loop_loss: float | None = None

    for loop_idx in range(train_loops):
        loop_stop_reason = "completed"
        stop_after_summary = False
        loop_str = f"[train][loop {loop_idx + 1}/{train_loops}]"
        if loss_name_lower == "mixed":
            current_loss_name = "mse" if loop_idx % 2 == 0 else "l1"
        else:
            current_loss_name = loss_name_lower
        criterion = make_criterion(current_loss_name)
        optimizer_adam = torch.optim.Adam(model.parameters(), lr=current_adam_lr)
        no_improve = 0
        if growth_factor is not None:
            batch_size_loop = int(round(effective_batch * (growth_factor ** loop_idx)))
            batch_size_loop = max(1, min(batch_size_loop, n_samples))
        else:
            batch_size_loop = effective_batch
        log_msg(
            f"{loop_str} Starting Adam: epochs={max_adam_epochs}, loss={current_loss_name.upper()}, "
            f"lr={optimizer_adam.param_groups[0]['lr']:.3e}, batch={batch_size_loop}, domain={loss_domain_lower}",
            level=2,
        )
        for epoch in range(1, max_adam_epochs + 1):
            loader = DataLoader(indices, batch_size=batch_size_loop, shuffle=True)
            for batch_idx in loader:
                batch_idx = batch_idx.to(device)
                batch_X = X_tensor[batch_idx]
                optimizer_adam.zero_grad()
                preds = model(batch_X)

                if use_logpi_loss:
                    batch_par = par_tensor_raw[batch_idx]
                    batch_logpi = logpi_tensor[batch_idx]
                    logpi_pred = _logpi_from_preds(
                        batch_par, preds, y_obs_tensor, sigma_prior, sigma_lik
                    )
                    loss = criterion(logpi_pred, batch_logpi)
                else:
                    batch_y = y_tensor[batch_idx]
                    loss = criterion(preds, batch_y)

                loss.backward()
                _ensure_grad_contiguous(model)
                optimizer_adam.step()

            with torch.no_grad():
                full_preds = model(X_tensor)
                if use_logpi_loss:
                    logpi_pred_full = _logpi_from_preds(
                        par_tensor_raw, full_preds, y_obs_tensor, sigma_prior, sigma_lik
                    )
                    full_loss = criterion(logpi_pred_full, logpi_tensor)
                else:
                    full_loss = criterion(full_preds, y_tensor)
                loss_val = float(full_loss.item())

            if verbose >= 2 and (epoch == 1 or epoch % 10 == 0):
                current_lr = optimizer_adam.param_groups[0]["lr"]
                log_msg(
                    f"{loop_str}[Adam] epoch {
                        epoch:4d} | loss({current_loss_name}) = {
                        loss_val:.6e} | lr = {
                        current_lr:.3e}",
                    level=2,
                )

            if best_loss == float("inf") or loss_val < best_loss - tol * (abs(best_loss) + 1e-12):
                best_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= adam_patience:
                    loop_stop_reason = "adam_plateau"
                    current_adam_lr *= 0.25
                    log_msg(
                        f"{loop_str}[Adam] plateau at epoch {epoch}, reducing LR to {
                            current_adam_lr:.3e} and stopping early.",
                        level=2,
                    )
                    if current_adam_lr < 1e-6:
                        log_msg(
                            f"LR {current_adam_lr:.3e} below 1e-6 threshold; terminating training early.",
                            level=2,
                        )
                        summary_loss = best_loss if best_loss != float("inf") else loss_val
                        log_loop_summary(loop_idx, summary_loss, current_adam_lr, "adam_lr_min")
                        return best_loss if best_loss != float("inf") else loss_val
                    break

        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=1,
            history_size=100,
            line_search_fn="strong_wolfe",
        )
        log_msg(
            f"{loop_str} Starting LBFGS: max_iter={max_lbfgs_iter}, loss={current_loss_name.upper()}, "
            f"lr={optimizer_lbfgs.param_groups[0]['lr']:.3e}",
            level=2,
        )
        no_improve = 0
        for it in range(1, max_lbfgs_iter + 1):

            def closure():
                optimizer_lbfgs.zero_grad()
                preds = model(X_tensor)
                if use_logpi_loss:
                    logpi_pred = _logpi_from_preds(
                        par_tensor_raw, preds, y_obs_tensor, sigma_prior, sigma_lik
                    )
                    loss_closure = criterion(logpi_pred, logpi_tensor)
                else:
                    loss_closure = criterion(preds, y_tensor)
                loss_closure.backward()
                _ensure_grad_contiguous(model)
                return loss_closure

            loss = optimizer_lbfgs.step(closure)
            loss_val = float(loss.item())
            if verbose >= 2 and (it == 1 or it % 5 == 0):
                current_lr = optimizer_lbfgs.param_groups[0]["lr"]
                log_msg(
                    f"{loop_str}[LBFGS] iter {
                        it:3d} | loss({current_loss_name}) = {
                        loss_val:.6e} | lr = {
                        current_lr:.3e}",
                    level=2,
                )

            if loss_val < best_loss - tol * (abs(best_loss) + 1e-12):
                best_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= adam_patience:
                    loop_stop_reason = "lbfgs_plateau"
                    log_msg(f"{loop_str}[LBFGS] plateau detected at iter {it}, stopping early.", level=2)
                    break

        loop_loss = best_loss if best_loss != float("inf") else loss_val
        if prev_loop_loss is None:
            prev_loop_loss = loop_loss
        else:
            required_improvement = abs(prev_loop_loss) * (loop_improvement_pct / 100.0)
            actual_improvement = prev_loop_loss - loop_loss
            if actual_improvement < required_improvement:
                loop_stop_reason = "improvement_stop"
                stop_after_summary = True
            else:
                prev_loop_loss = loop_loss

        log_loop_summary(loop_idx, loop_loss, current_adam_lr, loop_stop_reason)

        if stop_after_summary:
            break

    final_loss = best_loss
    summary_label = "MIXED" if loss_name_lower == "mixed" else loss_name_lower.upper()
    log_msg(
        f"[train] Finished training with final train loss({summary_label}) = {final_loss:.6e}",
        level=2,
    )
    return final_loss


def evaluate_da_metrics(
    model: nn.Module,
    par: np.ndarray,
    obs: np.ndarray,
    logpi_true: np.ndarray,
    y_obs: np.ndarray,
    chain: np.ndarray,
    props: np.ndarray,
    val_start: int,
    val_len: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    sigma_prior: float,
    sigma_lik: float,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate DA-MH quality metrics on a validation window of the chain.

    Uses pairs (chain[k], props[k]) for k in [val_start, val_start+val_len).

    Returns a dict with:
    - val_mse_obs: MSE between predicted and true obs (current states)
    - mean_da_reject: mean a1 * (1 - a2)
    - mean_da_accept: mean a1 * a2
    - mean_a1, mean_a2_reject: averages of stage-1 acceptance and stage-2 rejection
    """
    model.eval()

    idx_curr = chain[val_start: val_start + val_len]
    idx_prop = props[val_start: val_start + val_len]

    par_curr = par[idx_curr]
    par_prop = par[idx_prop]
    obs_curr_true = obs[idx_curr]
    obs_prop_true = obs[idx_prop]

    # Standardize inputs
    X_curr = apply_standardization(par_curr, x_mean, x_std)
    X_prop = apply_standardization(par_prop, x_mean, x_std)

    with torch.no_grad():
        X_curr_t = torch.from_numpy(X_curr.astype(np.float32)).to(device)
        X_prop_t = torch.from_numpy(X_prop.astype(np.float32)).to(device)

        obs_curr_pred = model(X_curr_t).cpu().numpy()
        obs_prop_pred = model(X_prop_t).cpu().numpy()

    # Validation MSE on obs (current states)
    val_mse_obs = float(np.mean((obs_curr_pred - obs_curr_true) ** 2))

    # Surrogate log posterior for current / proposed
    tilde_logpi_curr = log_posterior_unnorm_numpy(
        par_curr, obs_curr_pred, y_obs, sigma_prior, sigma_lik
    )
    tilde_logpi_prop = log_posterior_unnorm_numpy(
        par_prop, obs_prop_pred, y_obs, sigma_prior, sigma_lik
    )

    # True log posterior (precomputed)
    logpi_curr = logpi_true[idx_curr]
    logpi_prop = logpi_true[idx_prop]

    # DA-MH probabilities
    delta_tilde = tilde_logpi_prop - tilde_logpi_curr
    delta_true = logpi_prop - logpi_curr
    delta_stage2 = delta_true - delta_tilde

    # Stage 1 acceptance (using surrogate)
    a1 = np.where(delta_tilde >= 0.0, 1.0, np.exp(delta_tilde))

    # Stage 2 acceptance (correction using true vs surrogate)
    a2 = np.where(delta_stage2 >= 0.0, 1.0, np.exp(delta_stage2))
    a2_reject = 1.0 - a2

    da_reject = a1 * a2_reject  # probability: cheap accept but expensive reject
    da_accept = a1 * a2         # full DA acceptance probability

    metrics = {
        "val_mse_obs": val_mse_obs,
        "mean_da_reject": float(np.mean(da_reject)),
        "mean_da_accept": float(np.mean(da_accept)),
        "mean_a1": float(np.mean(a1)),
        "mean_a2_reject": float(np.mean(a2_reject)),
    }

    print(
        "[eval] val_start=%d | val_mse_obs=%.6e | mean_da_reject=%.6e | mean_da_accept=%.6e"
        % (
            val_start,
            metrics["val_mse_obs"],
            metrics["mean_da_reject"],
            metrics["mean_da_accept"],
        )
    )

    return metrics


def compute_logpi_l1_error(
    model: nn.Module,
    par: np.ndarray,
    logpi_true: np.ndarray,
    chain: np.ndarray,
    props: np.ndarray,
    val_start: int,
    val_len: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_obs: np.ndarray,
    sigma_prior: float,
    sigma_lik: float,
    device: torch.device,
) -> float:
    """
    Mean absolute error between true and surrogate log posterior on unique validation samples.
    """
    model.eval()
    idx_curr = chain[val_start: val_start + val_len]
    idx_prop = props[val_start: val_start + val_len]
    unique_idx = unique_preserve_order(np.concatenate([idx_curr, idx_prop]))

    par_unique = par[unique_idx]
    X_unique = apply_standardization(par_unique, x_mean, x_std)

    with torch.no_grad():
        X_t = torch.from_numpy(X_unique.astype(np.float32)).to(device)
        obs_pred = model(X_t).cpu().numpy()

    logpi_pred = log_posterior_unnorm_numpy(
        par_unique, obs_pred, y_obs, sigma_prior, sigma_lik
    )
    logpi_true_subset = logpi_true[unique_idx]
    return float(np.mean(np.abs(logpi_true_subset - logpi_pred)))


def load_data(path: str, sigma_prior: float, sigma_lik: float):
    """
    Load data from HDF5 file.

    Expects datasets:
    - 'par'   : (N, d_par)
    - 'obs'   : (N, d_obs)
    - 'y_obs' : (d_obs,)
    - 'chain' : (T,)
    - 'props' : (T,)
    - optionally 'logpi' : (N,)

    If 'logpi' is missing, it is computed using log_posterior_unnorm_numpy.
    """
    with h5py.File(path, "r") as f:
        par = f["par"][:]
        obs = f["obs"][:]
        y_obs = f["y_obs"][:]
        chain = f["chain"][:]
        props = f["props"][:]

        if "logpi" in f:
            logpi_true = f["logpi"][:]
            print("[data] Loaded 'logpi' from file.")
        else:
            print("[data] 'logpi' not found, computing from par/obs/y_obs ...")
            logpi_true = log_posterior_unnorm_numpy(par, obs, y_obs, sigma_prior, sigma_lik)

    print("[data] par shape   :", par.shape)
    print("[data] obs shape   :", obs.shape)
    print("[data] y_obs shape :", y_obs.shape)
    print("[data] chain shape :", chain.shape)
    print("[data] props shape :", props.shape)

    return par, obs, y_obs, chain, props, logpi_true


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Experiment: DA-MH efficiency vs. training data size for a given MLP architecture.\n"
            "The MLP approximates obs = f(par); DA metrics are computed on a held-out window of the chain."
        )
    )
    parser.add_argument("--data", type=str, default="data.h5", help="Path to HDF5 file with datasets.")
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        required=True,
        help="Hidden layer sizes, e.g. --hidden-sizes 64 64 64 for a 3-layer MLP.",
    )
    parser.add_argument("--sigma-prior", type=float, default=1.0, help="Prior std (for log posterior).")
    parser.add_argument("--sigma-lik", type=float, default=0.3, help="Likelihood std (for log posterior).")
    parser.add_argument(
        "--train-cutoff",
        type=int,
        default=19500,
        help="Maximum number of chain steps allowed for training (bounded by total length).",
    )
    parser.add_argument(
        "--train-step",
        type=int,
        default=500,
        help="Step for increasing train size (and default validation window length).")
    parser.add_argument(
        "--progress-train-step",
        action="store_true",
        help="When set, increase train size multiplicatively using --train-step-multiplier.",
    )
    parser.add_argument(
        "--train-step-multiplier",
        type=float,
        default=1.0,
        help="Multiplier used for progressive train sizes when --progress-train-step is enabled.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="Validation window length; defaults to --train-step when omitted.",
    )
    parser.add_argument("--adam-epochs", type=int, default=200, help="Max Adam epochs.")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--adam-patience", type=int, default=30, help="Patience for plateau detection.")
    parser.add_argument("--tol", type=float, default=1e-5, help="Relative improvement tolerance.")
    parser.add_argument("--lbfgs-steps", type=int, default=50, help="Max LBFGS steps.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save CSV results.")
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize inputs before training (default: use raw parameters).",
    )
    parser.add_argument(
        "--train-loss",
        choices=["l1", "mse", "mixed"],
        default="l1",
        help="Training loss to optimize (default: l1). Use 'mixed' to alternate MSE/L1 per loop.",
    )
    parser.add_argument(
        "--loss-domain",
        choices=["obs", "logpi"],
        default="obs",
        help="Optimize loss in observation space ('obs') or log posterior ('logpi').",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "tanh"],
        default="tanh",
        help="Hidden-layer activation (default: tanh).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size for Adam (default: full batch).",
    )
    parser.add_argument(
        "--batch-growth",
        type=float,
        default=None,
        help="If >1, multiply Adam batch size by this factor on each train loop.",
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=None,
        help="Optional limit on how many initial chain steps are ever used for training; "
             "validation always starts immediately after this window when provided.",
    )
    parser.add_argument(
        "--train-loops",
        type=int,
        default=1,
        help="Number of outer training loops (Adam + LBFGS repetitions).",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Reinitialize the MLP from scratch for every train size increment.",
    )
    parser.add_argument(
        "--reinit-before-train",
        action="store_true",
        help="Reinitialize model weights before each training size update (after potential architecture growth).",
    )
    parser.add_argument(
        "--loop-improvement-pct",
        type=float,
        default=1.0,
        help="Minimum percentage improvement required between training loops; stops early if not met.",
    )
    args = parser.parse_args()

    # Determine validation size (defaults to train step when not provided)
    val_size = args.train_step if args.val_size is None else args.val_size
    if val_size <= 0:
        raise ValueError("val_size must be positive.")

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Using device: {device}")
    print(f"[setup] Hidden sizes: {args.hidden_sizes}")

    # Load data
    par, obs, y_obs, chain, props, logpi_true = load_data(
        args.data, args.sigma_prior, args.sigma_lik
    )

    n_chain = chain.shape[0]
    if args.train_window is not None:
        train_window = args.train_window
        if train_window <= 0:
            raise ValueError("train_window must be positive.")
        if train_window + val_size > n_chain:
            raise ValueError("train_window + val_size exceeds chain length.")
        val_start = train_window
        max_train_allowed = train_window
    else:
        if val_size >= n_chain:
            raise ValueError("Validation size must be smaller than the chain length.")
        val_start = n_chain - val_size
        max_train_allowed = val_start
    if max_train_allowed <= 0:
        raise ValueError("Not enough samples remain for training once validation is reserved.")
    max_train = min(args.train_cutoff, max_train_allowed)

    if args.train_step <= 0:
        raise ValueError("train_step must be positive.")

    if args.progress_train_step:
        if args.train_step_multiplier <= 1.0:
            raise ValueError("train-step-multiplier must be > 1.0 when progressive stepping is enabled.")
        train_sizes = []
        current = args.train_step
        while current <= max_train:
            train_sizes.append(int(current))
            current = max(
                int(math.ceil(current * args.train_step_multiplier)),
                train_sizes[-1] + 1,
            )
    else:
        train_sizes = list(range(args.train_step, max_train + 1, args.train_step))
    print(f"[setup] Train sizes to test: {train_sizes}")
    if args.train_window is not None:
        print(
            f"[setup] Validation window starts after train window {val_start} "
            f"with length {val_size}."
        )
    else:
        print(f"[setup] Validation window fixed to final {val_size} steps (start index {val_start}).")

    input_dim = par.shape[1]
    output_dim = obs.shape[1]

    def init_model() -> nn.Module:
        m = MLP(
            input_dim=input_dim,
            hidden_sizes=args.hidden_sizes,
            output_dim=output_dim,
            activation=args.activation,
        )
        return m

    model = init_model()
    print(model)
    records = []
    depth = len(args.hidden_sizes)
    layer_str = "x".join(str(h) for h in args.hidden_sizes)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"da_mlp_depth{depth}_layers{layer_str}.csv",
    )

    # Loop over train sizes
    for n_train in train_sizes:
        print("=" * 80)
        train_idx = collect_training_indices(chain, props, n_train)
        print(
            "[run] Using %d unique samples (chain + proposals) from first %d chain steps."
            % (train_idx.size, n_train)
        )
        print(f"[train] Training samples range: [0, {n_train}) with {train_idx.size} unique inputs.")

        X_train_raw = par[train_idx]
        y_train = obs[train_idx]
        logpi_train = logpi_true[train_idx]

        # Optionally standardize inputs on training set
        if args.standardize:
            X_train_std, x_mean, x_std = standardize_features(X_train_raw)
        else:
            X_train_std = X_train_raw
            x_mean = np.zeros(X_train_raw.shape[1], dtype=X_train_raw.dtype)
            x_std = np.ones(X_train_raw.shape[1], dtype=X_train_raw.dtype)

        # Reinitialize model if requested, otherwise continue training the existing one
        if args.retrain:
            model = init_model()
        elif args.reinit_before_train:
            model.reinitialize()

        final_train_loss = train_mlp(
            model,
            X_train_std,
            y_train,
            device=device,
            max_adam_epochs=args.adam_epochs,
            adam_lr=args.adam_lr,
            adam_patience=args.adam_patience,
            tol=args.tol,
            max_lbfgs_iter=args.lbfgs_steps,
            loss_name=args.train_loss,
            train_loops=args.train_loops,
            batch_size=args.batch_size,
            loss_domain=args.loss_domain,
            par_train_raw=X_train_raw,
            logpi_targets=logpi_train,
            y_obs=y_obs,
            sigma_prior=args.sigma_prior,
            sigma_lik=args.sigma_lik,
            batch_growth=args.batch_growth,
            loop_improvement_pct=args.loop_improvement_pct,
        )

        # Evaluate on the held-out tail of the chain
        metrics = evaluate_da_metrics(
            model,
            par,
            obs,
            logpi_true,
            y_obs,
            chain,
            props,
            val_start=val_start,
            val_len=val_size,
            x_mean=x_mean,
            x_std=x_std,
            sigma_prior=args.sigma_prior,
            sigma_lik=args.sigma_lik,
            device=device,
        )
        logpi_l1_error = compute_logpi_l1_error(
            model,
            par,
            logpi_true,
            chain,
            props,
            val_start=val_start,
            val_len=val_size,
            x_mean=x_mean,
            x_std=x_std,
            y_obs=y_obs,
            sigma_prior=args.sigma_prior,
            sigma_lik=args.sigma_lik,
            device=device,
        )

        record = {
            "train_size": n_train,
            "unique_train_size": int(train_idx.size),
            "val_start": val_start,
            "val_size": val_size,
            "train_step": args.train_step,
            "train_loss": args.train_loss,
            "final_train_loss": final_train_loss,
            "val_logpi_l1_error": logpi_l1_error,
        }
        record.update(metrics)
        records.append(record)

        # Persist intermediate results so partial progress is saved if interrupted
        df_partial = pd.DataFrame(records)
        df_partial.to_csv(out_path, index=False)
        print(f"[save] Wrote {len(df_partial)} records to {out_path}")

    # Final confirmation message (file already written incrementally)
    print(f"[done] Results saved to: {out_path}")


if __name__ == "__main__":
    main()
