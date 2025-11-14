#!/usr/bin/env python3
import argparse
import os
from typing import Sequence, Tuple, Dict, Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn


class MLP(nn.Module):
    """
    Simple fully-connected MLP with ReLU activations.

    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    hidden_sizes : Sequence[int]
        Sizes of hidden layers.
    output_dim : int
        Dimension of output vector.
    """

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], output_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


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
) -> float:
    """
    Train MLP with Adam followed by LBFGS until plateau or max iterations.
    Optionally repeat the Adam+LBFGS pair multiple times (train_loops).

    Parameters
    ----------
    model : nn.Module
        MLP model to train.
    X_train : np.ndarray
        Training inputs of shape (N, d_in) (already standardized).
    y_train : np.ndarray
        Training targets of shape (N, d_out).
    device : torch.device
        Device on which to run the training.
    max_adam_epochs : int
        Maximum number of Adam epochs.
    adam_lr : float
        Learning rate for Adam optimizer.
    adam_patience : int
        Number of epochs/steps with small improvement before stopping.
    tol : float
        Relative improvement tolerance for plateau detection.
    max_lbfgs_iter : int
        Maximum number of LBFGS iterations.
    loss_name : str
        Either 'l1' or 'mse' to choose the training loss.
    train_loops : int
        Number of outer loops repeating the Adam + LBFGS sequence.

    Returns
    -------
    final_loss : float
        Final training loss after both optimizers.
    """
    model.to(device)
    model.train()

    X_tensor = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_tensor = torch.from_numpy(y_train.astype(np.float32)).to(device)

    loss_name_lower = loss_name.lower()
    if loss_name_lower == "l1":
        criterion = nn.L1Loss()
    elif loss_name_lower == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss_name '{loss_name}'. Expected 'l1' or 'mse'.")

    if train_loops <= 0:
        raise ValueError("train_loops must be a positive integer.")

    best_loss = float("inf")

    for loop_idx in range(train_loops):
        loop_str = f"[train][loop {loop_idx + 1}/{train_loops}]"

        # Phase 1: Adam
        optimizer_adam = torch.optim.Adam(model.parameters(), lr=adam_lr)
        no_improve = 0
        print(
            f"{loop_str} Starting Adam: max_epochs={max_adam_epochs}, loss={loss_name_lower.upper()}, "
            f"lr={optimizer_adam.param_groups[0]['lr']:.3e}"
        )
        for epoch in range(1, max_adam_epochs + 1):
            optimizer_adam.zero_grad()
            preds = model(X_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer_adam.step()

            loss_val = loss.item()
            if epoch == 1 or epoch % 10 == 0:
                current_lr = optimizer_adam.param_groups[0]["lr"]
                print(
                    f"{loop_str}[Adam] epoch {epoch:4d} | loss({loss_name_lower}) = {loss_val:.6e} | lr = {current_lr:.3e}"
                )

            # Plateau detection (relative improvement)
            if best_loss == float("inf") or loss_val < best_loss - tol * (abs(best_loss) + 1e-12):
                best_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= adam_patience:
                    print(f"{loop_str}[Adam] plateau detected at epoch {epoch}, stopping early.")
                    break

        # Phase 2: LBFGS (full-batch)
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=1,  # we'll loop manually
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        print(
            f"{loop_str} Starting LBFGS: max_iter={max_lbfgs_iter}, loss={loss_name_lower.upper()}, "
            f"lr={optimizer_lbfgs.param_groups[0]['lr']:.3e}"
        )
        # reset plateau counter for LBFGS phase
        no_improve = 0
        for it in range(1, max_lbfgs_iter + 1):

            def closure():
                optimizer_lbfgs.zero_grad()
                preds = model(X_tensor)
                loss = criterion(preds, y_tensor)
                loss.backward()
                return loss

            loss = optimizer_lbfgs.step(closure)
            loss_val = float(loss.item())
            if it == 1 or it % 5 == 0:
                current_lr = optimizer_lbfgs.param_groups[0]["lr"]
                print(
                    f"{loop_str}[LBFGS] iter {it:3d} | loss({loss_name_lower}) = {loss_val:.6e} | lr = {current_lr:.3e}"
                )

            if loss_val < best_loss - tol * (abs(best_loss) + 1e-12):
                best_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= adam_patience:
                    print(f"{loop_str}[LBFGS] plateau detected at iter {it}, stopping early.")
                    break

    final_loss = best_loss
    print(f"[train] Finished training with final train loss({loss_name_lower}) = {final_loss:.6e}")
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
    parser.add_argument("--max-train-size", type=int, default=19500, help="Max training size (<= len(chain)-val_size).")
    parser.add_argument("--train-step", type=int, default=500, help="Step for increasing train size (and default validation window length).")
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
        choices=["l1", "mse"],
        default="l1",
        help="Training loss to optimize (default: l1).",
    )
    parser.add_argument(
        "--train-loops",
        type=int,
        default=1,
        help="Number of outer training loops (Adam + LBFGS repetitions).",
    )
    args = parser.parse_args()

    # Determine validation size (defaults to train step when not provided)
    val_size = args.train_step if args.val_size is None else args.val_size

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
    max_train_allowed = n_chain - val_size
    if max_train_allowed <= 0:
        raise ValueError("Validation size is too large compared to chain length.")
    max_train = min(args.max_train_size, max_train_allowed)

    # train sizes: 500, 1000, ..., up to max_train
    train_sizes = list(range(args.train_step, max_train + 1, args.train_step))
    print(f"[setup] Train sizes to test: {train_sizes}")

    input_dim = par.shape[1]
    output_dim = obs.shape[1]

    model = MLP(input_dim=input_dim, hidden_sizes=args.hidden_sizes, output_dim=output_dim)
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

        X_train_raw = par[train_idx]
        y_train = obs[train_idx]

        # Optionally standardize inputs on training set
        if args.standardize:
            X_train_std, x_mean, x_std = standardize_features(X_train_raw)
        else:
            X_train_std = X_train_raw
            x_mean = np.zeros(X_train_raw.shape[1], dtype=X_train_raw.dtype)
            x_std = np.ones(X_train_raw.shape[1], dtype=X_train_raw.dtype)

        # Continue training the existing model instead of restarting each time
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
        )

        # Evaluate on the next val_size samples
        val_start = n_train
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

        record = {
            "train_size": n_train,
            "unique_train_size": int(train_idx.size),
            "val_start": val_start,
            "val_size": val_size,
            "train_step": args.train_step,
            "train_loss": args.train_loss,
            "final_train_loss": final_train_loss,
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
