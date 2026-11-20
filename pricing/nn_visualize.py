from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq

from pricing.analytical import reiner_rubinstein_dip
from pricing.neural_network import (
    CHECKPOINT_DIR,
    FEATURES,
    FIGURE_DIR,
    TABLE_DIR,
    TARGET,
    load_checkpoint,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot NN pricing figures.")
    parser.add_argument("--num-points", type=int, default=20, help="Test points to plot (default: 20).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--barrier", type=float, default=85.0)
    parser.add_argument("--volatility", type=float, default=0.30, help="True sigma used for the surface.")
    parser.add_argument("--rate", type=float, default=0.03)
    return parser.parse_args()


def nn_price(model, scaler, features: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(scaler.transform(features.astype(np.float32)))
    with torch.no_grad():
        return model(x).numpy().astype(float)


# ---------------------------------------------------------------------------
# 1. Predicted vs analytical on a few test contracts
# ---------------------------------------------------------------------------

def plot_sampled_predictions(num_points: int, seed: int) -> None:
    test = pd.read_csv(TABLE_DIR / "nn_test_predictions.csv")

    # Spread the sample across the whole price range: one random contract
    # from each of `num_points` equal-count price buckets.
    buckets = pd.qcut(test[TARGET], q=num_points, labels=False, duplicates="drop")
    sample = (
        test.groupby(buckets, group_keys=False)
        .apply(lambda g: g.sample(1, random_state=seed))
        .sort_values(TARGET)
    )

    true = sample[TARGET].to_numpy()
    pred = sample["predicted_price"].to_numpy()
    limit = max(true.max(), pred.max()) * 1.05

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, limit], [0, limit], "k--", linewidth=1, label="Perfect fit")
    ax.scatter(true, true, s=55, color="tab:blue", label="True (Reiner–Rubinstein)", zorder=3)
    ax.scatter(true, pred, s=55, color="tab:orange", label="Predicted (NN)", zorder=4,
               edgecolor="white", linewidth=0.6)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("True price")
    ax.set_ylabel("Predicted price")
    mae = np.mean(np.abs(pred - true))
    ax.set_title(f"Predicted vs true DIP prices ({len(sample)} test contracts, MAE {mae:.3f})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    output = FIGURE_DIR / "nn_predicted_vs_true_sample.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)

    sample[FEATURES + [TARGET, "predicted_price"]].to_csv(
        TABLE_DIR / "nn_predicted_vs_true_sample.csv", index=False, float_format="%.6f"
    )
    print(f"Saved {output}")


# ---------------------------------------------------------------------------
# 2. NN-implied volatility surface over (S, T)
# ---------------------------------------------------------------------------

def implied_volatility(price: float, S: float, K: float, H: float, T: float, r: float,
                       low: float = 0.01, high: float = 1.5) -> float:
    """Sigma such that Reiner--Rubinstein(sigma) = price; NaN if not bracketed."""
    def objective(sigma: float) -> float:
        return reiner_rubinstein_dip(S=S, K=K, H=H, T=T, r=r, sigma=sigma) - price

    f_low, f_high = objective(low), objective(high)
    if np.sign(f_low) == np.sign(f_high):
        return np.nan
    return brentq(objective, low, high, xtol=1e-8)


def plot_implied_volatility_surface(model, scaler, K: float, H: float, sigma: float, r: float) -> None:
    # Keep the grid inside the training domain: H/S in [0.65, 0.95], S in [80, 120].
    spot_grid = np.linspace(max(80.0, H / 0.95), min(120.0, H / 0.65), 31)
    maturity_grid = np.linspace(0.1, 2.0, 39)
    S_mesh, T_mesh = np.meshgrid(spot_grid, maturity_grid)

    S_flat, T_flat = S_mesh.ravel(), T_mesh.ravel()
    features = np.column_stack([
        S_flat,
        np.full_like(S_flat, K),
        H / S_flat,
        T_flat,
        np.full_like(S_flat, sigma),
        np.full_like(S_flat, r),
    ])
    prices = nn_price(model, scaler, features)

    iv = np.array([
        implied_volatility(p, S, K, H, T, r) for p, S, T in zip(prices, S_flat, T_flat)
    ]).reshape(S_mesh.shape)

    failed = int(np.isnan(iv).sum())
    deviation = np.abs(iv - sigma)
    print(
        f"Implied vol: mean {np.nanmean(iv):.4f}, min {np.nanmin(iv):.4f}, max {np.nanmax(iv):.4f}, "
        f"mean |IV - sigma| {np.nanmean(deviation):.4f}, unsolved points {failed}/{iv.size}"
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    surface = ax.plot_surface(S_mesh, T_mesh, iv, cmap="viridis", edgecolor="none",
                              antialiased=True, vmin=np.nanmin(iv), vmax=np.nanmax(iv))
    ax.set_xlabel("Spot price")
    ax.set_ylabel("Time to maturity (years)")
    ax.set_zlabel("Implied volatility")
    ax.set_title(
        f"NN-implied volatility surface (K={K:g}, H={H:g}, r={r:.0%}, true σ={sigma:.0%})"
    )
    ax.view_init(elev=25, azim=-130)
    fig.colorbar(surface, ax=ax, shrink=0.6, pad=0.1, label="Implied volatility")
    fig.tight_layout()
    output = FIGURE_DIR / "nn_implied_volatility_surface.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)

    pd.DataFrame(iv, index=np.round(maturity_grid, 4), columns=np.round(spot_grid, 4)).to_csv(
        TABLE_DIR / "nn_implied_volatility_surface.csv", index_label="maturity\\spot"
    )
    print(f"Saved {output}")


def main() -> None:
    args = parse_arguments()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    model, scaler = load_checkpoint(CHECKPOINT_DIR / "dip_mlp.pt")

    plot_sampled_predictions(args.num_points, args.seed)
    plot_implied_volatility_surface(model, scaler, args.strike, args.barrier, args.volatility, args.rate)


if __name__ == "__main__":
    main()
