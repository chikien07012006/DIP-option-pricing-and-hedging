"""Compare the four pricing methods on the baseline down-and-in put.

Methods
-------
1. Reiner--Rubinstein closed form (continuous monitoring).
2. Standard Monte Carlo (discrete daily monitoring).
3. Monte Carlo with antithetic variates (same total path budget).
4. MLP pricing surrogate (trained on Reiner--Rubinstein labels).

Reference
---------
Every method is compared with the continuous-monitoring Reiner--Rubinstein
closed form (no continuity correction, as in configs/pricing.yaml). The Monte
Carlo methods monitor the barrier daily, so their bias includes the
continuous-vs-daily monitoring gap.

Run from the project root (needs results/checkpoints/dip_mlp.pt):
    python -m pricing.compare_methods
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import norm

from pricing.analytical import reiner_rubinstein_dip
from pricing.monte_carlo_antithetic_variates import monte_carlo_av_dip
from pricing.monte_carlo_standard import monte_carlo_standard_dip
from pricing.neural_network import CHECKPOINT_DIR, load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "pricing.yaml"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"


def average_runtime(function, repeats: int) -> float:
    """Mean wall-clock seconds per call."""
    function()  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - start) / repeats


def to_markdown(frame: pd.DataFrame) -> str:
    """Minimal markdown table (avoids the optional `tabulate` dependency)."""
    def fmt(value) -> str:
        if isinstance(value, float):
            return "—" if np.isnan(value) else f"{value:.6g}"
        return str(value)

    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def format_runtime(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def compact_table(table: pd.DataFrame, nn_batch_us: float | None) -> pd.DataFrame:
    """Report view: Method | Price | Bias | SE / CI 95% | Estimator variance | VRR | Runtime."""
    rows = []
    for row in table.itertuples(index=False):
        is_mc = not pd.isna(getattr(row, "std_error", np.nan))
        if row.reference == "RR (itself)":
            bias = "— (benchmark)"
        else:
            bias = f"{row.bias:+.4f} (vs {row.reference})"
        se_ci = (
            f"{row.std_error:.4f} / [{row.ci95_lower:.4f}, {row.ci95_upper:.4f}]" if is_mc else "—"
        )
        variance = f"{row.estimator_variance:.3e}" if is_mc else (
            "0" if row.estimator_variance == 0 else "—"
        )
        vrr = f"{row.vrr_vs_standard:.2f}" if is_mc else "—"
        runtime = format_runtime(row.runtime_seconds)
        if row.method.startswith("Neural") and nn_batch_us is not None:
            runtime += f" ({nn_batch_us:.2f} µs/contract batched)"
        rows.append({
            "Method": row.method,
            "Price": f"{row.price:.4f}",
            "Bias vs reference": bias,
            "SE / CI 95%": se_ci,
            "Estimator variance": variance,
            "VRR": vrr,
            "Runtime per pricing": runtime,
        })
    return pd.DataFrame(rows)


def main() -> None:
    with open(CONFIG_PATH) as handle:
        config = yaml.safe_load(handle)

    contract, market, mc = config["contract"], config["market"], config["monte_carlo"]
    S = contract["spot_price"]
    K = contract["strike_price"]
    H = contract["barrier_level"]
    T = contract["maturity"]
    steps = contract["monitoring_steps"]
    r = market["risk_free_rate"]
    sigma = market["volatility"]
    seed = mc["random_seed"]
    num_paths = mc["standard"]["num_simulations"]
    num_pairs = mc["antithetic_variates"]["num_pairs"]
    z = norm.ppf(0.5 + mc["confidence_level"] / 2)

    params = dict(S=S, K=K, H=H, T=T, r=r, sigma=sigma)
    print(f"Contract: S={S}, K={K}, H={H}, T={T}, r={r}, sigma={sigma}, steps={steps}")

    # ---- 1. Analytical prices --------------------------------------------
    rr_price = reiner_rubinstein_dip(**params)
    rr_seconds = average_runtime(lambda: reiner_rubinstein_dip(**params), repeats=2000)
    print(f"Reiner-Rubinstein (continuous): {rr_price:.6f}")

    # ---- 2. Standard Monte Carlo -----------------------------------------
    print(f"Running standard MC ({num_paths:,} paths) ...")
    start = time.perf_counter()
    std = monte_carlo_standard_dip(**params, num_simulations=num_paths, num_steps=steps, random_seed=seed)
    std_seconds = time.perf_counter() - start

    # ---- 3. Antithetic Monte Carlo ---------------------------------------
    print(f"Running antithetic MC ({num_pairs:,} pairs = {2 * num_pairs:,} paths) ...")
    start = time.perf_counter()
    av = monte_carlo_av_dip(**params, num_pairs=num_pairs, num_steps=steps, random_seed=seed)
    av_seconds = time.perf_counter() - start

    # ---- 4. Neural network -----------------------------------------------
    model, scaler = load_checkpoint(CHECKPOINT_DIR / "dip_mlp.pt")
    features = np.array([[S, K, H / S, T, sigma, r]], dtype=np.float32)
    x = torch.from_numpy(scaler.transform(features))

    def nn_call() -> float:
        with torch.no_grad():
            return float(model(x).item())

    nn_price = nn_call()
    nn_seconds = average_runtime(nn_call, repeats=2000)

    # ---- Assemble table --------------------------------------------------
    std_est_var = std["standard_error"] ** 2
    av_est_var = av["standard_error"] ** 2
    vrr = std_est_var / av_est_var
    efficiency_ratio = (std_est_var * std_seconds) / (av_est_var * av_seconds)
    pair_correlation = 1.0 / vrr - 1.0  # VRR = 1 / (1 + rho) at equal path budget

    def mc_row(name: str, result: dict, paths: int, per_sample_label: str,
               est_var: float, seconds: float, ratio: float, eff: float) -> dict:
        low, high = result["confidence_interval"]
        return {
            "method": name,
            "barrier_monitoring": "discrete daily",
            "num_paths": paths,
            "price": result["price"],
            "reference": "RR",
            "reference_price": rr_price,
            "bias": result["price"] - rr_price,
            "abs_relative_error": abs(result["price"] - rr_price) / rr_price,
            "std_error": result["standard_error"],
            "ci95_lower": low,
            "ci95_upper": high,
            "ci95_width": high - low,
            "ci_contains_reference": bool(low <= rr_price <= high),
            "per_sample_variance": result["variance"],
            "per_sample_unit": per_sample_label,
            "estimator_variance": est_var,
            "vrr_vs_standard": ratio,
            "efficiency_ratio_vs_standard": eff,
            "runtime_seconds": seconds,
        }

    rows = [
        {
            "method": "Reiner-Rubinstein (closed form)",
            "barrier_monitoring": "continuous",
            "price": rr_price,
            "reference": "RR (itself)",
            "reference_price": rr_price,
            "bias": 0.0,
            "abs_relative_error": 0.0,
            "estimator_variance": 0.0,
            "runtime_seconds": rr_seconds,
        },
        mc_row("Standard Monte Carlo", std, num_paths, "path", std_est_var, std_seconds, 1.0, 1.0),
        mc_row("Monte Carlo antithetic variates", av, 2 * num_pairs, "pair average", av_est_var,
               av_seconds, vrr, efficiency_ratio),
        {
            "method": "Neural network (MLP)",
            "barrier_monitoring": "continuous (learned)",
            "price": nn_price,
            "reference": "RR",
            "reference_price": rr_price,
            "bias": nn_price - rr_price,
            "abs_relative_error": abs(nn_price - rr_price) / rr_price,
            "runtime_seconds": nn_seconds,
        },
    ]
    table = pd.DataFrame(rows)

    summary = {
        "contract": {"S": S, "K": K, "H": H, "T": T, "r": r, "sigma": sigma, "monitoring_steps": steps},
        "reiner_rubinstein_continuous": rr_price,
        "variance_reduction_ratio": vrr,
        "implied_pair_correlation": pair_correlation,
        "std_error_reduction_factor": float(np.sqrt(vrr)),
        "efficiency_ratio": efficiency_ratio,
        "paths_standard_mc_needs_to_match_av": int(round(num_paths * vrr)),
        "nn_speedup_vs_standard_mc": std_seconds / nn_seconds,
        "z_critical": float(z),
    }

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "pricing_comparison.csv", index=False, float_format="%.8g")
    with open(TABLE_DIR / "pricing_comparison_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    nn_metrics_path = TABLE_DIR / "nn_metrics.json"
    nn_batch_us = (
        json.loads(nn_metrics_path.read_text())["inference_microseconds_per_contract"]
        if nn_metrics_path.exists() else None
    )
    markdown = to_markdown(compact_table(table, nn_batch_us))
    (TABLE_DIR / "pricing_comparison.md").write_text(markdown + "\n")

    print("\n" + markdown)
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSaved tables under {TABLE_DIR}")


if __name__ == "__main__":
    main()
