"""Strategy 1: analytical-Delta hedging of a short down-and-in put.

Simulates real-world GBM paths, hedges with the closed-form Delta, and reports
mean / std / CVaR of terminal P&L and turnover for each transaction-cost level,
next to an unhedged baseline. Target: P&L = 0 on every path.

Run from the project root:
    python -m hedging.run_analytical_hedge
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from hedging.analytical_delta import analytical_delta_positions, option_value_path
from hedging.portfolio import run_self_financing_hedge, summarize_pnl
from pricing.analytical import price_dip
from stock_simulation.geometric_brownian_motion import simulate_gbm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"


def load_configs() -> tuple[dict, dict]:
    with open(CONFIG_DIR / "pricing.yaml") as handle:
        pricing = yaml.safe_load(handle)
    with open(CONFIG_DIR / "hedging.yaml") as handle:
        hedging = yaml.safe_load(handle)
    return pricing, hedging


def to_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def plot_pnl_histogram(unhedged: np.ndarray, hedged: np.ndarray, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(min(unhedged.min(), hedged.min()), max(unhedged.max(), hedged.max()), 120)
    axes[0].hist(unhedged, bins=bins, alpha=0.6, label="Unhedged")
    axes[0].hist(hedged, bins=bins, alpha=0.6, label="Analytical Δ (no costs)")
    axes[0].set_yscale("log")
    axes[0].set_title("Terminal P&L: unhedged vs hedged (log count)")
    axes[0].set_xlabel("P&L per contract")
    axes[0].set_ylabel("Number of paths")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(hedged, bins=100, color="tab:orange", alpha=0.8)
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1, label="Target P&L = 0")
    axes[1].axvline(hedged.mean(), color="red", linewidth=1, label=f"Mean = {hedged.mean():.3f}")
    axes[1].set_title("Analytical Δ hedge, daily, no costs")
    axes[1].set_xlabel("P&L per contract")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_hedge_path(path, positions, portfolio_value, option_value, strike, barrier, output: Path) -> None:
    days = np.arange(len(path))
    hit_day = int(np.argmax(path <= barrier))

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(days, path, color="tab:blue", label="Stock price")
    axes[0].axhline(barrier, color="red", linestyle="--", label=f"Barrier H = {barrier:g}")
    axes[0].axhline(strike, color="grey", linestyle=":", label=f"Strike K = {strike:g}")
    axes[0].axvline(hit_day, color="red", alpha=0.3)
    axes[0].set_ylabel("Price")
    axes[0].set_title(f"Analytical Δ hedge on a barrier-hit path (knock-in on day {hit_day})")
    axes[0].legend(loc="upper right")

    axes[1].step(days[:-1], positions, where="post", color="tab:purple")
    axes[1].axvline(hit_day, color="red", alpha=0.3)
    axes[1].set_ylabel("Shares held (Δ)")

    axes[2].plot(days, portfolio_value, label="Hedge portfolio (cash + Δ·S)")
    axes[2].plot(days, option_value, label="Liability (model value / payoff)")
    axes[2].axvline(hit_day, color="red", alpha=0.3)
    axes[2].set_ylabel("Value")
    axes[2].set_xlabel("Trading day")
    axes[2].legend(loc="upper left")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    pricing, hedging = load_configs()
    contract, market = pricing["contract"], pricing["market"]
    S0, K, H = contract["spot_price"], contract["strike_price"], contract["barrier_level"]
    T, steps = contract["maturity"], contract["monitoring_steps"]
    r, mu = market["risk_free_rate"], market["expected_return"]
    dt = T / steps

    sim, hedger = hedging["simulation"], hedging["hedger"]
    sigma_true, sigma_model = sim["true_volatility"], hedger["model_volatility"]
    cvar_level = hedging["evaluation"]["cvar_level"]
    cvar_key = f"cvar_{int(round(cvar_level * 100))}"

    # Real-world paths (drift mu, true vol).
    paths = simulate_gbm(S0, T, sigma_true, mu, num_steps=steps,
                         num_paths=sim["num_paths"], random_seed=sim["random_seed"])

    premium = price_dip(S0, K, H, T, r, sigma_model)
    print(f"Paths: {len(paths):,} | mu={mu} sigma_true={sigma_true} sigma_model={sigma_model}")
    print(f"Premium (Reiner-Rubinstein, sigma_model): {premium:.4f}")

    start = time.perf_counter()
    positions = analytical_delta_positions(
        paths, K, H, T, r, sigma_model,
        rebalance_every=hedger["rebalance_every"],
        bump_relative=hedger["delta_bump_relative"],
    )
    print(f"Delta computation: {time.perf_counter() - start:.1f}s")

    common = dict(premium=premium, risk_free_rate=r, dt=dt, strike=K, barrier=H)
    unhedged = run_self_financing_hedge(paths, np.zeros_like(positions), transaction_cost=0.0, **common)

    rows = [{"strategy": "Unhedged", "transaction_cost": 0.0,
             **summarize_pnl(unhedged["pnl"], unhedged["turnover"], cvar_level)}]
    pnl_by_scenario = {"unhedged": unhedged["pnl"]}
    results = {}
    for cost in hedging["transaction_costs"]:
        result = run_self_financing_hedge(paths, positions, transaction_cost=cost, **common)
        results[cost] = result
        pnl_by_scenario[f"analytical_tc_{cost:g}"] = result["pnl"]
        rows.append({"strategy": "Analytical Δ", "transaction_cost": cost,
                     **summarize_pnl(result["pnl"], result["turnover"], cvar_level)})

    summary = pd.DataFrame(rows)
    barrier_hit_rate = float(unhedged["barrier_hit"].mean())

    # ---- Save tables ----------------------------------------------------
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "hedging_analytical_summary.csv", index=False, float_format="%.6f")
    pd.DataFrame(pnl_by_scenario).assign(barrier_hit=unhedged["barrier_hit"]).to_csv(
        TABLE_DIR / "hedging_analytical_pnl_paths.csv", index=False, float_format="%.6f"
    )

    display = pd.DataFrame({
        "Strategy": summary["strategy"],
        "Transaction cost": [f"{c:.1%}" for c in summary["transaction_cost"]],
        "Mean P&L": summary["mean_pnl"].map("{:+.4f}".format),
        "Std P&L": summary["std_pnl"].map("{:.4f}".format),
        f"CVaR {cvar_level:.0%}": summary[cvar_key].map("{:+.4f}".format),
        "Turnover (shares)": summary["mean_turnover"].map("{:.3f}".format),
    })
    markdown = to_markdown(display)
    (TABLE_DIR / "hedging_analytical_summary.md").write_text(markdown + "\n")

    metadata = {
        "num_paths": len(paths),
        "premium": premium,
        "barrier_hit_rate": barrier_hit_rate,
        "drift": mu,
        "true_volatility": sigma_true,
        "model_volatility": sigma_model,
        "rebalance_every": hedger["rebalance_every"],
        "variance_reduction_vs_unhedged_no_cost": float(
            1 - np.var(results[0.0]["pnl"]) / np.var(unhedged["pnl"])
        ) if 0.0 in results else None,
    }
    with open(TABLE_DIR / "hedging_analytical_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    # ---- Figures --------------------------------------------------------
    base = results[min(results)]
    plot_pnl_histogram(unhedged["pnl"], base["pnl"], FIGURE_DIR / "hedging_analytical_pnl_hist.png")

    hit_paths = np.flatnonzero(unhedged["barrier_hit"])
    if len(hit_paths):
        j = hit_paths[0]
        plot_hedge_path(
            paths[j], positions[j], base["portfolio_value"][j],
            option_value_path(paths[j], K, H, T, r, sigma_model), K, H,
            FIGURE_DIR / "hedging_analytical_path_example.png",
        )

    print("\n" + markdown)
    print("\n" + json.dumps(metadata, indent=2))
    print(f"\nSaved tables to {TABLE_DIR} and figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
