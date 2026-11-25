"""Strategy 2 (Monte Carlo Delta with CRN) and comparison with Strategy 1.

Both strategies hedge the same real-world paths, start from the same premium,
and use the same self-financing accounting and transaction-cost levels, so
differences in P&L come only from how Delta is computed.

Run from the project root:
    python -m hedging.run_mc_hedge
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hedging.analytical_delta import analytical_delta_positions, option_value_path
from hedging.mc_delta import mc_delta_positions
from hedging.portfolio import run_self_financing_hedge, summarize_pnl
from hedging.run_analytical_hedge import FIGURE_DIR, TABLE_DIR, load_configs, to_markdown
from pricing.analytical import price_dip
from stock_simulation.geometric_brownian_motion import simulate_gbm


def summary_markdown(summary: pd.DataFrame, cvar_key: str, cvar_label: str) -> str:
    display = pd.DataFrame({
        "Strategy": summary["strategy"],
        "Transaction cost": [f"{c:.1%}" for c in summary["transaction_cost"]],
        "Mean P&L": summary["mean_pnl"].map("{:+.4f}".format),
        "Std P&L": summary["std_pnl"].map("{:.4f}".format),
        cvar_label: summary[cvar_key].map("{:+.4f}".format),
        "Turnover (shares)": summary["mean_turnover"].map("{:.3f}".format),
    })
    return to_markdown(display)


def plot_pnl_comparison(unhedged, analytical, mc, output) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    low = min(analytical.min(), mc.min())
    high = max(analytical.max(), mc.max())
    bins = np.linspace(low, high, 120)
    axes[0].hist(analytical, bins=bins, alpha=0.55, label=f"Analytical Δ (std {analytical.std(ddof=1):.3f})")
    axes[0].hist(mc, bins=bins, alpha=0.55, label=f"MC Δ, CRN (std {mc.std(ddof=1):.3f})")
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1, label="Target P&L = 0")
    axes[0].set_xlim(np.percentile(np.r_[analytical, mc], [0.2, 99.8]))
    axes[0].set_title("Terminal P&L, daily rebalancing, no costs")
    axes[0].set_xlabel("P&L per contract")
    axes[0].set_ylabel("Number of paths")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    all_bins = np.linspace(min(unhedged.min(), low), max(unhedged.max(), high), 120)
    axes[1].hist(unhedged, bins=all_bins, alpha=0.5, label="Unhedged")
    axes[1].hist(analytical, bins=all_bins, alpha=0.5, label="Analytical Δ")
    axes[1].hist(mc, bins=all_bins, alpha=0.5, label="MC Δ, CRN")
    axes[1].set_yscale("log")
    axes[1].set_title("Including the unhedged baseline (log count)")
    axes[1].set_xlabel("P&L per contract")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_cost_sensitivity(summary: pd.DataFrame, cvar_key: str, output) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for strategy, marker in (("Analytical Δ", "o"), ("MC Δ (CRN)", "s")):
        part = summary[summary["strategy"] == strategy]
        x = part["transaction_cost"] * 100
        axes[0].plot(x, part["mean_pnl"], marker=marker, label=strategy)
        axes[1].plot(x, part["std_pnl"], marker=marker, label=strategy)
        axes[2].plot(x, part[cvar_key], marker=marker, label=strategy)
    for ax, title in zip(axes, ("Mean P&L", "Std P&L", "CVaR 5%")):
        ax.set_title(title)
        ax.set_xlabel("Transaction cost (%)")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_path_comparison(path, pos_analytical, pos_mc, value_analytical, value_mc, liability,
                         strike, barrier, output) -> None:
    days = np.arange(len(path))
    hit_day = int(np.argmax(path <= barrier))
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(days, path, color="tab:blue", label="Stock price")
    axes[0].axhline(barrier, color="red", linestyle="--", label=f"Barrier H = {barrier:g}")
    axes[0].axhline(strike, color="grey", linestyle=":", label=f"Strike K = {strike:g}")
    axes[0].set_ylabel("Price")
    axes[0].set_title(f"Same barrier-hit path, two Delta methods (knock-in on day {hit_day})")
    axes[0].legend(loc="upper right")

    axes[1].step(days[:-1], pos_analytical, where="post", label="Analytical Δ")
    axes[1].step(days[:-1], pos_mc, where="post", label="MC Δ (CRN)")
    axes[1].set_ylabel("Shares held (Δ)")
    axes[1].legend(loc="lower left")

    axes[2].plot(days, value_analytical, label="Hedge portfolio, analytical Δ")
    axes[2].plot(days, value_mc, label="Hedge portfolio, MC Δ")
    axes[2].plot(days, liability, color="black", linestyle="--", linewidth=1,
                 label="Liability (RR model value / payoff)")
    axes[2].set_ylabel("Value")
    axes[2].set_xlabel("Trading day")
    axes[2].legend(loc="upper left")
    for ax in axes:
        ax.axvline(hit_day, color="red", alpha=0.3)
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

    sim, hedger, mc_cfg = hedging["simulation"], hedging["hedger"], hedging["mc_delta"]
    sigma_true, sigma_model = sim["true_volatility"], hedger["model_volatility"]
    cvar_level = hedging["evaluation"]["cvar_level"]
    cvar_key = f"cvar_{int(round(cvar_level * 100))}"
    cvar_label = f"CVaR {cvar_level:.0%}"

    # Same real-world paths and premium as Strategy 1.
    paths = simulate_gbm(S0, T, sigma_true, mu, num_steps=steps,
                         num_paths=sim["num_paths"], random_seed=sim["random_seed"])
    premium = price_dip(S0, K, H, T, r, sigma_model)
    print(f"Paths: {len(paths):,} | premium {premium:.4f} | inner paths per date {mc_cfg['inner_paths']:,}")

    start = time.perf_counter()
    pos_analytical = analytical_delta_positions(
        paths, K, H, T, r, sigma_model,
        rebalance_every=hedger["rebalance_every"], bump_relative=hedger["delta_bump_relative"],
    )
    analytical_seconds = time.perf_counter() - start

    start = time.perf_counter()
    pos_mc = mc_delta_positions(
        paths, K, H, T, r, sigma_model,
        num_inner=mc_cfg["inner_paths"], bump_relative=mc_cfg["bump_relative"],
        rebalance_every=hedger["rebalance_every"], random_seed=mc_cfg["random_seed"],
    )
    mc_seconds = time.perf_counter() - start
    print(f"Delta computation: analytical {analytical_seconds:.1f}s, MC {mc_seconds:.1f}s")

    common = dict(premium=premium, risk_free_rate=r, dt=dt, strike=K, barrier=H)
    unhedged = run_self_financing_hedge(paths, np.zeros_like(pos_mc), transaction_cost=0.0, **common)

    rows = [{"strategy": "Unhedged", "transaction_cost": 0.0,
             **summarize_pnl(unhedged["pnl"], unhedged["turnover"], cvar_level)}]
    pnl_columns = {"unhedged": unhedged["pnl"]}
    runs = {}
    for name, positions in (("Analytical Δ", pos_analytical), ("MC Δ (CRN)", pos_mc)):
        for cost in hedging["transaction_costs"]:
            result = run_self_financing_hedge(paths, positions, transaction_cost=cost, **common)
            runs[(name, cost)] = result
            key = "analytical" if name.startswith("Analytical") else "mc"
            pnl_columns[f"{key}_tc_{cost:g}"] = result["pnl"]
            rows.append({"strategy": name, "transaction_cost": cost,
                         **summarize_pnl(result["pnl"], result["turnover"], cvar_level)})
    summary = pd.DataFrame(rows)

    # ---- Tables ---------------------------------------------------------
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    mc_only = summary[summary["strategy"].isin(["Unhedged", "MC Δ (CRN)"])]
    mc_only.to_csv(TABLE_DIR / "hedging_mc_summary.csv", index=False, float_format="%.6f")
    (TABLE_DIR / "hedging_mc_summary.md").write_text(summary_markdown(mc_only, cvar_key, cvar_label) + "\n")

    summary.to_csv(TABLE_DIR / "hedging_comparison.csv", index=False, float_format="%.6f")
    comparison_md = summary_markdown(summary, cvar_key, cvar_label)
    (TABLE_DIR / "hedging_comparison.md").write_text(comparison_md + "\n")

    pd.DataFrame(pnl_columns).assign(barrier_hit=unhedged["barrier_hit"]).to_csv(
        TABLE_DIR / "hedging_mc_pnl_paths.csv", index=False, float_format="%.6f"
    )

    base_cost = min(hedging["transaction_costs"])
    pnl_a = runs[("Analytical Δ", base_cost)]["pnl"]
    pnl_m = runs[("MC Δ (CRN)", base_cost)]["pnl"]
    delta_gap = pos_mc - pos_analytical
    metadata = {
        "num_paths": len(paths),
        "premium": premium,
        "barrier_hit_rate": float(unhedged["barrier_hit"].mean()),
        "inner_paths_per_date": mc_cfg["inner_paths"],
        "bump_relative": mc_cfg["bump_relative"],
        "delta_seconds": {"analytical": analytical_seconds, "mc": mc_seconds},
        "initial_delta": {"analytical": float(pos_analytical[0, 0]), "mc": float(pos_mc[0, 0])},
        "mean_abs_delta_gap_mc_minus_analytical": float(np.mean(np.abs(delta_gap))),
        f"paired_pnl_difference_mc_minus_analytical_tc_{base_cost:g}": {
            "mean": float(np.mean(pnl_m - pnl_a)),
            "std": float(np.std(pnl_m - pnl_a, ddof=1)),
            "correlation": float(np.corrcoef(pnl_m, pnl_a)[0, 1]),
        },
        "variance_reduction_vs_unhedged": {
            "analytical": float(1 - np.var(pnl_a) / np.var(unhedged["pnl"])),
            "mc": float(1 - np.var(pnl_m) / np.var(unhedged["pnl"])),
        },
    }
    with open(TABLE_DIR / "hedging_comparison_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    # ---- Figures --------------------------------------------------------
    plot_pnl_comparison(unhedged["pnl"], pnl_a, pnl_m, FIGURE_DIR / "hedging_comparison_pnl_hist.png")
    plot_cost_sensitivity(summary, cvar_key, FIGURE_DIR / "hedging_comparison_cost_sensitivity.png")

    hit_paths = np.flatnonzero(unhedged["barrier_hit"])
    if len(hit_paths):
        j = hit_paths[0]  # same path as the Strategy 1 illustration
        plot_path_comparison(
            paths[j], pos_analytical[j], pos_mc[j],
            runs[("Analytical Δ", base_cost)]["portfolio_value"][j],
            runs[("MC Δ (CRN)", base_cost)]["portfolio_value"][j],
            option_value_path(paths[j], K, H, T, r, sigma_model), K, H,
            FIGURE_DIR / "hedging_comparison_path_example.png",
        )

    print("\n" + comparison_md)
    print("\n" + json.dumps(metadata, indent=2))
    print(f"\nSaved tables to {TABLE_DIR} and figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
