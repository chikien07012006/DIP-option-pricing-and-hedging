"""Self-financing hedge accounting for a short down-and-in put.

Timeline (N monitoring steps, dt = T / N):

    t_0          Cash_0 = premium - Delta_0 * S_0 - cost_0
    t_i, 0<i<N   Cash_i = Cash_{i-1} * e^{r dt} - (Delta_i - Delta_{i-1}) * S_i - cost_i
    t_N = T      liquidate: Cash_T = Cash_{N-1} * e^{r dt} + Delta_{N-1} * S_T - cost_N
                 P&L_T = Cash_T - payoff

where cost = transaction_cost * |shares traded| * S. The barrier is monitored
on every date t_0..t_N, and payoff = max(K - S_T, 0) * 1{min S_t <= H}.
"""

from __future__ import annotations

import numpy as np


def run_self_financing_hedge(
    paths: np.ndarray,
    positions: np.ndarray,
    premium: float,
    risk_free_rate: float,
    dt: float,
    transaction_cost: float,
    strike: float,
    barrier: float,
) -> dict:
    """Simulate the hedge on every path.

    Parameters
    ----------
    paths
        Stock prices, shape (num_paths, N + 1), columns t_0..t_N.
    positions
        Shares held from t_i to t_{i+1}, shape (num_paths, N). The position
        held into maturity, positions[:, -1], is liquidated at T.
        All zeros = unhedged.
    """
    num_paths, num_dates = paths.shape
    num_steps = num_dates - 1
    if positions.shape != (num_paths, num_steps):
        raise ValueError(f"positions must have shape {(num_paths, num_steps)}, got {positions.shape}.")

    growth = np.exp(risk_free_rate * dt)
    portfolio_value = np.empty((num_paths, num_dates))  # cash + shares * S (mark-to-market)

    # t_0: receive premium, open the initial position.
    trade = positions[:, 0]
    cost = transaction_cost * np.abs(trade) * paths[:, 0]
    cash = premium - trade * paths[:, 0] - cost
    turnover = np.abs(trade)
    total_cost = cost
    portfolio_value[:, 0] = cash + positions[:, 0] * paths[:, 0]

    # t_1 .. t_{N-1}: accrue interest, rebalance.
    for i in range(1, num_steps):
        S = paths[:, i]
        trade = positions[:, i] - positions[:, i - 1]
        cost = transaction_cost * np.abs(trade) * S
        cash = cash * growth - trade * S - cost
        turnover = turnover + np.abs(trade)
        total_cost = total_cost + cost
        portfolio_value[:, i] = cash + positions[:, i] * S

    # t_N = T: accrue interest, liquidate the remaining position.
    S_T = paths[:, -1]
    final_position = positions[:, -1]
    cost = transaction_cost * np.abs(final_position) * S_T
    cash = cash * growth + final_position * S_T - cost
    turnover = turnover + np.abs(final_position)
    total_cost = total_cost + cost
    portfolio_value[:, -1] = cash

    barrier_hit = paths.min(axis=1) <= barrier
    payoff = np.where(barrier_hit, np.maximum(strike - S_T, 0.0), 0.0)

    return {
        "pnl": cash - payoff,
        "payoff": payoff,
        "barrier_hit": barrier_hit,
        "turnover": turnover,                # shares traded per path
        "transaction_costs": total_cost,     # currency per path
        "portfolio_value": portfolio_value,  # hedge portfolio value at each date
    }


def summarize_pnl(pnl: np.ndarray, turnover: np.ndarray, cvar_level: float = 0.05) -> dict:
    """Mean, std, CVaR (mean of the worst `cvar_level` share of paths), turnover."""
    worst = np.sort(pnl)[: max(1, int(np.ceil(cvar_level * len(pnl))))]
    return {
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)),
        f"cvar_{int(round(cvar_level * 100))}": float(np.mean(worst)),
        "mean_turnover": float(np.mean(turnover)),
    }
