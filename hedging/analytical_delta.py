"""Strategy 1: analytical Delta for a short down-and-in put.

* Before the barrier is hit: central finite difference of the closed-form
  Reiner--Rubinstein price, (V(S + h) - V(S - h)) / (2h) with h = bump * S.
  If S - h falls below H, `price_dip` returns the vanilla put value there,
  which equals the DIP value at the barrier (continuous monitoring).
* After the barrier is hit (on any monitoring date so far): Black--Scholes
  put Delta, N(d1) - 1. The hit flag never resets.

The hedger holds Delta shares (negative = short) of the stock.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from pricing.analytical import price_dip


_price_dip_vectorized = np.vectorize(price_dip, otypes=[float])


def black_scholes_put_delta(S, K, tau, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return ndtr(d1) - 1.0


def dip_delta_before_hit(S, K, H, tau, r, sigma, bump_relative: float = 1e-4):
    h = bump_relative * S
    up = _price_dip_vectorized(S + h, K, H, tau, r, sigma)
    down = _price_dip_vectorized(S - h, K, H, tau, r, sigma)
    return (up - down) / (2.0 * h)


def analytical_delta_positions(
    paths: np.ndarray,
    strike: float,
    barrier: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    rebalance_every: int = 1,
    bump_relative: float = 1e-4,
) -> np.ndarray:
    """Shares held from t_i to t_{i+1}, shape (num_paths, N).

    Delta is recomputed every `rebalance_every` steps and held constant
    in between. Barrier monitoring itself is still done on every date.
    """
    num_paths, num_dates = paths.shape
    num_steps = num_dates - 1
    dt = maturity / num_steps
    hit_so_far = np.minimum.accumulate(paths, axis=1) <= barrier

    positions = np.empty((num_paths, num_steps))
    for i in range(num_steps):
        if i % rebalance_every:
            positions[:, i] = positions[:, i - 1]
            continue

        S = paths[:, i]
        tau = maturity - i * dt
        hit = hit_so_far[:, i]
        delta = np.empty(num_paths)
        if hit.any():
            delta[hit] = black_scholes_put_delta(S[hit], strike, tau, risk_free_rate, volatility)
        if (~hit).any():
            delta[~hit] = dip_delta_before_hit(
                S[~hit], strike, barrier, tau, risk_free_rate, volatility, bump_relative
            )
        positions[:, i] = delta
    return positions


def option_value_path(path: np.ndarray, strike: float, barrier: float, maturity: float,
                      risk_free_rate: float, volatility: float) -> np.ndarray:
    """Model value of the liability along one path (for illustrations)."""
    num_steps = len(path) - 1
    dt = maturity / num_steps
    hit_so_far = np.minimum.accumulate(path) <= barrier
    values = np.empty(len(path))
    for i, S in enumerate(path[:-1]):
        values[i] = price_dip(S, strike, barrier, maturity - i * dt, risk_free_rate, volatility,
                              barrier_hit=bool(hit_so_far[i]))
    values[-1] = max(strike - path[-1], 0.0) if hit_so_far[-1] else 0.0
    return values
