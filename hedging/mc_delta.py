"""Strategy 2: Monte Carlo Delta with Common Random Numbers (CRN).

At each rebalancing date t_i, for every hedge path with spot S:

    Delta = [V(S + h) - V(S)] / h,     h = bump * S

where V is a Monte Carlo price of the remaining down-and-in put with daily
barrier monitoring on t_{i+1}..t_N:

    barrier already hit before t_i  -> payoff = max(K - S_T, 0)          (vanilla put)
    not hit yet                     -> payoff = max(K - S_T, 0) * 1{min S_t <= H}

CRN: V(S) and V(S + h) use the same M inner paths. One fresh set of M inner
paths is drawn per date and shared by all hedge paths. Because GBM is
multiplicative, an inner path started at S is S * R_t with R independent of S,
so only R_T and min_t R_t are simulated:

    S_T = S * R_T,    min_t S_t = S * min_t R_t.
"""

from __future__ import annotations

import numpy as np


def simulate_growth_factors(rng: np.random.Generator, num_inner: int, num_steps: int,
                            dt: float, r: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Risk-neutral R_T and min over monitoring dates of R_t, each shape (num_inner,)."""
    z = rng.standard_normal((num_inner, num_steps))
    log_paths = np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    return np.exp(log_paths[:, -1]), np.exp(log_paths.min(axis=1))


def mc_dip_value(S: np.ndarray, already_hit: np.ndarray, R_T: np.ndarray, R_min: np.ndarray,
                 strike: float, barrier: float, discount: float, chunk: int = 500) -> np.ndarray:
    """MC value of the remaining DIP for each spot in S, using the same inner paths."""
    values = np.empty(len(S))
    for start in range(0, len(S), chunk):
        s = S[start:start + chunk, None]                     # (chunk, 1)
        payoff = np.maximum(strike - s * R_T[None, :], 0.0)  # (chunk, M)
        knocked_in = already_hit[start:start + chunk, None] | (s * R_min[None, :] <= barrier)
        values[start:start + chunk] = discount * np.mean(payoff * knocked_in, axis=1)
    return values


def mc_delta_positions(
    paths: np.ndarray,
    strike: float,
    barrier: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    num_inner: int = 5000,
    bump_relative: float = 0.01,
    rebalance_every: int = 1,
    random_seed: int = 7,
) -> np.ndarray:
    """Shares held from t_i to t_{i+1}, shape (num_paths, N), from forward-difference MC Delta."""
    num_paths, num_dates = paths.shape
    num_steps = num_dates - 1
    dt = maturity / num_steps
    rng = np.random.default_rng(random_seed)
    hit_so_far = np.minimum.accumulate(paths, axis=1) <= barrier

    positions = np.empty((num_paths, num_steps))
    for i in range(num_steps):
        if i % rebalance_every:
            positions[:, i] = positions[:, i - 1]
            continue

        remaining = num_steps - i
        R_T, R_min = simulate_growth_factors(rng, num_inner, remaining, dt, risk_free_rate, volatility)
        discount = np.exp(-risk_free_rate * remaining * dt)

        S = paths[:, i]
        h = bump_relative * S
        hit = hit_so_far[:, i]
        v_base = mc_dip_value(S, hit, R_T, R_min, strike, barrier, discount)
        v_up = mc_dip_value(S + h, hit, R_T, R_min, strike, barrier, discount)
        positions[:, i] = (v_up - v_base) / h
    return positions
