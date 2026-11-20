
import numpy as np
from scipy.special import ndtr


def black_scholes_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")

    if T < 0 or sigma <= 0:
        raise ValueError("T must be non-negative and sigma positive.")

    if T == 0:
        return max(K - S, 0.0)

    sqrt_T = np.sqrt(T)

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * sqrt_T)

    d2 = d1 - sigma * sqrt_T

    price = (
        K * np.exp(-r * T) * ndtr(-d2)
        - S * ndtr(-d1)
    )

    return float(price)


def reiner_rubinstein_dip(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
) -> float:

    if S <= 0 or K <= 0 or H <= 0:
        raise ValueError("S, K, and H must be positive.")

    if not S > H:
        raise ValueError("Reiner-Rubinstein formula requires S > H.")

    if not H < K:
        raise ValueError("This formula requires H < K.")

    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")

    # Step 1: Define common parameters
    mu_0 = (r - 0.5 * sigma**2) / sigma**2

    v = sigma * np.sqrt(T)

    # Step 2: Calculate x2, y1, and y2
    x2 = (
        np.log(S / H) / v
        + (1 + mu_0) * v
    )

    y1 = (
        np.log(H**2 / (S * K)) / v
        + (1 + mu_0) * v
    )

    y2 = (
        np.log(H / S) / v
        + (1 + mu_0) * v
    )

    discount = np.exp(-r * T)

    # Step 3: Calculate the first component
    term_0 = (
        K * discount * ndtr(-x2 + v)
        - S * ndtr(-x2)
    )

    # Step 4: Calculate the y2 component
    term_y2 = (
        S * (H / S)**(2 * (mu_0 + 1)) * ndtr(y2)
        - K * discount * (H / S)**(2 * mu_0)
        * ndtr(y2 - v)
    )

    # Step 5: Calculate the y1 component
    term_y1 = (
        S * (H / S)**(2 * (mu_0 + 1)) * ndtr(y1)
        - K * discount * (H / S)**(2 * mu_0)
        * ndtr(y1 - v)
    )

    # Step 6: Reiner-Rubinstein DIP price
    price = term_0 - term_y2 + term_y1

    return float(price)


def price_dip(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    barrier_hit: bool = False,
) -> float:

    if barrier_hit or S <= H:
        return black_scholes_put(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
        )

    return reiner_rubinstein_dip(
        S=S,
        K=K,
        H=H,
        T=T,
        r=r,
        sigma=sigma,
    )
