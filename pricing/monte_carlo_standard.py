import numpy as np
from scipy.stats import norm


def monte_carlo_standard_dip(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    num_simulations: int = 100_000,
    num_steps: int = 63,
    random_seed: int = 42,
) -> dict:
    if S <= 0 or K <= 0 or H <= 0:
        raise ValueError("S, K, H must be positive.")

    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")

    if H >= K:
        raise ValueError("This implementation assumes H < K.")

    if num_simulations < 2 or num_steps <= 0:
        raise ValueError("num_simulations must be >= 2 and num_steps > 0.")

    rng = np.random.default_rng(random_seed)

    dt = T / num_steps
    discount = np.exp(-r * T)
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    discounted_payoffs = np.zeros(num_simulations)

    for i in range(num_simulations):
        terminal_price = S
        minimum_price = S

        for _ in range(num_steps):
            z = rng.standard_normal()
            terminal_price *= np.exp(drift + diffusion * z)
            minimum_price = min(minimum_price, terminal_price)

        if minimum_price <= H:
            payoff = max(K - terminal_price, 0.0)
            discounted_payoffs[i] = discount * payoff

    price = np.mean(discounted_payoffs)
    variance = np.var(discounted_payoffs, ddof=1)
    standard_error = np.sqrt(variance / num_simulations)

    z_critical = norm.ppf(0.975)
    ci_lower = price - z_critical * standard_error
    ci_upper = price + z_critical * standard_error

    return {
        "price": float(price),
        "variance": float(variance),
        "standard_error": float(standard_error),
        "confidence_interval": (
            float(ci_lower),
            float(ci_upper),
        ),
        "num_simulations": num_simulations,
        "num_steps": num_steps,
    }
