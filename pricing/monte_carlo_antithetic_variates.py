
import numpy as np
from scipy.stats import norm


def monte_carlo_av_dip(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    num_pairs: int = 100_000,
    num_steps: int = 63,
    random_seed: int = 42,
) -> dict:

    if S <= 0 or K <= 0 or H <= 0:
        raise ValueError("S, K, H must be positive.")

    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")

    if H >= K:
        raise ValueError("This implementation assumes H < K.")

    if num_pairs < 2 or num_steps <= 0:
        raise ValueError("num_pairs must be >= 2 and num_steps > 0.")

    rng = np.random.default_rng(random_seed)

    dt = T / num_steps
    discount = np.exp(-r * T)

    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Store average discounted payoff of each pair
    pair_payoffs = np.zeros(num_pairs)

    # Outer loop: Generate M antithetic pairs
    for i in range(num_pairs):

        # Initial stock prices for both paths
        S_plus = S
        S_minus = S

        # Track minimum stock prices
        min_plus = S
        min_minus = S

        # Inner loop: Simulate N daily steps
        for j in range(num_steps):

            # Generate Gaussian shock
            z = rng.standard_normal()

            # Update the two antithetic paths
            S_plus *= np.exp(drift + diffusion * z)
            S_minus *= np.exp(drift - diffusion * z)

            # Update minimum stock prices
            min_plus = min(min_plus, S_plus)
            min_minus = min(min_minus, S_minus)

        # Calculate terminal payoffs
        payoff_plus = max(K - S_plus, 0.0)
        payoff_minus = max(K - S_minus, 0.0)

        # Apply down-and-in barrier condition
        if min_plus <= H:
            value_plus = discount * payoff_plus
        else:
            value_plus = 0.0

        if min_minus <= H:
            value_minus = discount * payoff_minus
        else:
            value_minus = 0.0

        # Average the two antithetic payoffs
        pair_payoffs[i] = (value_plus + value_minus) / 2.0

    # Monte Carlo estimator
    price = np.mean(pair_payoffs)

    # Sample variance across pair averages
    variance = np.var(pair_payoffs, ddof=1)

    # Standard error of estimated price
    standard_error = np.sqrt(variance / num_pairs)

    # 95% confidence interval
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
        "num_pairs": num_pairs,
        "num_paths": 2 * num_pairs,
        "num_steps": num_steps,
    }
