
import numpy as np


def simulate_gbm(
    spot_price: float,
    maturity: float,
    volatility: float,
    drift: float,
    num_steps: int = 63,
    num_paths: int = 1,
    random_seed: int = 42,
):
    dt = maturity / num_steps

    rng = np.random.default_rng(random_seed)

    # Initialize price matrix
    # Each row represents one simulated path
    # Each column represents one trading day
    prices = np.zeros((num_paths, num_steps + 1))

    # Initial stock price
    prices[:, 0] = spot_price

    # Simulate each time step
    for t in range(num_steps):

        # Generate random shocks for all paths
        Z = rng.standard_normal(num_paths)

        # Exact GBM formula
        prices[:, t + 1] = prices[:, t] * np.exp(
            (drift - 0.5 * volatility**2) * dt
            + volatility * np.sqrt(dt) * Z
        )

    return prices
