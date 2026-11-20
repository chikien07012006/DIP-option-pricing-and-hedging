
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stock_simulation.geometric_brownian_motion import simulate_gbm


def plot_stock_simulation(
    spot_price=100.0,
    maturity=0.25,
    volatility=0.30,
    drift=0.03,
    num_steps=63,
    num_paths=1000,
    barrier=85.0,
    random_seed=42,
):

    # Generate stock price paths
    paths = simulate_gbm(
        spot_price=spot_price,
        maturity=maturity,
        volatility=volatility,
        drift=drift,
        num_steps=num_steps,
        num_paths=num_paths,
        random_seed=random_seed,
    )

    # Trading day indices: 0, 1, ..., 63
    time_steps = np.arange(num_steps + 1)

    # Create output directory
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "results" / "figures"

    output_path = output_dir / "gbm_simulation.png"

    # Plot
    plt.figure(figsize=(12, 6))

    for i in range(num_paths):
        plt.plot(
            time_steps,
            paths[i],
            linewidth=0.6,
            alpha=0.12,
        )

    # Initial stock price
    plt.axhline(
        y=spot_price,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"Initial Price (${spot_price:.0f})",
    )

    # DIP barrier
    if barrier is not None:
        plt.axhline(
            y=barrier,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Down Barrier (${barrier:.0f})",
        )

    plt.title(
        f"Monte Carlo Stock Price Simulation: "
        f"{num_paths:,} GBM Paths",
        fontsize=15,
        fontweight="bold",
    )

    plt.xlabel("Trading Days", fontsize=12)
    plt.ylabel("Stock Price ($)", fontsize=12)

    plt.xlim(0, num_steps)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left")

    plt.tight_layout()

    # Save figure
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print(f"Simulation figure saved to: {output_path}")


if __name__ == "__main__":
    plot_stock_simulation()
