# Down-and-In Put Pricing and Dynamic Hedging

A research-oriented Python project for pricing and dynamically hedging a down-and-in European put (DIP). The repository combines a closed-form barrier-option benchmark, Monte Carlo simulation, neural-network pricing, and reinforcement-learning hedging in one reproducible workflow.

## What this repository does

- Simulates underlying-price paths with Geometric Brownian Motion (GBM).
- Prices a down-and-in put with a closed-form Black--Scholes / Reiner--Rubinstein benchmark.
- Estimates prices with standard Monte Carlo and antithetic variates.
- Trains an MLP pricing surrogate on synthetically generated contracts.
- Compares analytical-Delta, Monte-Carlo-Delta, and PPO-based dynamic hedging.
- Evaluates hedging error, tail risk, turnover, and transaction costs under different market scenarios.

## Down-and-in put

The option has strike `K`, lower barrier `H < K`, and maturity `T`. It becomes active only when the underlying price reaches or falls below `H` at least once before maturity. Its payoff is

`max(K - S_T, 0) × 1{min(S_t) ≤ H}`

The simulation model assumes that the underlying follows GBM:

`dS_t = μ S_t dt + σ S_t dW_t`

Pricing uses risk-neutral drift `r`; hedging experiments can instead use the real-world drift `μ`.

## GBM simulation

The currently implemented simulator generates exact discrete GBM paths and saves the visualisation below. The dashed red line is the down barrier; a path crossing it activates the option.

![Simulated GBM paths and down barrier](results/figures/gbm_simulation.png)

Run the simulation and regenerate the figure with:

```bash
python -m stock_simulation.visualize
```

## Methodology

### Pricing

| Method | Purpose |
| --- | --- |
| Closed-form Black--Scholes / Reiner--Rubinstein | Analytical pricing benchmark for the barrier option. |
| Standard Monte Carlo | Estimates the discounted expected payoff from simulated paths. |
| Antithetic variates | Reduces Monte Carlo estimator variance by pairing shocks `Z` and `-Z`. |
| MLP regressor | Approximates prices from normalized contract and market features. |

### Dynamic hedging

All hedgers share the same simulated paths and self-financing cash-account convention.

1. **Analytical Delta:** finite-difference Delta for the barrier option before activation, then Black--Scholes put Delta after activation.
2. **Monte Carlo Delta:** finite-difference Delta with Common Random Numbers for a lower-variance bump-and-revalue estimate.
3. **PPO hedger:** a reinforcement-learning policy that adjusts the analytical Delta while accounting for its previous position and transaction costs.

## Baseline experiment

The default pricing configuration is in [`configs/pricing.yaml`](configs/pricing.yaml).

| Parameter | Value |
| --- | ---: |
| Initial spot / strike / barrier | 100 / 100 / 85 |
| Maturity | 0.25 years (3 months) |
| Monitoring dates | 63 daily observations |
| Volatility / risk-free rate | 30% / 3% |
| Real-world drift | 3% |
| Transaction-cost scenarios | 0, 0.1%, 0.5%, 1% |

## Evaluation

Pricing methods are assessed with MAE, RMSE, relative error, confidence intervals, estimator variance, and runtime. Hedging strategies are compared using terminal P&L, P&L dispersion, quantiles, 5% CVaR, transaction costs, turnover, and hedging effectiveness.

## Project layout

```text
configs/             # Pricing, hedging, and PPO experiment settings
pricing/             # Analytical, Monte Carlo, and neural-network pricing modules
stock_simulation/    # GBM path generator and plotting utility
results/figures/     # Generated charts, including the GBM simulation
data/                # Local generated datasets (not committed)
plan.md              # Implementation roadmap
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

See [plan.md](plan.md) for the implementation roadmap and validation criteria.
