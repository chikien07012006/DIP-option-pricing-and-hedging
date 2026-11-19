# Computational Pricing and Dynamic Hedging of a Down-and-In Put

This repository implements pricing and dynamic hedging workflows for a down-and-in European put (DIP). It compares closed-form Black--Scholes pricing, simulation-based pricing, and learning-based methods, then evaluates their hedging performance under realistic trading frictions.

## Product

The option has strike `K`, lower barrier `H < K`, and maturity `T`. Its payoff is activated only when the underlying price reaches or falls below the barrier at an observation date:

`payoff = max(K - S_T, 0) × 1{min(S_t) ≤ H}`

The baseline model assumes that the underlying follows a Geometric Brownian Motion. Pricing uses the risk-neutral measure, while hedging experiments simulate paths under the real-world measure.

## Methods

### Pricing

- **Closed-form Black--Scholes / Reiner--Rubinstein formula:** the analytical benchmark for the barrier option.
- **Monte Carlo:** simulates GBM paths and checks the barrier condition path by path.
- **Monte Carlo with Antithetic Variates:** pairs shocks `Z` and `-Z` to reduce estimator variance.
- **MLP neural network:** learns the pricing function from synthetic data with normalized inputs `(K/S0, H/S0, T, sigma, r)`.

### Dynamic hedging

Three strategies are evaluated on identical price paths within the same self-financing accounting framework:

1. Analytical Delta, using finite differences before barrier activation and Black--Scholes put Delta afterward.
2. Monte Carlo Delta estimated with finite differences and **Common Random Numbers**.
3. PPO reinforcement learning, where an agent applies a continuous adjustment to the analytical Delta while accounting for its previous position and transaction costs.

Experiments examine the effects of rebalancing frequency, transaction costs, and volatility misspecification.

## Evaluation metrics

- **Pricing:** MAE, RMSE, relative error, confidence intervals, variance-reduction ratio, and runtime.
- **Hedging:** mean P&L, standard deviation, P&L MAE/RMSE, quantiles, 5% CVaR, turnover, transaction costs, and hedging effectiveness.

## Baseline configuration

| Parameter | Value |
| --- | ---: |
| Spot / strike / barrier | 100 / 100 / 85 |
| Maturity / observation dates | 3 months / 63 |
| Volatility / risk-free rate | 30% / 3% |
| Real-world drift | 3% |
| Transaction cost | 0, 0.1%, 0.5%, 1% |

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

See [plan.md](plan.md) for the implementation roadmap.
