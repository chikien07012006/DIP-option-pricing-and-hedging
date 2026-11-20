# Implementation Plan

## Objective

Build a reproducible workflow for pricing and dynamically hedging a down-and-in European put. The workflow will compare analytical, simulation-based, and learning-based methods under a common set of assumptions and evaluation metrics.

## Phase 1 — Simulation and pricing foundation

- [x] Implement exact discrete-time GBM path simulation.
- [x] Create a GBM visualisation with the initial price and down barrier.
- [ ] Implement the down-and-in put payoff with discrete daily barrier monitoring.
- [ ] Implement vanilla Black--Scholes put pricing and the closed-form Reiner--Rubinstein down-and-in put price.
- [ ] Implement standard Monte Carlo pricing, including standard error and 95% confidence intervals.
- [ ] Implement Monte Carlo with antithetic variates and report the variance-reduction ratio.
- [ ] Validate in--out parity on matched simulation paths: `DIP + DOP = vanilla put`.
- [ ] Compare Monte Carlo results with the closed-form analytical benchmark.

**Deliverables:** tested pricing modules, a pricing-comparison table, confidence intervals, and convergence/variance-reduction figures.

## Phase 2 — Neural-network pricing surrogate

- [ ] Generate valid synthetic contracts subject to `S0 > H` and `H < K`, using Sobol or Latin-hypercube sampling.
- [ ] Generate labels with the analytical pricing engine and split data into train, validation, and test sets.
- [ ] Train an MLP with architecture `5 → 128 → 128 → 64 → 1`, Adam optimization, MSE loss, and early stopping.
- [ ] Compare raw and normalized features.
- [ ] Report MAE, RMSE, relative error, and inference time.
- [ ] Analyse errors by moneyness, volatility, maturity, and proximity to the barrier.

**Deliverables:** reproducible dataset-generation pipeline, trained model checkpoint, learning curves, and generalisation report.

## Phase 3 — Classical dynamic hedging

- [ ] Implement self-financing portfolio accounting for stock holdings, cash, transaction costs, and terminal liquidation.
- [ ] Mark the liability as a barrier option before activation and a vanilla put after activation.
- [ ] Implement analytical Delta hedging with finite differences and daily rebalancing.
- [ ] Implement Monte Carlo Delta hedging with Common Random Numbers.
- [ ] Run paired strategy comparisons on the same real-world price paths.
- [ ] Compare daily and weekly rebalancing.
- [ ] Evaluate transaction-cost sensitivity at 0%, 0.1%, 0.5%, and 1%.
- [ ] Evaluate volatility misspecification with true volatilities of 20% and 40% versus a 30% hedging model.

**Deliverables:** terminal P&L distributions, transaction-cost and turnover summaries, CVaR/quantile statistics, and a pathwise hedge illustration for a barrier-hit path.

## Phase 4 — PPO dynamic hedging

- [ ] Build a Gymnasium environment with state `(time to maturity, log(S/K), log(S/H), barrier-hit flag, previous Delta, analytical Delta)`.
- [ ] Define continuous actions as bounded adjustments to the analytical Delta.
- [ ] Implement a reward based on self-financing P&L, transaction costs, and an optional mean--variance risk penalty.
- [ ] Train PPO with Stable-Baselines3 across multiple random seeds.
- [ ] Select the policy on validation paths and evaluate the deterministic policy on fresh test paths.
- [ ] Compare PPO with analytical-Delta and Monte-Carlo-Delta hedging.

**Deliverables:** PPO checkpoint, training curves, and comparative P&L, tail-risk, turnover, and cost analysis.

## Final validation and reporting

- [ ] Use identical initial premiums, price paths, monitoring schedules, and accounting rules in all hedge comparisons.
- [ ] Report pricing accuracy, statistical efficiency, and runtime for every pricing method.
- [ ] Report mean P&L, standard deviation, MAE/RMSE, 1%/5%/50%/95%/99% quantiles, 5% CVaR, turnover, and total transaction costs for every hedging strategy.
- [ ] Keep negative results: PPO is an experimental extension and does not need to outperform classical Delta hedging for the project to be complete.
