# Implementation Plan

## Objective

Build a reproducible workflow for pricing and dynamically hedging a down-and-in European put. The workflow will compare analytical, simulation-based, and learning-based methods under a common set of assumptions and evaluation metrics.

## Phase 1 — Simulation and pricing foundation ✅ closed

- [x] Implement exact discrete-time GBM path simulation.
- [x] Create a GBM visualisation with the initial price and down barrier.
- [x] Implement the down-and-in put payoff with discrete daily barrier monitoring (inside both Monte Carlo pricers).
- [x] Implement vanilla Black--Scholes put pricing and the closed-form Reiner--Rubinstein down-and-in put price.
- [x] Implement standard Monte Carlo pricing, including standard error and 95% confidence intervals.
- [x] Implement Monte Carlo with antithetic variates and report the variance-reduction ratio.
- [x] Compare Monte Carlo results with the closed-form analytical benchmark (`pricing/compare_methods.py`).
- [ ] *(deferred)* Validate in--out parity on matched simulation paths: `DIP + DOP = vanilla put`.
- [ ] *(deferred)* Convergence / variance-reduction figures across path budgets.
- [ ] *(deferred)* Unit tests for the pricing modules.

**Delivered:** pricing modules, pricing-comparison table (`results/tables/pricing_comparison.*`), confidence intervals, and variance-reduction ratio.

**Decisions:**
- All methods are benchmarked against the continuous-monitoring Reiner--Rubinstein price with no continuity correction. The Monte Carlo bias therefore includes the continuous-vs-daily monitoring gap (≈ −0.28 on the baseline contract).
- Variance reduction is compared at an equal path budget: 100,000 standard paths vs 50,000 antithetic pairs.

## Phase 2 — Neural-network pricing surrogate ✅ closed

- [x] Generate valid synthetic contracts subject to `S0 > H` and `H < K`, using Latin-hypercube sampling (140,000 contracts).
- [x] Generate labels with the analytical pricing engine and split data into train, validation, and test sets (100k / 20k / 20k).
- [x] Train an MLP with architecture `6 → 128 → 128 → 64 → 1` (ReLU after each hidden layer, linear output), Adam optimization, MSE loss, and early stopping.
- [x] Report MAE, RMSE, relative error, and inference time.
- [x] Analyse errors by moneyness, volatility, maturity, and proximity to the barrier.
- [x] Plot predicted vs analytical prices and the NN-implied volatility surface.
- [ ] *(deferred)* Compare raw and normalized features.

**Delivered:** dataset-generation and split scripts, trained checkpoint (`results/checkpoints/dip_mlp.pt`), learning curves, metrics (`results/tables/nn_metrics.json`), error breakdown, and figures.

**Decisions:**
- Inputs: `spot_price, strike_price, barrier_to_spot_ratio, time_to_maturity, volatility, risk_free_rate`. `barrier_level` is dropped because it equals `spot_price × barrier_to_spot_ratio`.
- Inputs are standardised with train-set mean and standard deviation; the target is the raw analytical price, with no target scaling.

## Phase 3 — Classical dynamic hedging

- [ ] Vectorise Monte Carlo pricing so that bump-and-revalue Delta is feasible inside the hedging loop.
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
- [x] Report pricing accuracy, statistical efficiency, and runtime for every pricing method.
- [ ] Report mean P&L, standard deviation, MAE/RMSE, 1%/5%/50%/95%/99% quantiles, 5% CVaR, turnover, and total transaction costs for every hedging strategy.
- [ ] Keep negative results: PPO is an experimental extension and does not need to outperform classical Delta hedging for the project to be complete.
