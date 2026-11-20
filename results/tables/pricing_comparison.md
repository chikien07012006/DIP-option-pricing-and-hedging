| Method | Price | Bias vs reference | SE / CI 95% | Estimator variance | VRR | Runtime per pricing |
| --- | --- | --- | --- | --- | --- | --- |
| Reiner-Rubinstein (closed form) | 4.2622 | — (benchmark) | — | 0 | — | 2.8 µs |
| Standard Monte Carlo | 3.9854 | -0.2768 (vs RR) | 0.0249 / [3.9366, 4.0342] | 6.198e-04 | 1.00 | 3.07 s |
| Monte Carlo antithetic variates | 3.9852 | -0.2770 (vs RR) | 0.0214 / [3.9432, 4.0272] | 4.600e-04 | 1.35 | 2.33 s |
| Neural network (MLP) | 4.2991 | +0.0370 (vs RR) | — | — | — | 24.5 µs (0.17 µs/contract batched) |
