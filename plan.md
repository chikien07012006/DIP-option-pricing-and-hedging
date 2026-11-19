# Kế hoạch triển khai

## Mục tiêu hoàn thành

Xây dựng pipeline tái lập được để định giá và hedge một down-and-in European put, so sánh phương pháp giải tích, Monte Carlo, neural network và PPO trên cùng giả định/thước đo.

## Giai đoạn 1 — Pricing core và kiểm định

- [ ] Cài đặt payoff DIP với daily monitoring và GBM simulator dưới risk-neutral measure.
- [ ] Cài đặt vanilla Black--Scholes put và analytical DIP continuous-monitoring.
- [ ] Áp dụng BGK continuity correction cho daily monitoring.
- [ ] Cài đặt standard Monte Carlo: giá, standard error và 95% confidence interval.
- [ ] Cài đặt antithetic variates; báo cáo variance-reduction ratio.
- [ ] Xác nhận in-out parity: `DIP + DOP = vanilla put` trên cùng đường mô phỏng.
- [ ] So sánh MC daily-monitoring với benchmark BGK và kiểm tra hội tụ khi tăng tần suất monitoring.

**Đầu ra:** pricing engine đã được test, bảng accuracy/runtime, biểu đồ hội tụ và variance reduction.

## Giai đoạn 2 — Neural-network pricing surrogate

- [ ] Sinh dữ liệu tổng hợp hợp lệ (`S0 > H`, `H < K`) theo Sobol hoặc Latin Hypercube sampling.
- [ ] Sinh nhãn bằng analytical engine kèm BGK correction; tách train/validation/test là 100k/20k/20k ban đầu.
- [ ] Huấn luyện MLP `5 → 128 → 128 → 64 → 1` với Adam, MSE và early stopping.
- [ ] So sánh feature raw với normalized feature; đánh giá MAE, RMSE, relative error và inference time.
- [ ] Phân tích lỗi theo moneyness, volatility, maturity và khoảng cách tới barrier.

**Đầu ra:** checkpoint mô hình, pipeline sinh dữ liệu tái lập được và báo cáo generalization.

## Giai đoạn 3 — Classical hedging

- [ ] Cài đặt sổ cái self-financing: cash account, vị thế cổ phiếu, phí giao dịch và liquidation tại đáo hạn.
- [ ] Mark liability: DIP trước khi hit barrier, vanilla put sau khi hit.
- [ ] Cài đặt analytical Delta (central finite difference cho DIP) và daily rebalancing.
- [ ] Cài đặt Monte Carlo Delta finite difference với Common Random Numbers.
- [ ] Chạy paired comparison trên cùng các đường giá thực dưới các mức transaction cost.
- [ ] Thực hiện daily-vs-weekly rebalancing và volatility misspecification (true sigma 20%/40%, model sigma 30%).

**Đầu ra:** phân phối P&L, turnover, tổng phí, quantile/CVaR và một pathwise illustration có barrier hit.

## Giai đoạn 4 — PPO hedger

- [ ] Xây dựng Gymnasium environment với state `(time-to-maturity, log(S/K), log(S/H), hit flag, previous delta, analytical delta)`.
- [ ] Action là phần điều chỉnh liên tục quanh analytical Delta; áp dụng clipping theo giới hạn vị thế.
- [ ] Cài đặt reward accounting có transaction cost và tuỳ chọn penalty mean--variance.
- [ ] Huấn luyện PPO (Stable-Baselines3) với nhiều random seeds; chọn model trên validation paths.
- [ ] Test deterministic policy trên fresh paths và so sánh với hai hedge cổ điển.

**Đầu ra:** checkpoint PPO, learning curves và bảng so sánh P&L/tail risk/costs.

## Tiêu chí báo cáo cuối

- Giá MC phải nằm nhất quán với benchmark BGK trong confidence interval ở ngân sách mô phỏng đủ lớn.
- Tất cả chiến lược hedge dùng cùng premium ban đầu, paths, monitoring schedule và accounting convention.
- Báo cáo đầy đủ mean P&L, standard deviation, MAE/RMSE, 1/5/50/95/99% quantiles, CVaR 5%, turnover và costs.
- Không giả định PPO phải thắng; nếu không hội tụ hoặc không cải thiện, phân tích kết quả âm và giữ pricing/classical hedging là core deliverables.

## Cấu trúc đề xuất

```text
src/          # pricing, simulation, hedging và RL environment
notebooks/    # exploratory analysis và figures
tests/        # unit/integration tests
data/         # dữ liệu sinh ra cục bộ (không commit dữ liệu lớn)
models/       # checkpoints cục bộ (không commit)
reports/      # bảng và hình cho báo cáo
```
