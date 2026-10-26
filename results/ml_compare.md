# Ablation: what adds value?

Synthetic fleet, same trips as `eval_report.md`. Trip-level: alert = detector fired at least once.

Isolation Forest is fit per vehicle on healthy calibration data; its alarm threshold flags 0.5% of healthy samples.

## clean

| Detector | Faults caught | False-alert trips | Median delay | Alert events / benign trip |
|---|---|---|---|---|
| Thresholds | 78% | 39% | 63 min | 1.6 |
| IForest-raw | 97% | 95% | 16 min | 22.4 |
| IForest-baseline | 100% | 93% | 5 min | 4.5 |
| IForest-baseline+P | 57% | 0% | 12 min | 0.0 |
| Engine | 100% | 1% | 13 min | 0.0 |

## noisy

| Detector | Faults caught | False-alert trips | Median delay | Alert events / benign trip |
|---|---|---|---|---|
| Thresholds | 77% | 41% | 62 min | 2.3 |
| IForest-raw | 98% | 98% | 14 min | 31.8 |
| IForest-baseline | 100% | 98% | 4 min | 12.7 |
| IForest-baseline+P | 57% | 0% | 15 min | 0.0 |
| Engine | 100% | 2% | 14 min | 0.0 |

## Reading it (clean condition)
- **Just adding ML doesn't work:** Isolation Forest on raw signals catches 97% of faults but false-alerts on 95% of normal trips: worse than thresholds.
- **The vehicle baseline alone isn't enough:** still 93% false-alert trips, because single-sample anomalies (potholes, noise) still fire.
- **Persistence kills false alerts** (0%) but a multivariate IForest score then misses 43% of faults: one drifting signal is diluted across all dimensions.
- **Engine** (per-signal baseline z + persistence + multi-signal): 100% caught, 1% false-alert trips. Each decision also names the signal, the deviation in real units and how long it has lasted.
- **Takeaway for the report:** the value comes from vehicle-specific baselines + temporal persistence + per-signal evidence, not from model complexity. IForest stays as a secondary detector for multi-signal patterns once real data exists.
- Synthetic data: re-run on real logs before claiming any of this in the field.