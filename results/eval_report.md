# Evaluation: Decision Engine vs Simple Thresholds

**Synthetic data** from `engine/simulator.py`. This shows the pipeline logic works. It is NOT proof of real-world accuracy. Re-run on real OBD logs in week 4.

Caveat: the simulator's physics is close to the baseline model's structure (linear load/speed/ambient effects), so these numbers are an upper bound. Real vehicles add gear shifts, thermostat behaviour, road variety and sensor quirks.

Alert = engine decision is Inspect Soon or High Priority · threshold = any reading past a fixed limit.

## Condition: clean  (30 vehicles × 10 scenarios)

| Metric | Engine | Thresholds |
|---|---|---|
| Fault trips detected (recall) | 100% | 78% |
| False-alert rate on benign trips | 1% | 39% |
| Precision (trip level) | 99% | 66% |
| F1 | 1.00 | 0.72 |
| Median detection delay after fault onset | 13 min | 63 min |
| Alert events per fault trip (noise) | 1.0 | 7.0 |
| Alert events per benign trip | 0.01 | 1.64 |
| Device dropout flagged as DATA UNAVAILABLE | 100% | 0% (shows OK) |

Per scenario (share of trips alerted · median delay):

| Scenario | Kind | Engine alerted | Thresholds alerted | Engine delay | Threshold delay | Most common final decision |
|---|---|---|---|---|---|---|
| normal | benign | 0% | 0% | – | – | Continue |
| pothole | benign | 3% | 100% | – | – | Continue |
| hill_climb | benign | 0% | 47% | – | – | Continue |
| hot_day | benign | 0% | 10% | – | – | Continue |
| cooling_fault | fault | 100% | 100% | 14 min | 61 min | High Priority |
| mount_wear | fault | 100% | 10% | 5 min | 80 min | High Priority |
| overheat_severe | fault | 100% | 100% | 12 min | 53 min | High Priority |
| battery_degradation | fault | 100% | 100% | 14 min | 69 min | High Priority |
| sensor_dropout | data | 0% | 0% | – | – | DATA UNAVAILABLE |
| stuck_sensor | data | 0% | 0% | – | – | Monitor |

## Condition: noisy  (30 vehicles × 10 scenarios)

| Metric | Engine | Thresholds |
|---|---|---|
| Fault trips detected (recall) | 100% | 77% |
| False-alert rate on benign trips | 2% | 41% |
| Precision (trip level) | 98% | 65% |
| F1 | 0.99 | 0.70 |
| Median detection delay after fault onset | 14 min | 62 min |
| Alert events per fault trip (noise) | 1.2 | 17.1 |
| Alert events per benign trip | 0.02 | 2.27 |
| Device dropout flagged as DATA UNAVAILABLE | 100% | 0% (shows OK) |

Per scenario (share of trips alerted · median delay):

| Scenario | Kind | Engine alerted | Thresholds alerted | Engine delay | Threshold delay | Most common final decision |
|---|---|---|---|---|---|---|
| normal | benign | 3% | 0% | – | – | Continue |
| pothole | benign | 3% | 100% | – | – | Continue |
| hill_climb | benign | 0% | 50% | – | – | Continue |
| hot_day | benign | 0% | 13% | – | – | Continue |
| cooling_fault | fault | 100% | 100% | 14 min | 60 min | High Priority |
| mount_wear | fault | 100% | 7% | 11 min | 80 min | High Priority |
| overheat_severe | fault | 100% | 100% | 12 min | 52 min | High Priority |
| battery_degradation | fault | 100% | 100% | 20 min | 67 min | High Priority |
| sensor_dropout | data | 0% | 0% | – | – | DATA UNAVAILABLE |
| stuck_sensor | data | 7% | 0% | 3 min | – | Monitor |
