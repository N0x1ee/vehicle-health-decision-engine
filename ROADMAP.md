# Vehicle Health Decision Engine — Roadmap

**Core hypothesis:** we can turn noisy telemetry into a trustworthy maintenance decision: Continue / Monitor / Inspect Soon / High Priority.
**Success test:** the engine beats simple threshold alerts on false alerts and detection lead time, and explains why it decided what it did.

Roles: **R1** ML/Data · **R2** Backend · **R3** Frontend · **R4** Hardware/Integration · **R5** Product/Research

## Built (Oct 24–26)
| Area | What's in the repo |
|---|---|
| Engine | Vehicle-specific baseline, data-quality rules (missing, frozen, warm-up, engine-off), persistence, multi-signal check, risk + confidence, explanations |
| Trips module | Trips, engine on/off events, overspeeding (> 80 km/h for 30 s+), daily totals |
| Simulator | 10 labelled scenarios, engine-off stops, normal/aggressive drivers |
| Evaluation | Engine vs thresholds, ablation vs Isolation Forest, Scania cost-weighted benchmark script |
| Backend | FastAPI + SQLite persistence, ingestion, calibration, trips endpoint, demo fleet |
| Frontend | Fleet dashboard: evidence charts, Today panel, engine chips, speed vs limit |
| Real data | OBD log importer (Torque Pro / Car Scanner) + calibrate / assess / trips CLI |
| Discovery | Interview script, interview log, competitor map |

## Today, Mon Oct 26 (everything left, same day)
| Time (IST) | Task | Owner |
|---|---|---|
| 17:00–17:15 | Commit + push the trips module; put names on roles; confirm the submission deadline | R2, all |
| 17:00–17:45 | Scania benchmark running (`python -m eval.scania_benchmark --download`) | R1 |
| 17:00–17:45 | Buy an ELM327 adapter at a local car-accessory shop; install Torque Pro / Car Scanner | R4 |
| 17:15–20:30 | 10 phone interviews (2 per person) with fleet operators | R5 + all |
| 18:00–19:30 | Real drives: 3 × 20 min healthy + 1 × 15 min holdout + dropout test | R4 |
| 19:30–20:15 | Import, calibrate, assess, trips; tune if needed | R1, R2 |
| 20:30–20:45 | Go / no-go checkpoint + pilot commitment from an operator (starts tomorrow) | All |
| 19:00–23:00 | Report, pitch deck, demo video (Claude drafts; team fills real numbers) | All |
| 23:30 | Submission-ready | All |

## Go / no-go (20:30)
- **GO** if 3+ operators describe the pain, 1 agrees to a pilot, and the engine stays on Continue for the real healthy drive.
- **PIVOT** if operators say breakdowns are rare or their tracker already solves it, or the real data has no signal.

## Not possible today (say so honestly)
- A real pilot needs days of driving: today we get a commitment, and it runs after submission.
- One evening of drives proves the pipeline on a real car, not long-term accuracy.

## Rules
- Nothing goes in the MVP unless it helps prove the core hypothesis.
- Missing data = **DATA UNAVAILABLE**, never "healthy".
- Results on synthetic data are labelled as synthetic. The Scania benchmark ≠ real-world proof.
