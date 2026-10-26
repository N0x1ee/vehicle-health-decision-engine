# Vehicle Health Decision Engine — Roadmap

**Core hypothesis:** we can turn noisy telemetry into a trustworthy maintenance decision: Continue / Monitor / Inspect Soon / High Priority.
**Success test:** the engine beats simple threshold alerts on false alerts and detection lead time, and explains why it decided what it did.

Roles: **R1** ML/Data · **R2** Backend · **R3** Frontend · **R4** Hardware/Integration · **R5** Product/Research

## Built (done, Oct 24–26)
| Area | What's in the repo |
|---|---|
| Engine | Vehicle-specific baseline, data-quality rules (missing, frozen, warm-up, engine-off), persistence, multi-signal check, risk + confidence, explanations |
| Simulator | 10 labelled scenarios incl. potholes, hill climb, cooling fault, dropout, frozen sensor |
| Evaluation | Engine vs thresholds (`results/eval_report.md`), ablation vs Isolation Forest (`results/ml_compare.md`), Scania cost-weighted benchmark script |
| Backend | FastAPI + SQLite persistence, ingestion, calibration, demo fleet |
| Frontend | Fleet dashboard with evidence charts |
| Real data | OBD log importer (Torque Pro / Car Scanner) + calibrate/assess CLI |
| Discovery | Interview script, interview log, competitor map |

## Still needed (needs people or hardware, not code)
| # | Task | Owner | Due |
|---|---|---|---|
| 1 | Order an ELM327 OBD-II adapter, log 2–3 healthy drives + 1 more, run calibrate/assess | R4 | Oct 30 |
| 2 | 10+ fleet operator interviews using `docs/interview_script.md`, filled into the log | R5 + all | Nov 2 |
| 3 | Download the Scania dataset, run `python -m eval.scania_benchmark` | R1 | Oct 27 |
| 4 | Re-tune `engine/config.py` on real drives if needed, re-run evals | R1 | after #1 |
| 5 | Pilot / demo with 1 fleet operator on 2–3 vehicles | R5 | Nov 9 |
| 6 | Capstone report, pitch deck, demo video | All | after #1–#5 |

## Go / no-go
- **GO** if: 3+ operators describe the pain, one agrees to a pilot, and the engine works on real drives.
- **PIVOT** if: operators say breakdowns are rare or their tracker already solves it, or the real data has no signal.

## Rules
- Nothing goes in the MVP unless it helps prove the core hypothesis.
- Missing data = **DATA UNAVAILABLE**, never "healthy".
- Results on synthetic data are labelled as synthetic. The Scania benchmark ≠ real-world proof.
