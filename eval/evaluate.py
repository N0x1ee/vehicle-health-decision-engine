"""Engine vs simple thresholds on a simulated fleet.

Run:  python -m eval.evaluate            (from the repo root)
      python -m eval.evaluate --vehicles 10 --quick

Conditions:
  clean  - simulator as-is
  noisy  - 2x sensor noise, 10% random dropped readings, small sensor offset (calibration drift)
Outputs results/eval_report.md and results/trips.csv
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

from engine import VehicleBaseline, timeline, threshold_alerts, generate_trip, calibration_data, SCENARIOS
from engine.config import SAMPLES_PER_MIN

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def degrade(df, rng):
    df = df.copy()
    n = len(df)
    df["coolant_temp"] += rng.normal(0, 0.4, n) + rng.uniform(-1, 1)
    df["vibration"] = (df["vibration"] + rng.normal(0, 0.04, n)).clip(lower=0.02)
    df["battery_v"] += rng.normal(0, 0.08, n) + rng.uniform(-0.1, 0.1)
    df["rpm"] += rng.normal(0, 25, n)
    drop = rng.random(n) < 0.10
    df.loc[drop, ["coolant_temp", "vibration", "rpm", "battery_v"]] = np.nan
    return df


def events(b):
    b = b.fillna(False).astype(int)
    return int((b.diff().fillna(b.iloc[0]) == 1).sum())


def run(n_vehicles=30, conditions=("clean", "noisy")):
    rows = []
    for cond in conditions:
        for i in range(n_vehicles):
            vid = f"V{i+1:02d}"
            rng = np.random.default_rng(1000 + i)
            cal = calibration_data(vid)
            if cond == "noisy":
                cal = degrade(cal, rng)
            base = VehicleBaseline(vid).fit(cal)
            for sc, meta in SCENARIOS.items():
                df = generate_trip(vid, sc, seed=f"eval{i}")
                if cond == "noisy":
                    df = degrade(df, rng)
                    if sc == "stuck_sensor":   # a frozen sensor outputs a constant: no noise on it
                        k0 = int(df["onset_index"].iloc[0])
                        frozen = df.loc[k0:, "coolant_temp"]
                        df.loc[k0:, "coolant_temp"] = frozen.where(frozen.isna(), frozen.dropna().iloc[0])
                tl = timeline(df, base)
                eng = tl["level"] >= 2
                thr = threshold_alerts(df)
                k = int(df["onset_index"].iloc[0])
                r = {"condition": cond, "vehicle": vid, "scenario": sc, "kind": meta["kind"],
                     "final_decision": tl["decision"].iloc[-1],
                     "eng_alert": bool(eng.any()), "thr_alert": bool(thr.any()),
                     "eng_events": events(eng), "thr_events": events(thr),
                     "eng_data_unavailable": bool((tl["level"] == -1).any())}
                if k >= 0:
                    after_e, after_t = eng.iloc[k:], thr.iloc[k:]
                    r["eng_delay_min"] = after_e.values.argmax() / SAMPLES_PER_MIN if after_e.any() else np.nan
                    r["thr_delay_min"] = after_t.values.argmax() / SAMPLES_PER_MIN if after_t.any() else np.nan
                rows.append(r)
            print(f"  {cond} {vid} done", end="\r")
    return pd.DataFrame(rows)


def summarise(df):
    lines = ["# Evaluation: Decision Engine vs Simple Thresholds", "",
             "**Synthetic data** from `engine/simulator.py`. This shows the pipeline logic works. "
             "It is NOT proof of real-world accuracy. Re-run on real OBD logs in week 4.", "",
             "Caveat: the simulator's physics is close to the baseline model's structure (linear load/speed/ambient "
             "effects), so these numbers are an upper bound. Real vehicles add gear shifts, thermostat behaviour, "
             "road variety and sensor quirks.", "",
             "Alert = engine decision is Inspect Soon or High Priority · threshold = any reading past a fixed limit.", ""]
    for cond, d in df.groupby("condition", sort=False):
        ben, flt, dat = d[d.kind == "benign"], d[d.kind == "fault"], d[d.kind == "data"]
        tp_e, tp_t = flt.eng_alert.sum(), flt.thr_alert.sum()
        fp_e, fp_t = ben.eng_alert.sum(), ben.thr_alert.sum()
        prec = lambda tp, fp: tp / (tp + fp) if tp + fp else float("nan")
        rec_e, rec_t = tp_e / len(flt), tp_t / len(flt)
        f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0
        pe, pt = prec(tp_e, fp_e), prec(tp_t, fp_t)
        drop = dat[dat.scenario == "sensor_dropout"]
        lines += [f"## Condition: {cond}  ({d.vehicle.nunique()} vehicles × {d.scenario.nunique()} scenarios)", "",
                  "| Metric | Engine | Thresholds |", "|---|---|---|",
                  f"| Fault trips detected (recall) | {rec_e:.0%} | {rec_t:.0%} |",
                  f"| False-alert rate on benign trips | {fp_e/len(ben):.0%} | {fp_t/len(ben):.0%} |",
                  f"| Precision (trip level) | {pe:.0%} | {pt:.0%} |",
                  f"| F1 | {f1(pe, rec_e):.2f} | {f1(pt, rec_t):.2f} |",
                  f"| Median detection delay after fault onset | {flt.eng_delay_min.median():.0f} min | {flt.thr_delay_min.median():.0f} min |",
                  f"| Alert events per fault trip (noise) | {flt.eng_events.mean():.1f} | {flt.thr_events.mean():.1f} |",
                  f"| Alert events per benign trip | {ben.eng_events.mean():.2f} | {ben.thr_events.mean():.2f} |",
                  f"| Device dropout flagged as DATA UNAVAILABLE | {drop.eng_data_unavailable.mean():.0%} | 0% (shows OK) |",
                  "", "Per scenario (share of trips alerted · median delay):", "",
                  "| Scenario | Kind | Engine alerted | Thresholds alerted | Engine delay | Threshold delay | Most common final decision |",
                  "|---|---|---|---|---|---|---|"]
        for sc, s in d.groupby("scenario", sort=False):
            ed = f"{s.eng_delay_min.median():.0f} min" if "eng_delay_min" in s and s.eng_delay_min.notna().any() else "–"
            td = f"{s.thr_delay_min.median():.0f} min" if "thr_delay_min" in s and s.thr_delay_min.notna().any() else "–"
            lines.append(f"| {sc} | {s.kind.iloc[0]} | {s.eng_alert.mean():.0%} | {s.thr_alert.mean():.0%} | {ed} | {td} | "
                         f"{s.final_decision.mode().iloc[0]} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicles", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="clean condition only")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = run(a.vehicles, ("clean",) if a.quick else ("clean", "noisy"))
    res.to_csv(os.path.join(OUT, "trips.csv"), index=False)
    report = summarise(res)
    with open(os.path.join(OUT, "eval_report.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print("\n" + report)
