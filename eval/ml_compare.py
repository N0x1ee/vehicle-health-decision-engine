"""Ablation: which part of the engine actually adds value?

Compares 5 detectors on the simulated fleet (same trips as eval.evaluate):
  1. Thresholds            fixed limits on raw readings (typical tracker)
  2. IForest-raw           Isolation Forest on raw signals, per vehicle, no persistence  ("just add ML")
  3. IForest-baseline      Isolation Forest on the vehicle-baseline deviations, no persistence
  4. IForest-baseline+P    same + 5-min persistence rule
  5. Engine                z-score baseline + persistence + multi-signal + decision (ours)

Run:  python -m eval.ml_compare [--vehicles 30]
Writes results/ml_compare.md
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from engine import VehicleBaseline, timeline, threshold_alerts, generate_trip, calibration_data, SCENARIOS
from engine.config import WINDOW, PERSIST_RATIO, SAMPLES_PER_MIN, CHANNELS
from engine.quality import clean, add_features
from eval.evaluate import degrade, events

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
RAW = ["speed", "engine_load", "ambient_temp", "coolant_temp", "vibration", "rpm", "battery_v"]
DEV = [f"{c}_z" for c in CHANNELS]
CAL_FPR = 0.005   # IForest alarm threshold = score flagging 0.5% of healthy calibration samples


class IFDetector:
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X):
        X = X[self.cols].dropna()
        self.model = IsolationForest(n_estimators=150, random_state=0).fit(X.values)
        self.cut = np.quantile(self.model.score_samples(X.values), CAL_FPR)
        return self

    def flags(self, X):
        out = pd.Series(False, index=X.index)
        ok = X[self.cols].notna().all(axis=1)
        if ok.any():
            out[ok] = self.model.score_samples(X.loc[ok, self.cols].values) < self.cut
        return out


def persist(flags):
    return flags.astype(float).rolling(WINDOW, min_periods=WINDOW // 2).mean() >= PERSIST_RATIO


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
            cal_raw = add_features(clean(cal))
            cal_dev = base.deviations(cal_raw)
            if_raw = IFDetector(RAW).fit(cal_raw)
            if_dev = IFDetector(DEV).fit(cal_dev)
            for sc, meta in SCENARIOS.items():
                df = generate_trip(vid, sc, seed=f"eval{i}")
                if cond == "noisy":
                    df = degrade(df, rng)
                raw = add_features(clean(df))
                dev = base.deviations(raw)
                dets = {
                    "Thresholds": threshold_alerts(df),
                    "IForest-raw": if_raw.flags(raw),
                    "IForest-baseline": if_dev.flags(dev),
                }
                dets["IForest-baseline+P"] = persist(dets["IForest-baseline"])
                dets["Engine"] = timeline(df, base)["level"] >= 2
                k = int(df["onset_index"].iloc[0])
                for name, a in dets.items():
                    r = {"condition": cond, "detector": name, "scenario": sc, "kind": meta["kind"],
                         "alert": bool(a.any()), "events": events(a)}
                    if k >= 0 and meta["kind"] == "fault":
                        aa = a.iloc[k:]
                        r["delay"] = aa.values.argmax() / SAMPLES_PER_MIN if aa.any() else np.nan
                    rows.append(r)
            print(f"  {cond} {vid}", end="\r")
    return pd.DataFrame(rows)


def report(df):
    order = ["Thresholds", "IForest-raw", "IForest-baseline", "IForest-baseline+P", "Engine"]
    lines = ["# Ablation: what adds value?", "",
             "Synthetic fleet, same trips as `eval_report.md`. Trip-level: alert = detector fired at least once.", "",
             "Isolation Forest is fit per vehicle on healthy calibration data; its alarm threshold flags 0.5% of "
             "healthy samples.", ""]
    for cond, d in df.groupby("condition", sort=False):
        lines += [f"## {cond}", "", "| Detector | Faults caught | False-alert trips | Median delay | "
                  "Alert events / benign trip |", "|---|---|---|---|---|"]
        for name in order:
            x = d[d.detector == name]
            f, b = x[x.kind == "fault"], x[x.kind == "benign"]
            lines.append(f"| {name} | {f.alert.mean():.0%} | {b.alert.mean():.0%} | "
                         f"{f.delay.median():.0f} min | {b.events.mean():.1f} |")
        lines.append("")
    c = df[df.condition == df.condition.iloc[0]]
    g = lambda n, k: c[(c.detector == n) & (c.kind == k)].alert.mean()
    lines += ["## Reading it (clean condition)",
              f"- **Just adding ML doesn't work:** Isolation Forest on raw signals catches {g('IForest-raw','fault'):.0%} "
              f"of faults but false-alerts on {g('IForest-raw','benign'):.0%} of normal trips: worse than thresholds.",
              f"- **The vehicle baseline alone isn't enough:** still {g('IForest-baseline','benign'):.0%} false-alert trips, "
              "because single-sample anomalies (potholes, noise) still fire.",
              f"- **Persistence kills false alerts** ({g('IForest-baseline+P','benign'):.0%}) but a multivariate IForest "
              f"score then misses {1 - g('IForest-baseline+P','fault'):.0%} of faults: one drifting signal is diluted "
              "across all dimensions.",
              f"- **Engine** (per-signal baseline z + persistence + multi-signal): {g('Engine','fault'):.0%} caught, "
              f"{g('Engine','benign'):.0%} false-alert trips. Each decision also names the signal, the deviation in "
              "real units and how long it has lasted.",
              "- **Takeaway for the report:** the value comes from vehicle-specific baselines + temporal persistence "
              "+ per-signal evidence, not from model complexity. IForest stays as a secondary detector for "
              "multi-signal patterns once real data exists.",
              "- Synthetic data: re-run on real logs before claiming any of this in the field."]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicles", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = run(a.vehicles, ("clean",) if a.quick else ("clean", "noisy"))
    md = report(res)
    with open(os.path.join(OUT, "ml_compare.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
