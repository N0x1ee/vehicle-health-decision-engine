"""Decision engine.

Pipeline (all causal, i.e. only uses data up to time t):
  clean -> features -> vehicle baseline deviation -> per-signal anomaly flags
  -> temporal persistence (5-min window) -> multi-signal agreement
  -> risk + confidence -> decision + evidence
"""
import numpy as np
import pandas as pd

from .config import (CHANNELS, WINDOW, Z_ANOM, PERSIST_RATIO, EMERGING_RATIO, SEVERE_Z,
                     MIN_COVERAGE, MIN_VALID, HOLD, DECISIONS, ACTIONS, SAMPLES_PER_MIN, SIGNAL_COLS)
from .quality import clean, add_features
from .baseline import VehicleBaseline

RAW_OF = {"coolant_temp": "coolant_temp", "vibration": "vibration",
          "battery_v": "battery_v", "rpm_instability": "rpm"}


def _run_length(b: pd.Series) -> pd.Series:
    b = b.fillna(False).astype(bool)
    return b.groupby((~b).cumsum()).cumsum()


def timeline(df: pd.DataFrame, baseline: VehicleBaseline) -> pd.DataFrame:
    """Score every row. Returns one row per input row with decision, risk, confidence, per-channel stats."""
    raw = add_features(clean(df))
    dev = baseline.deviations(raw)
    out = pd.DataFrame(index=raw.index)
    if "timestamp" in df:
        out["timestamp"] = df["timestamp"].values

    any_data = raw[SIGNAL_COLS].notna().any(axis=1).astype(float)
    out["coverage"] = any_data.rolling(WINDOW, min_periods=1).mean()

    warm = raw["warming_up"].astype(float).rolling(WINDOW, min_periods=1).max() > 0
    out["warming_up"] = warm

    persist_cols, z_cols, ratio_cols, usable_cols, severe_cols = [], [], [], [], []
    for ch, spec in CHANNELS.items():
        out[f"{ch}_active"] = baseline.has(ch)
        if not baseline.has(ch):        # vehicle doesn't report this signal: skip, don't fake it
            for c, v in [("z", np.nan), ("resid", np.nan), ("ratio", 0.0), ("cov", 0.0), ("flat", False),
                         ("usable", False), ("persistent", False), ("persist_min", 0.0),
                         ("degraded", False), ("severe", False)]:
                out[f"{ch}_{c}"] = v
            if ch != "rpm_instability":
                out[f"{ch}_exp"] = np.nan
            continue
        z = dev[f"{ch}_z"] * spec["direction"]
        resid = dev[f"{ch}_resid"]
        valid = z.notna().astype(float)
        n_valid = valid.rolling(WINDOW, min_periods=1).sum()
        flag = (z > Z_ANOM).astype(float)
        ratio = flag.rolling(WINDOW, min_periods=1).sum() / n_valid.clip(lower=1)
        zmed = z.rolling(WINDOW, min_periods=5).median()
        cov = n_valid / WINDOW

        flat = pd.Series(False, index=raw.index)
        if ch != "rpm_instability":
            sigma = baseline.models[ch]["sigma"]
            flat = raw[RAW_OF[ch]].rolling(WINDOW, min_periods=WINDOW // 2).std() < 0.05 * sigma

        usable = (n_valid >= MIN_VALID) & ~flat
        persistent = (ratio >= PERSIST_RATIO) & usable
        # a signal missing only because the engine is warming up is not a degraded sensor
        degraded_ch = ~usable & ~warm if ch == "coolant_temp" else ~usable
        out[f"{ch}_degraded"] = degraded_ch
        out[f"{ch}_z"] = zmed
        out[f"{ch}_resid"] = resid.rolling(WINDOW, min_periods=5).median()
        out[f"{ch}_severe"] = persistent & (out[f"{ch}_resid"] * spec["direction"] >= spec["severe"])
        severe_cols.append(f"{ch}_severe")
        out[f"{ch}_ratio"] = ratio.where(usable, 0.0)
        out[f"{ch}_cov"] = cov
        out[f"{ch}_flat"] = flat
        out[f"{ch}_usable"] = usable
        out[f"{ch}_persistent"] = persistent
        out[f"{ch}_persist_min"] = _run_length(persistent) / SAMPLES_PER_MIN
        persist_cols.append(f"{ch}_persistent"); z_cols.append(f"{ch}_z")
        ratio_cols.append(f"{ch}_ratio"); usable_cols.append(f"{ch}_degraded")
        # expected value for charts
        if ch != "rpm_instability":
            out[f"{ch}_exp"] = dev[f"{ch}_exp"]
    out["rpm_instability_val"] = dev["rpm_instability_val"]

    P = out[persist_cols].astype(bool).values
    Z = out[z_cols].values
    R = out[ratio_cols].values
    n_persist = P.sum(axis=1)
    max_z = np.where(P, np.nan_to_num(Z, nan=0.0), 0).max(axis=1)
    max_ratio = R.max(axis=1)
    degraded = out[usable_cols].values.astype(bool).any(axis=1)
    unavailable = (out["coverage"] < MIN_COVERAGE).values

    any_severe = out[severe_cols].values.astype(bool).any(axis=1)
    level = np.where((n_persist >= 2) | any_severe, 3,
             np.where(n_persist == 1, 2,
             np.where((max_ratio >= EMERGING_RATIO) | degraded, 1, 0)))
    level = pd.Series(level, index=out.index).rolling(HOLD, min_periods=1).max().astype(int).values
    level = np.where(unavailable, -1, level)

    # risk score kept inside the band of its decision so the number never contradicts the label
    sev = np.select(
        [level == 0, level == 1, level == 2, level == 3],
        [np.clip(max_ratio / EMERGING_RATIO, 0, 1),
         np.clip(max_ratio / PERSIST_RATIO, 0, 1),
         np.clip((max_z - Z_ANOM) / (SEVERE_Z - Z_ANOM), 0, 1),
         np.clip(0.5 * np.clip((n_persist - 1) / 2, 0, 1) + 0.5 * np.clip((max_z - Z_ANOM) / 10, 0, 1), 0, 1)],
        default=0.0)
    risk = np.where(level >= 0, 25 * np.clip(level, 0, 3) + 5 + 19 * sev, np.nan)

    # confidence = data coverage x how clear-cut the evidence is
    with np.errstate(invalid="ignore"):
        mean_persist_ratio = np.where(n_persist > 0, (R * P).sum(axis=1) / np.maximum(n_persist, 1), 0)
    clarity = np.select(
        [level == 0, level == 1, level >= 2],
        [1 - np.clip(max_ratio / EMERGING_RATIO, 0, 1) * 0.6, np.full(len(level), 0.55),
         np.clip(mean_persist_ratio + 0.1 * (n_persist - 1), 0, 1)], default=0.0)
    cov = out["coverage"].values
    conf = np.where(level >= 0, cov * (0.55 + 0.44 * clarity), cov)
    conf = np.where(degraded & (level >= 0), conf * 0.85, conf)

    out["n_persistent"] = n_persist
    out["level"] = level
    out["decision"] = [DECISIONS[int(x)] for x in level]
    out["risk"] = np.round(risk, 1)
    out["health"] = np.round(100 - risk, 1)
    out["confidence"] = np.round(np.clip(conf, 0.05, 0.99), 2)
    return out


def explain(row: pd.Series) -> list:
    """Human-readable evidence for one timeline row."""
    ev = []
    if row["level"] == -1:
        return [f"Only {row['coverage']:.0%} of the last 5 min has telemetry — device or connection issue."]
    for ch, spec in CHANNELS.items():
        if not row[f"{ch}_active"]:
            continue
        if ch == "coolant_temp" and not row[f"{ch}_usable"] and row["warming_up"]:
            ev.append("Engine warming up — coolant checks start once it reaches operating temperature.")
            continue
        if row[f"{ch}_flat"]:
            ev.append(f"{spec['label']} sensor reading frozen for 5+ min — check sensor (signal excluded).")
            continue
        if not row[f"{ch}_usable"]:
            ev.append(f"{spec['label']}: not enough data in the last 5 min (signal excluded).")
            continue
        ratio = row[f"{ch}_ratio"]
        if ratio <= 0:
            continue
        resid = row[f"{ch}_resid"]
        if ch == "rpm_instability":
            what = f"RPM fluctuating ±{row['rpm_instability_val']:.0f} rpm (≈{resid:+.0f} rpm above this vehicle's normal)"
        else:
            word = "above" if resid > 0 else "below"
            what = f"{spec['label']} {abs(resid):.1f} {spec['unit']} {word} this vehicle's expected level for current conditions"
        if row[f"{ch}_severe"]:
            ev.append(f"{what}; SEVERE (≥{spec['severe']:g} {spec['unit']}), persistent for "
                      f"{row[f'{ch}_persist_min']:.0f} min.")
        elif row[f"{ch}_persistent"]:
            ev.append(f"{what}; persistent for {row[f'{ch}_persist_min']:.0f} min ({ratio:.0%} of last 5 min).")
        elif ratio >= EMERGING_RATIO:
            ev.append(f"{what}; emerging ({ratio:.0%} of last 5 min) — watching.")
        else:
            ev.append(f"Short {spec['label'].lower()} spikes ({ratio:.0%} of last 5 min) — transient, not persistent; no action.")
    if row["n_persistent"] >= 2:
        ev.append(f"{int(row['n_persistent'])} independent signals agree — corroborated evidence.")
    if not ev:
        ev.append("All signals within this vehicle's normal range for current conditions.")
    return ev


def assess(df: pd.DataFrame, baseline: VehicleBaseline) -> dict:
    """Decision for the latest moment in df."""
    tl = timeline(df, baseline)
    row = tl.iloc[-1]
    lvl = int(row["level"])
    return {
        "vehicle_id": baseline.vehicle_id,
        "timestamp": str(row.get("timestamp", "")),
        "decision": row["decision"],
        "level": lvl,
        "health": None if lvl < 0 else float(row["health"]),
        "risk": None if lvl < 0 else float(row["risk"]),
        "confidence": float(row["confidence"]),
        "evidence": explain(row),
        "action": ACTIONS[lvl],
    }


def threshold_alerts(df: pd.DataFrame) -> pd.Series:
    """The simple baseline system: any reading over a fixed limit = alert. Missing data = silent."""
    from .config import THRESHOLDS
    d = clean(df)
    return ((d["coolant_temp"] > THRESHOLDS["coolant_temp"]) |
            (d["vibration"] > THRESHOLDS["vibration"]) |
            (d["battery_v"] < THRESHOLDS["battery_v"]) |
            (d["rpm"] > THRESHOLDS["rpm"]))
