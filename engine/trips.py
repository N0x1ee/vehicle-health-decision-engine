"""Trips, engine on/off events and overspeeding.

Supporting module (not the core product): plain deterministic rules, as the strategy doc says
simple conditions like overspeeding don't need ML.

  engine on   = rpm >= ENGINE_ON_RPM
  trip        = a continuous run of engine-on rows; it ends when the engine turns off,
                when data stops (a gap > TRIP_GAP_MIN, or rows with no rpm), or it is still ongoing
  distance    = integral of OBD speed over time (no GPS needed)
  overspeed   = speed > OVERSPEED_KMH for at least OVERSPEED_MIN_SEC; events closer than
                OVERSPEED_MERGE_SEC are merged
"""
import numpy as np
import pandas as pd

from .config import (SAMPLE_SEC, ENGINE_ON_RPM, TRIP_GAP_MIN, IDLE_KMH, OVERSPEED_KMH,
                     OVERSPEED_MIN_SEC, OVERSPEED_MERGE_SEC)
from .quality import clean


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = clean(df)
    d["timestamp"] = pd.to_datetime(df["timestamp"]).values
    d = d.sort_values("timestamp").reset_index(drop=True)
    # seconds each row represents: time to the next row, capped (a long gap is not driving time)
    dt = d["timestamp"].diff().shift(-1).dt.total_seconds()
    d["dt"] = dt.clip(upper=3 * SAMPLE_SEC).fillna(SAMPLE_SEC)
    d["state"] = np.where(d["rpm"].isna(), "unknown", np.where(d["rpm"] >= ENGINE_ON_RPM, "on", "off"))
    return d


def _runs(mask: pd.Series):
    """(start_idx, end_idx) of each contiguous True run."""
    m = mask.fillna(False).astype(bool).values
    if not m.any():
        return []
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1))


def overspeed_events(df: pd.DataFrame, limit=OVERSPEED_KMH, min_sec=OVERSPEED_MIN_SEC,
                     merge_sec=OVERSPEED_MERGE_SEC, _prepped=None) -> list:
    d = _prepped if _prepped is not None else _prep(df)
    runs = _runs(d["speed"] > limit)
    merged = []
    for s, e in runs:
        if merged:
            gap = (d["timestamp"].iloc[s] - d["timestamp"].iloc[merged[-1][1]]).total_seconds()
            if gap <= merge_sec:
                merged[-1] = (merged[-1][0], e)
                continue
        merged.append((s, e))
    events = []
    for s, e in merged:
        seg = d.iloc[s:e + 1]
        dur = float(seg["dt"].sum())
        if dur < min_sec:
            continue
        events.append({
            "start": seg["timestamp"].iloc[0], "end": seg["timestamp"].iloc[-1] + pd.Timedelta(seconds=seg["dt"].iloc[-1]),
            "duration_s": round(dur), "max_speed_kmh": round(float(seg["speed"].max()), 1),
            "avg_speed_kmh": round(float(seg["speed"].mean()), 1), "limit_kmh": limit,
        })
    return events


def trips(df: pd.DataFrame, limit=OVERSPEED_KMH) -> list:
    """One dict per trip, oldest first."""
    d = _prep(df)
    if d.empty:
        return []
    on = d["state"] == "on"
    gap_break = d["timestamp"].diff().dt.total_seconds() > TRIP_GAP_MIN * 60
    trip_id = (on & (~on.shift(fill_value=False) | gap_break)).cumsum().where(on)
    out = []
    for tid, seg in d[on].groupby(trip_id[on]):
        s, e = seg.index[0], seg.index[-1]
        nxt = d.iloc[e + 1] if e + 1 < len(d) else None
        if nxt is None:
            end_reason = "ongoing"
        elif (nxt["timestamp"] - seg["timestamp"].iloc[-1]).total_seconds() > TRIP_GAP_MIN * 60:
            end_reason = "data_gap"
        elif nxt["state"] == "off":
            end_reason = "engine_off"
        else:
            end_reason = "data_gap"
        hours = seg["dt"] / 3600
        idle = seg["speed"] < IDLE_KMH
        events = overspeed_events(None, limit=limit, _prepped=seg.reset_index(drop=True))
        dur_min = float(seg["dt"].sum()) / 60
        km = float((seg["speed"].fillna(0) * hours).sum())
        out.append({
            "trip": int(tid), "start": seg["timestamp"].iloc[0],
            "end": seg["timestamp"].iloc[-1] + pd.Timedelta(seconds=seg["dt"].iloc[-1]),
            "end_reason": end_reason, "duration_min": round(dur_min, 1), "distance_km": round(km, 1),
            "avg_speed_kmh": round(km / (dur_min / 60), 1) if dur_min else 0.0,
            "max_speed_kmh": round(float(seg["speed"].max()), 1) if seg["speed"].notna().any() else None,
            "idle_min": round(float(seg.loc[idle, "dt"].sum()) / 60, 1),
            "overspeed_events": len(events), "overspeed_s": int(sum(ev["duration_s"] for ev in events)),
            "max_coolant_c": round(float(seg["coolant_temp"].max()), 1) if seg["coolant_temp"].notna().any() else None,
        })
    return out


def engine_events(df: pd.DataFrame) -> list:
    """engine_on / engine_off / data_lost events derived from trips."""
    ev = []
    for t in trips(df):
        ev.append({"time": t["start"], "event": "engine_on", "trip": t["trip"]})
        if t["end_reason"] == "engine_off":
            ev.append({"time": t["end"], "event": "engine_off", "trip": t["trip"]})
        elif t["end_reason"] == "data_gap":
            ev.append({"time": t["end"], "event": "data_lost", "trip": t["trip"]})
    return ev


def engine_state(df: pd.DataFrame) -> dict:
    """Current state from the latest row: on / off / unknown, and since when."""
    d = _prep(df)
    if d.empty:
        return {"state": "unknown", "since": None}
    last = d["state"].iloc[-1]
    change = d.index[d["state"] != last]
    since = d["timestamp"].iloc[change[-1] + 1] if len(change) else d["timestamp"].iloc[0]
    return {"state": last, "since": since}


def summary(df: pd.DataFrame, limit=OVERSPEED_KMH) -> dict:
    """Totals across all trips in df (e.g. one day)."""
    ts = trips(df, limit=limit)
    eng_min = sum(t["duration_min"] for t in ts)
    idle_min = sum(t["idle_min"] for t in ts)
    return {
        "trips": len(ts), "distance_km": round(sum(t["distance_km"] for t in ts), 1),
        "engine_hours": round(eng_min / 60, 2), "idle_pct": round(100 * idle_min / eng_min, 1) if eng_min else 0.0,
        "overspeed_events": sum(t["overspeed_events"] for t in ts),
        "max_speed_kmh": max((t["max_speed_kmh"] or 0 for t in ts), default=None), "limit_kmh": limit,
    }
