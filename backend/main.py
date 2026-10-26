"""FastAPI backend for the Vehicle Health Decision Engine.

Run from repo root:   uvicorn backend.main:app --reload
Open:                 http://127.0.0.1:8000

Demo mode: a 12-vehicle simulated fleet replays telemetry. POST /api/demo/advance moves the clock.
Real mode: POST /api/telemetry to push rows for any vehicle, then /api/vehicles/{id}/calibrate.
"""
import math
import os
import threading
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.store import Store

from engine import VehicleBaseline, timeline, explain, generate_trip, calibration_data, SCENARIOS
from engine.config import ACTIONS, SAMPLES_PER_MIN, CHANNELS, ALL_COLS, OVERSPEED_KMH
from engine.trips import trips, summary, overspeed_events, engine_state, engine_events

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEMO_FLEET = [("V01", "normal"), ("V02", "pothole"), ("V03", "cooling_fault"), ("V04", "hill_climb"),
              ("V05", "mount_wear"), ("V06", "hot_day"), ("V07", "battery_degradation"), ("V08", "sensor_dropout"),
              ("V09", "overheat_severe"), ("V10", "stuck_sensor"), ("V11", "normal"), ("V12", "pothole")]
DEMO_START_MIN = 50          # faults begin at minute 60
# every demo vehicle parks 08:20-08:32 (two trips); a few park again later so "Engine off" shows live
DEMO_STOPS = {"V01": [(20, 12), (80, 15)], "V06": [(20, 12), (100, 10)], "V11": [(20, 12), (65, 20)]}
DEMO_AGGRESSIVE = {"V02", "V06", "V11"}      # drivers who go past the 80 km/h limit
SORT_KEY = {"High Priority": 5, "Inspect Soon": 4, "DATA UNAVAILABLE": 3, "Monitor": 2, "Continue": 1, "CALIBRATING": 0}

app = FastAPI(title="Vehicle Health Decision Engine", version="0.1")
lock = threading.Lock()
STATE = {"vehicles": {}, "cursor": 0}
STORE = Store()


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else round(float(v), 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def reset_demo():
    vehicles = {}
    for vid, sc in DEMO_FLEET:
        vehicles[vid] = {"baseline": VehicleBaseline(vid).fit(calibration_data(vid)),
                         "trip": generate_trip(vid, sc, seed="demo", stops=DEMO_STOPS.get(vid, [(20, 12)]),
                                              driver="aggressive" if vid in DEMO_AGGRESSIVE else "normal"),
                         "scenario": sc, "demo": True, "data": None}
    # real vehicles come back from the database (they survive restarts and demo resets)
    for vid, rec in STORE.load_all().items():
        vehicles[vid] = {"baseline": rec["baseline"], "data": rec["data"], "demo": False}
    STATE["vehicles"] = vehicles
    STATE["cursor"] = DEMO_START_MIN * SAMPLES_PER_MIN


def _data(v):
    if v["demo"]:
        return v["trip"].iloc[:STATE["cursor"]]
    return v["data"]


def _ser(obj):
    """JSON-safe copy of trip/event dicts (timestamps -> 'YYYY-MM-DD HH:MM:SS')."""
    if isinstance(obj, list):
        return [_ser(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _ser(x) for k, x in obj.items()}
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    return _clean(obj)


def _today(df):
    """Rows from the latest calendar day in df (the 'today' panel)."""
    ts = pd.to_datetime(df["timestamp"])
    return df[ts.dt.date == ts.iloc[-1].date()]


def _driving(df, detail=False):
    if df is None or len(df) == 0 or "timestamp" not in df:
        return {}
    day = _today(df)
    out = {"engine": _ser(engine_state(df)), "today": _ser(summary(day))}
    if detail:
        out["trips"] = _ser(trips(day))
        out["overspeed"] = _ser(overspeed_events(day))
        out["engine_events"] = _ser(engine_events(day))
    return out


def _status(vid, v, with_series=False):
    df = _data(v)
    if v["baseline"] is None or df is None or len(df) == 0:
        out = {"vehicle_id": vid, "decision": "CALIBRATING", "level": -2, "health": None, "risk": None,
               "confidence": None, "evidence": ["Collecting healthy history to learn this vehicle's normal behaviour."],
               "action": "Keep driving normally; calibrate once ~1 hour of healthy data is in.",
               "rows": 0 if df is None else len(df)}
        out.update(_driving(df, detail=with_series))
        return out
    tl = timeline(df, v["baseline"])
    row = tl.iloc[-1]
    lvl = int(row["level"])
    out = {"vehicle_id": vid, "timestamp": str(df["timestamp"].iloc[-1]) if "timestamp" in df else None,
           "decision": row["decision"], "level": lvl,
           "health": None if lvl < 0 else _clean(row["health"]), "risk": None if lvl < 0 else _clean(row["risk"]),
           "confidence": _clean(row["confidence"]), "evidence": explain(row), "action": ACTIONS[lvl], "rows": len(df)}
    if v["demo"]:
        out["demo_scenario"] = v["scenario"]
        out["demo_scenario_desc"] = SCENARIOS[v["scenario"]]["desc"]
    out.update(_driving(df, detail=with_series))
    if with_series:
        out["series"] = _series(df, tl, v["baseline"])
    return out


def _series(df, tl, base, minutes=60):
    n = minutes * SAMPLES_PER_MIN
    d, t = df.iloc[-n:], tl.iloc[-n:]
    step = 3                                    # every 30 s
    d, t = d.iloc[::step], t.iloc[::step]
    off = t["engine_off"].astype(bool).values
    d = d.copy()
    d.loc[off, ["coolant_temp", "vibration", "battery_v"]] = np.nan   # parked: show a gap, not a "fault"
    t = t.copy()
    for c in ["coolant_temp_exp", "vibration_exp", "battery_v_exp", "rpm_instability_val"]:
        if c in t:
            t[c] = t[c].astype(float).where(~off)
    s = {"t": [str(x)[11:16] for x in d["timestamp"]], "decision": t["decision"].tolist(),
         "risk": [_clean(x) for x in t["risk"]]}
    for sig in ["coolant_temp", "vibration", "battery_v"]:
        if sig not in base.models:          # this vehicle doesn't report it
            s[sig] = s[f"{sig}_lo"] = s[f"{sig}_hi"] = [None] * len(d)
            continue
        sigma = base.models[sig]["sigma"]
        exp = t[f"{sig}_exp"]
        s[sig] = [_clean(x) for x in d[sig]]
        s[f"{sig}_lo"] = [_clean(x) for x in exp - 3 * sigma]
        s[f"{sig}_hi"] = [_clean(x) for x in exp + 3 * sigma]
    s["rpm_instability"] = [_clean(x) for x in t["rpm_instability_val"]]
    s["rpm_instability_hi"] = (base.rpm_instab["median"] + 3 * base.rpm_instab["sigma"]) if base.rpm_instab else None
    s["speed"] = [_clean(x) for x in d["speed"]]
    s["overspeed_limit"] = OVERSPEED_KMH
    s["engine_off"] = [bool(x) for x in t["engine_off"]]
    return s


# ---------------- API ----------------
@app.on_event("startup")
def _startup():
    reset_demo()


@app.get("/api/health")
def health():
    return {"ok": True, "vehicles": len(STATE["vehicles"])}


@app.get("/api/fleet")
def fleet():
    with lock:
        items = [_status(vid, v) for vid, v in STATE["vehicles"].items()]
    items.sort(key=lambda x: (-SORT_KEY.get(x["decision"], 0), -(x["risk"] or 0)))
    clock = None
    demo = [v for v in STATE["vehicles"].values() if v["demo"]]
    if demo:
        clock = str(demo[0]["trip"]["timestamp"].iloc[min(STATE["cursor"], len(demo[0]["trip"])) - 1])
    counts = {}
    for x in items:
        counts[x["decision"]] = counts.get(x["decision"], 0) + 1
    return {"clock": clock, "counts": counts, "vehicles": items}


@app.get("/api/vehicles/{vid}")
def vehicle(vid: str):
    with lock:
        v = STATE["vehicles"].get(vid)
        if not v:
            raise HTTPException(404, "unknown vehicle")
        return _status(vid, v, with_series=True)


@app.get("/api/vehicles/{vid}/trips")
def vehicle_trips(vid: str, date: Optional[str] = None):
    """Trips, engine on/off events, overspeeding and totals for one day (default: the latest day)."""
    with lock:
        v = STATE["vehicles"].get(vid)
        if not v:
            raise HTTPException(404, "unknown vehicle")
        df = _data(v)
    if df is None or len(df) == 0:
        return {"vehicle_id": vid, "date": date, "trips": [], "overspeed": [], "engine_events": [], "summary": {}}
    ts = pd.to_datetime(df["timestamp"])
    day = df[ts.dt.date == (pd.Timestamp(date).date() if date else ts.iloc[-1].date())]
    return {"vehicle_id": vid, "date": date or str(ts.iloc[-1].date()), "limit_kmh": OVERSPEED_KMH,
            "engine": _ser(engine_state(df)), "summary": _ser(summary(day)), "trips": _ser(trips(day)),
            "overspeed": _ser(overspeed_events(day)), "engine_events": _ser(engine_events(day))}


class Advance(BaseModel):
    minutes: int = 10


@app.post("/api/demo/advance")
def advance(body: Advance):
    with lock:
        max_len = max((len(v["trip"]) for v in STATE["vehicles"].values() if v["demo"]), default=0)
        STATE["cursor"] = int(min(max(STATE["cursor"] + body.minutes * SAMPLES_PER_MIN, SAMPLES_PER_MIN * 6), max_len))
    return {"cursor_min": STATE["cursor"] / SAMPLES_PER_MIN}


@app.post("/api/demo/reset")
def demo_reset():
    with lock:
        reset_demo()
    return {"ok": True}


class Telemetry(BaseModel):
    vehicle_id: str
    rows: List[dict]


@app.post("/api/telemetry")
def ingest(body: Telemetry):
    new = pd.DataFrame(body.rows)
    if "timestamp" not in new:
        raise HTTPException(400, "rows need a timestamp")
    new["timestamp"] = pd.to_datetime(new["timestamp"])
    for c in ALL_COLS:
        if c not in new:
            new[c] = np.nan
    with lock:
        v = STATE["vehicles"].setdefault(body.vehicle_id, {"baseline": None, "data": None, "demo": False})
        if v["demo"]:
            raise HTTPException(400, "demo vehicle; use a different id")
        v["data"] = new if v["data"] is None else pd.concat([v["data"], new], ignore_index=True)
        v["data"] = (v["data"].drop_duplicates("timestamp", keep="last")
                     .sort_values("timestamp").reset_index(drop=True))
        STORE.append(body.vehicle_id, new)
        return _status(body.vehicle_id, v)


@app.post("/api/vehicles/{vid}/calibrate")
def calibrate(vid: str):
    with lock:
        v = STATE["vehicles"].get(vid)
        if not v or v["demo"] or v["data"] is None:
            raise HTTPException(400, "need ingested data for a non-demo vehicle")
        try:
            v["baseline"] = VehicleBaseline(vid).fit(v["data"])
        except ValueError as e:
            raise HTTPException(400, str(e))
        STORE.save_baseline(v["baseline"])
        return {"ok": True, "samples": v["baseline"].n_samples, "signals": list(v["baseline"].models),
                "skipped": v["baseline"].skipped}


@app.delete("/api/vehicles/{vid}")
def delete_vehicle(vid: str):
    with lock:
        v = STATE["vehicles"].get(vid)
        if not v or v["demo"]:
            raise HTTPException(400, "unknown or demo vehicle")
        del STATE["vehicles"][vid]
        STORE.delete(vid)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "frontend", "index.html"))
