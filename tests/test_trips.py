"""Trips, engine on/off and overspeeding."""
import os
import tempfile
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault("VHDE_DB", os.path.join(tempfile.mkdtemp(), "test.db"))
warnings.filterwarnings("ignore")

from engine import VehicleBaseline, timeline, generate_trip, calibration_data          # noqa: E402
from engine.trips import trips, summary, overspeed_events, engine_events, engine_state  # noqa: E402


def _const(speed_kmh, minutes, rpm=2000, start="2026-10-26 10:00"):
    n = int(minutes * 6)
    return pd.DataFrame({"timestamp": pd.date_range(start, periods=n, freq="10s"),
                         "speed": np.full(n, float(speed_kmh)), "rpm": np.full(n, float(rpm)),
                         "engine_load": 40.0, "coolant_temp": 90.0, "battery_v": 14.1})


def test_engine_off_stop_splits_trips():
    df = generate_trip("T01", "normal", minutes=60, seed="t", stops=[(20, 12)])
    ts = trips(df)
    assert len(ts) == 2
    assert ts[0]["end_reason"] == "engine_off" and ts[1]["end_reason"] == "ongoing"
    assert abs(ts[0]["duration_min"] - 20) < 0.5 and abs(ts[1]["duration_min"] - 28) < 0.5
    assert [e["event"] for e in engine_events(df)] == ["engine_on", "engine_off", "engine_on"]
    assert engine_state(df.iloc[:150])["state"] == "off"


def test_distance_from_obd_speed():
    s = summary(_const(60, 30))
    assert abs(s["distance_km"] - 30) < 0.3
    assert abs(s["engine_hours"] - 0.5) < 0.01


def test_overspeed_needs_to_be_sustained():
    base = _const(70, 10)
    spike = base.copy(); spike.loc[20, "speed"] = 95                       # one 10 s reading
    assert overspeed_events(spike) == []
    long = base.copy(); long.loc[20:25, "speed"] = 95                      # 60 s over the limit
    ev = overspeed_events(long)
    assert len(ev) == 1 and ev[0]["duration_s"] == 60 and ev[0]["max_speed_kmh"] == 95
    two = base.copy(); two.loc[20:23, "speed"] = 90; two.loc[26:29, "speed"] = 90   # 40 s, 20 s gap, 40 s
    assert len(overspeed_events(two)) == 1                                  # merged into one event
    assert len(overspeed_events(long, limit=100)) == 0                       # configurable limit


def test_gap_without_rows_ends_trip():
    a = _const(50, 10)
    b = _const(50, 10, start="2026-10-26 11:00")                           # 50 min later, no rows between
    ts = trips(pd.concat([a, b], ignore_index=True))
    assert len(ts) == 2 and ts[0]["end_reason"] == "data_gap"


def test_dropout_ends_trip_as_data_lost():
    ts = trips(generate_trip("T01", "sensor_dropout", seed="t"))
    assert len(ts) == 1 and ts[0]["end_reason"] == "data_gap"


def test_parked_engine_is_not_a_fault():
    b = VehicleBaseline("T02").fit(calibration_data("T02"))
    df = generate_trip("T02", "normal", seed="t", stops=[(20, 12), (80, 15)])
    tl = timeline(df, b)
    assert (tl["level"] < 2).all()
    assert tl["engine_off"].sum() == 27 * 6
    assert tl["decision"].iloc[85 * 6] == tl["decision"].iloc[80 * 6 - 1]   # last assessment carried


def test_aggressive_driver_overspeeds():
    df = generate_trip("T03", "normal", seed="t", driver="aggressive")
    assert summary(df)["overspeed_events"] >= 1
    assert summary(df)["max_speed_kmh"] > 85


def test_api_trips_endpoint():
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        c.post("/api/demo/advance", json={"minutes": 40})
        r = c.get("/api/vehicles/V11/trips").json()
        assert r["summary"]["trips"] >= 2 and r["trips"][0]["end_reason"] == "engine_off"
        f = c.get("/api/fleet").json()
        v01 = next(v for v in f["vehicles"] if v["vehicle_id"] == "V01")
        assert v01["engine"]["state"] == "off"                              # parked 09:20-09:35
        assert any("Engine off" in e for e in v01["evidence"])
        assert "trips" in c.get("/api/vehicles/V02").json()
