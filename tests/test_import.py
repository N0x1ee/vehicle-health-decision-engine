"""Importer tests using fake app logs built from the simulator."""
import warnings
import numpy as np
import pandas as pd

from engine import VehicleBaseline, timeline, generate_trip, assess
from tools.import_obd import load_log

warnings.filterwarnings("ignore")


def _torque_csv(path, scenario="normal", seed="t", warmup=True, ambient=True):
    """Torque Pro style: wide, 1 row/10 s, odd column names, includes cold start + engine-off tail."""
    d = generate_trip("CAR1", scenario, seed=seed)
    if warmup:   # first 8 min: cold engine 35 -> 85 °C
        n = 48
        d.loc[:n - 1, "coolant_temp"] = np.linspace(35, 85, n)
    off = d.tail(6).copy()
    off[["rpm", "speed", "engine_load"]] = 0
    off["battery_v"] = 12.5
    d = pd.concat([d, off], ignore_index=True)
    cols = {"timestamp": "Device Time", "speed": " Speed (OBD)(km/h)", "engine_load": " Engine Load(%)",
            "coolant_temp": " Engine Coolant Temperature(°C)", "rpm": " Engine RPM(rpm)",
            "battery_v": " Voltage (Control Module)(V)"}
    if ambient:
        cols["ambient_temp"] = " Ambient air temp(°C)"
    t = d[list(cols)].rename(columns=cols)
    t["Device Time"] = d["timestamp"].dt.strftime("%d-%b-%Y %H:%M:%S.000")
    t[" GPS Speed (Meters/second)"] = d["speed"] / 3.6
    t.to_csv(path, index=False)
    return d


def test_torque_format(tmp_path):
    p = tmp_path / "torque.csv"
    _torque_csv(p)
    df, rep = load_log(p, "CAR1")
    assert rep["matched"]["speed"].strip().startswith("Speed (OBD)")
    assert set(["coolant_temp", "rpm", "battery_v", "engine_load", "ambient_temp"]) <= set(rep["matched"])
    assert "vibration" in rep["missing"]
    assert (df["rpm"] >= 300).all()                 # engine-off rows dropped
    assert 140 <= rep["minutes"] <= 151


def test_real_car_without_vibration_or_ambient(tmp_path):
    healthy = []
    for i in range(3):
        p = tmp_path / f"h{i}.csv"
        _torque_csv(p, seed=f"h{i}", ambient=False)
        healthy.append(load_log(p, "CAR1")[0])
    b = VehicleBaseline("CAR1").fit(pd.concat(healthy, ignore_index=True))
    assert "vibration" in b.skipped and not b.has("vibration")
    assert b.models["coolant_temp"]["features"] == ["load_ema"]

    p = tmp_path / "normal.csv"
    _torque_csv(p, seed="n", ambient=False)
    tl = timeline(load_log(p, "CAR1")[0], b)
    assert (tl["level"] < 2).all()                  # cold start + missing vibration must not alert
    assert tl["decision"].iloc[-1] == "Continue"

    p = tmp_path / "fault.csv"
    _torque_csv(p, "cooling_fault", seed="f", ambient=False)
    r = assess(load_log(p, "CAR1")[0], b)
    assert r["level"] >= 2


def test_car_scanner_long_format(tmp_path):
    d = generate_trip("CAR2", "normal", minutes=30)
    secs = (d["timestamp"] - d["timestamp"].iloc[0]).dt.total_seconds()
    pids = {"speed": "Vehicle speed", "rpm": "Engine RPM", "coolant_temp": "Engine coolant temperature",
            "engine_load": "Calculated engine load value", "battery_v": "Control module voltage"}
    rows = []
    for col, pid in pids.items():
        for s, v in zip(secs, d[col]):
            rows.append({"SECONDS": f"{s:.3f}".replace(".", ","), "PID": pid, "VALUE": f"{v}".replace(".", ","), "UNITS": ""})
    p = tmp_path / "carscanner.csv"
    pd.DataFrame(rows).to_csv(p, sep=";", index=False)
    df, rep = load_log(p, "CAR2")
    assert set(pids) <= set(rep["matched"])
    assert abs(df["coolant_temp"].mean() - d["coolant_temp"].mean()) < 0.5
