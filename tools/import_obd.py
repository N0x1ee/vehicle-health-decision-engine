"""Import OBD-II app logs (Torque Pro, Car Scanner, OBD Auto Doctor, generic CSV) into engine format.

Usage (from repo root):
  # 1. convert + check what signals your car gives
  python -m tools.import_obd logs/drive1.csv --vehicle MYCAR

  # 2. learn the car's normal behaviour from 1+ healthy drives (saves baselines/MYCAR.json)
  python -m tools.import_obd logs/drive1.csv logs/drive2.csv --vehicle MYCAR --calibrate

  # 3. assess a new drive with the saved baseline
  python -m tools.import_obd logs/drive3.csv --vehicle MYCAR --assess

  # optional: also send it to the running dashboard
  ... --push http://127.0.0.1:8000

Handles: wide CSVs (one column per sensor) and long CSVs (SECONDS/PID/VALUE rows, e.g. Car Scanner),
',' or ';' separators, decimal commas, °F / mph / m/s units, phone accelerometer -> vibration,
engine-off rows dropped, 10 s resampling.
"""
import argparse
import json
import math
import os
import re
import sys
import urllib.request
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import VehicleBaseline, timeline, explain          # noqa: E402
from engine.config import SAMPLE_SEC, ACTIONS                  # noqa: E402

# first match wins; matched against lower-cased column/PID names
ALIASES = {
    "speed": ["speed (obd)", "vehicle speed", "obd speed", "speed"],
    "engine_load": ["calculated engine load", "engine load", "calculated load", "load value", "load"],
    "ambient_temp": ["ambient air temp", "ambient air temperature", "ambient temp", "ambient", "outside temp"],
    "coolant_temp": ["engine coolant temperature", "coolant temperature", "coolant temp", "coolant"],
    "rpm": ["engine rpm", "engine speed", "rpm"],
    "battery_v": ["control module voltage", "voltage (control module)", "module voltage", "battery voltage",
                  "obd voltage", "adapter voltage", "voltage"],
    "accel_total": ["acceleration sensor(total)", "acceleration sensor (total)", "acceleration total",
                    "g(calibrated)", "total acceleration"],
    "accel_x": ["acceleration sensor(x axis)", "acceleration sensor (x", "accel x", "acceleration x", "g(x)"],
    "accel_y": ["acceleration sensor(y axis)", "acceleration sensor (y", "accel y", "acceleration y", "g(y)"],
    "accel_z": ["acceleration sensor(z axis)", "acceleration sensor (z", "accel z", "acceleration z", "g(z)"],
}
EXCLUDE = {"speed": ["gps", "engine"], "rpm": ["turbo", "fan"], "coolant_temp": [" 2"],
           "engine_load": ["absolute"], "battery_v": ["o2", "sensor"]}
TIME_COLS = ["device time", "gps time", "timestamp", "time", "date", "seconds", "sec", "elapsed"]


def _norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _match(columns):
    found, used = {}, set()
    for target, aliases in ALIASES.items():
        for a in aliases:
            hit = next((c for c in columns if c not in used and a in _norm(c)
                        and not any(x in _norm(c).replace(a, "") for x in EXCLUDE.get(target, []))), None)
            if hit is not None:
                found[target] = hit
                used.add(hit)
                break
    return found


def _to_num(s):
    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype(str).str.strip().str.replace(",", ".", regex=False).replace({"-": None, "": None, "nan": None})
    return pd.to_numeric(s, errors="coerce")


def _read(path):
    df = pd.read_csv(path, sep=None, engine="python", encoding_errors="ignore")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    cols = {_norm(c): c for c in df.columns}
    if "pid" in cols and "value" in cols:                      # long format -> wide
        tcol = next((cols[c] for c in TIME_COLS if c in cols), None)
        if tcol is None:
            raise ValueError("long-format log without a time column")
        df["_v"] = _to_num(df[cols["value"]])
        df = df.pivot_table(index=tcol, columns=cols["pid"], values="_v", aggfunc="mean").reset_index()
        df.columns = [str(c) for c in df.columns]
    return df


def _timestamps(df, start):
    cols = {_norm(c): c for c in df.columns}
    for c in TIME_COLS:
        if c in cols:
            raw = df[cols[c]]
            num = _to_num(raw)
            if num.notna().mean() > 0.9:                          # seconds / millis since start or epoch
                num = num - num.min() if num.max() < 1e9 else num
                unit = "ms" if num.max() > 1e11 or (num.diff().median() or 0) > 100 else "s"
                base = pd.Timestamp(start) if num.max() < 1e9 else pd.Timestamp(0)
                return base + pd.to_timedelta(num, unit=unit)
            ts = pd.to_datetime(raw, errors="coerce", format="mixed", dayfirst=True)
            if ts.notna().mean() > 0.9:
                return ts
    raise ValueError("no usable time column found (looked for: " + ", ".join(TIME_COLS) + ")")


def load_log(path, vehicle_id, start="2026-01-01 00:00"):
    """Read one app log and return engine-format rows at 10 s resolution + a report dict."""
    df = _read(path)
    found = _match([c for c in df.columns])
    out = pd.DataFrame({"timestamp": _timestamps(df, start)})
    for target, col in found.items():
        s = _to_num(df[col])
        name = _norm(col)
        if target in ("coolant_temp", "ambient_temp") and ("°f" in name or "(f)" in name or "fahrenheit" in name):
            s = (s - 32) * 5 / 9
        if target == "speed":
            if "mph" in name:
                s = s * 1.609
            elif "m/s" in name or "meters/second" in name:
                s = s * 3.6
        out[target] = s.values
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")

    # phone accelerometer -> vibration (std of |a| within each 10 s bin)
    if "accel_total" not in out and all(c in out for c in ("accel_x", "accel_y", "accel_z")):
        out["accel_total"] = np.sqrt(out["accel_x"] ** 2 + out["accel_y"] ** 2 + out["accel_z"] ** 2)
    out = out.set_index("timestamp")
    num = [c for c in ["speed", "engine_load", "ambient_temp", "coolant_temp", "rpm", "battery_v"] if c in out]
    res = out[num].resample(f"{SAMPLE_SEC}s").mean()
    if "accel_total" in out:
        a = out["accel_total"]
        if a.median() > 5:                                   # m/s² -> g
            a = a / 9.81
        res["vibration"] = a.resample(f"{SAMPLE_SEC}s").std()
    res = res.reset_index()

    # drop long all-empty gaps (between separate drives) and engine-off rows
    empty = res.drop(columns="timestamp").isna().all(axis=1)
    run = empty.groupby((~empty).cumsum()).transform("sum")
    res = res[~(empty & (run > 30))]
    if "rpm" in res:
        res = res[~(res["rpm"] < 300)]
    for c in ["speed", "engine_load", "ambient_temp", "coolant_temp", "vibration", "rpm", "battery_v"]:
        if c not in res:
            res[c] = np.nan
    res["vehicle_id"] = vehicle_id
    res = res.reset_index(drop=True)

    report = {"file": os.path.basename(path), "rows_10s": len(res),
              "minutes": round(len(res) * SAMPLE_SEC / 60, 1),
              "matched": {k: v for k, v in found.items() if not k.startswith("accel")},
              "vibration_from": "phone accelerometer" if res["vibration"].notna().any() else None,
              "missing": [c for c in ["speed", "engine_load", "ambient_temp", "coolant_temp", "rpm",
                                      "battery_v", "vibration"] if res[c].notna().mean() < 0.5]}
    return res, report


def _push(url, vehicle_id, df, calibrate):
    rows = df.copy()
    rows["timestamp"] = rows["timestamp"].astype(str)
    rows = rows.drop(columns=["vehicle_id"])
    recs = [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            for r in rows.to_dict("records")]
    for i in range(0, len(recs), 2000):
        body = json.dumps({"vehicle_id": vehicle_id, "rows": recs[i:i + 2000]}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/api/telemetry", body, {"Content-Type": "application/json"})
        urllib.request.urlopen(req).read()
    if calibrate:
        req = urllib.request.Request(url.rstrip("/") + f"/api/vehicles/{vehicle_id}/calibrate", b"", method="POST")
        print("dashboard calibrate:", urllib.request.urlopen(req).read().decode())
    print(f"pushed {len(recs)} rows to {url}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--vehicle", required=True)
    ap.add_argument("--calibrate", action="store_true", help="treat these drives as healthy; save baseline")
    ap.add_argument("--assess", action="store_true", help="assess with saved baseline")
    ap.add_argument("--push", metavar="URL", help="also send rows to the running API")
    a = ap.parse_args(argv)

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    frames = []
    for f in a.files:
        df, rep = load_log(f, a.vehicle)
        frames.append(df)
        print(f"\n{rep['file']}: {rep['minutes']} min of running data ({rep['rows_10s']} rows)")
        for k, v in rep["matched"].items():
            print(f"   {k:13s} <- {v}")
        if rep["vibration_from"]:
            print(f"   vibration     <- {rep['vibration_from']}")
        if rep["missing"]:
            print("   NOT available:", ", ".join(rep["missing"]))
    data = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    tag = "calibration" if a.calibrate else os.path.splitext(os.path.basename(a.files[-1]))[0]
    out_csv = os.path.join(ROOT, "data", f"{a.vehicle}_{tag}.csv")
    data.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")

    bpath = os.path.join(ROOT, "baselines", f"{a.vehicle}.json")
    if a.calibrate:
        os.makedirs(os.path.dirname(bpath), exist_ok=True)
        b = VehicleBaseline(a.vehicle).fit(data)
        b.save(bpath)
        print(f"baseline saved {bpath}  (signals: {', '.join(b.models)}"
              f"{'; skipped: ' + ', '.join(b.skipped) if b.skipped else ''})")
        if data["timestamp"].diff().max() and len(data) < 360:
            print("tip: ~1 hour of healthy driving gives a much steadier baseline")
    if a.assess:
        if not os.path.exists(bpath):
            sys.exit(f"no baseline at {bpath} - run with --calibrate on healthy drives first")
        b = VehicleBaseline.load(bpath)
        tl = timeline(data, b)
        tpath = os.path.join(ROOT, "results", f"{a.vehicle}_timeline.csv")
        os.makedirs(os.path.dirname(tpath), exist_ok=True)
        tl.to_csv(tpath, index=False)
        row = tl.iloc[-1]
        print(f"\n=== {a.vehicle}: {row['decision']}  (health {row['health']}, confidence {row['confidence']:.0%}) ===")
        for e in explain(row):
            print(" -", e)
        print("Action:", ACTIONS[int(row["level"])])
        print("Decisions over the drive:", tl["decision"].value_counts().to_dict())
        print(f"timeline saved {tpath}")
    if a.push:
        _push(a.push, a.vehicle, data, a.calibrate)


if __name__ == "__main__":
    main()
