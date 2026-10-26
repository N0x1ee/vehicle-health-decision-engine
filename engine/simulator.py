"""Synthetic fleet telemetry generator with labelled scenarios.

Each vehicle gets its own "personality" (normal temp, vibration level, idle rpm...),
so a fixed universal threshold is unfair to some vehicles. That is exactly what the
vehicle-specific baseline is meant to handle.

NOTE: synthetic data proves the pipeline logic, not real-world accuracy.
"""
import zlib
import numpy as np
import pandas as pd

from .config import SAMPLE_SEC, SAMPLES_PER_MIN

# kind: benign = should NOT alert, fault = should alert, data = should say DATA UNAVAILABLE / degraded
SCENARIOS = {
    "normal":              {"kind": "benign", "expected": "Continue",         "desc": "Normal operation"},
    "pothole":             {"kind": "benign", "expected": "Continue",         "desc": "Rough road: short vibration spikes"},
    "hill_climb":          {"kind": "benign", "expected": "Continue",         "desc": "Long loaded hill climb on a hot day: temp rises because of load"},
    "hot_day":             {"kind": "benign", "expected": "Continue",         "desc": "Very hot ambient (44–46 °C)"},
    "cooling_fault":       {"kind": "fault",  "expected": "Inspect Soon",     "desc": "Cooling system degrading: temp drifts up regardless of load"},
    "mount_wear":          {"kind": "fault",  "expected": "High Priority",    "desc": "Engine mount / misfire: vibration up + RPM unstable"},
    "overheat_severe":     {"kind": "fault",  "expected": "High Priority",    "desc": "Temp deviation + vibration + RPM instability"},
    "battery_degradation": {"kind": "fault",  "expected": "Inspect Soon",     "desc": "Alternator/battery: charging voltage drops"},
    "sensor_dropout":      {"kind": "data",   "expected": "DATA UNAVAILABLE", "desc": "OBD device disconnects"},
    "stuck_sensor":        {"kind": "data",   "expected": "Monitor",          "desc": "Coolant sensor freezes at one value"},
}


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def vehicle_profile(vehicle_id: str) -> dict:
    rng = np.random.default_rng(_seed("profile", vehicle_id))
    return {
        "temp_base": rng.uniform(84, 92),
        "temp_load_gain": rng.uniform(0.18, 0.26),
        "temp_amb_gain": rng.uniform(0.25, 0.40),
        "vib_base": rng.uniform(0.18, 0.32),
        "vib_speed_gain": rng.uniform(0.004, 0.007),
        "idle_rpm": rng.uniform(720, 850),
        "rpm_per_kmh": rng.uniform(24, 34),
        "batt_v": rng.uniform(13.9, 14.3),
    }


def _speed_profile(n, rng, top=80.0, off=None):
    """top = highest highway target speed (80 normal driver, 100 aggressive); off = engine-off rows."""
    speed = np.zeros(n)
    v, target = 0.0, 0.0
    for i in range(n):
        if rng.random() < 0.02:
            r = rng.random()
            target = 0.0 if r < 0.15 else (rng.uniform(20, 45) if r < 0.55 else rng.uniform(45, top))
        v += (target - v) * 0.15 + rng.normal(0, 1.0)
        v = max(v, 0.0)
        if off is not None and off[i]:
            v = 0.0
        speed[i] = v
    return speed


def generate_trip(vehicle_id: str, scenario: str = "normal", minutes: int = 150,
                  onset_min: int = 60, seed=0, start="2026-10-26 08:00",
                  stops=(), driver: str = "normal") -> pd.DataFrame:
    """Return one drive of telemetry (10 s rows) with ground-truth columns.

    stops  = [(start_min, length_min), ...] engine-off stops (splits the drive into trips)
    driver = "normal" (highway up to 80 km/h) or "aggressive" (up to 100 km/h -> overspeeding)
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}")
    p = vehicle_profile(vehicle_id)
    rng = np.random.default_rng(_seed(vehicle_id, scenario, seed))
    n = minutes * SAMPLES_PER_MIN
    k = onset_min * SAMPLES_PER_MIN
    t = np.arange(n)
    f = np.clip((t - k) / max(n - k, 1), 0, 1) * (t >= k)   # fault progress 0..1
    off = np.zeros(n, dtype=bool)
    for s0, ln in stops:
        off[int(s0 * SAMPLES_PER_MIN):int((s0 + ln) * SAMPLES_PER_MIN)] = True

    speed = _speed_profile(n, rng, top=100.0 if driver == "aggressive" else 80.0, off=off if off.any() else None)
    accel = np.diff(speed, prepend=speed[0]) / SAMPLE_SEC

    # occasional mild grades on every trip so the baseline learns load effects
    grade = np.zeros(n)
    for _ in range(rng.integers(1, 4)):
        s = rng.integers(0, n - 60)
        grade[s:s + rng.integers(18, 48)] = rng.uniform(10, 25)

    amb0 = rng.uniform(24, 38)
    ambient = amb0 + 2 * np.sin(t / n * np.pi) + rng.normal(0, 0.2, n)

    if scenario == "hill_climb":
        grade[k:k + 20 * SAMPLES_PER_MIN] = 45
        ambient = 40 + rng.normal(0, 0.2, n)
    if scenario == "hot_day":
        ambient = 44 + 2 * np.sin(t / n * np.pi) + rng.normal(0, 0.2, n)

    load = np.clip(18 + 0.55 * speed + 8 * np.clip(accel, 0, None) + grade + rng.normal(0, 3, n), 10, 100)

    # coolant: first-order lag towards a load/ambient-dependent target
    target = p["temp_base"] + p["temp_load_gain"] * (load - 35) + p["temp_amb_gain"] * (ambient - 30)
    T = np.zeros(n)
    T[0] = target[0]
    for i in range(1, n):
        if off[i]:   # engine off: coolant slowly cools towards ambient
            T[i] = T[i - 1] + 0.006 * (ambient[i] - T[i - 1])
        else:
            T[i] = T[i - 1] + 0.08 * (target[i] - T[i - 1]) + rng.normal(0, 0.25)
    coolant = T + rng.normal(0, 0.4, n)

    vibration = p["vib_base"] + p["vib_speed_gain"] * speed + rng.normal(0, 1, n) * (0.03 + 0.0008 * speed)
    rpm = p["idle_rpm"] + p["rpm_per_kmh"] * speed + rng.normal(0, 30, n)
    battery = p["batt_v"] + rng.normal(0, 0.08, n)

    # ---- inject scenario effects ----
    if scenario == "pothole":
        for _ in range(4):
            s = rng.integers(k, n - 5)
            vibration[s:s + rng.integers(1, 4)] += rng.uniform(1.5, 2.5)
    elif scenario == "cooling_fault":
        coolant += 20 * f
    elif scenario == "mount_wear":
        on = (t >= k)
        vibration += on * (0.15 + 0.5 * f)
        rpm += on * rng.normal(0, 1, n) * (40 + 160 * f)
    elif scenario == "overheat_severe":
        on = (t >= k)
        coolant += 22 * f
        vibration += on * 0.4 * f
        rpm += on * rng.normal(0, 1, n) * (20 + 140 * f)
    elif scenario == "battery_degradation":
        battery -= 2.0 * f

    if off.any():
        load[off] = 0.0
        rpm[off] = 0.0
        vibration[off] = 0.02
        battery[off] = 12.5 + rng.normal(0, 0.03, off.sum())

    df = pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n, freq=f"{SAMPLE_SEC}s"),
        "vehicle_id": vehicle_id,
        "speed": speed.round(1), "engine_load": load.round(1), "ambient_temp": ambient.round(1),
        "coolant_temp": coolant.round(2), "vibration": np.clip(vibration, 0.02, None).round(3),
        "rpm": rpm.round(0), "battery_v": battery.round(2),
    })

    if scenario == "sensor_dropout":
        df.loc[k:, ["speed", "engine_load", "ambient_temp", "coolant_temp", "vibration", "rpm", "battery_v"]] = np.nan
    elif scenario == "stuck_sensor":
        df.loc[k:, "coolant_temp"] = df.loc[k, "coolant_temp"]

    df["scenario"] = scenario
    df["fault_active"] = (t >= k) & (SCENARIOS[scenario]["kind"] != "benign")
    df["onset_index"] = k if SCENARIOS[scenario]["kind"] != "benign" else -1
    return df


def calibration_data(vehicle_id: str, trips: int = 3) -> pd.DataFrame:
    """Healthy history used to learn this vehicle's normal behaviour."""
    return pd.concat([generate_trip(vehicle_id, "normal", minutes=120, seed=f"cal{i}") for i in range(trips)],
                     ignore_index=True)
