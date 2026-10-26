"""Central configuration for the Vehicle Health Decision Engine."""

SAMPLE_SEC = 10                      # one telemetry row every 10 s
WINDOW = 30                          # 30 samples = 5 min evaluation window
SAMPLES_PER_MIN = 60 // SAMPLE_SEC

# Raw telemetry columns
CONTEXT_COLS = ["speed", "engine_load", "ambient_temp"]
SIGNAL_COLS = ["coolant_temp", "vibration", "rpm", "battery_v"]
ALL_COLS = CONTEXT_COLS + SIGNAL_COLS

# Physical sanity ranges: values outside are treated as invalid (NaN)
VALID_RANGE = {
    "speed": (0, 200), "engine_load": (0, 100), "ambient_temp": (-20, 60),
    "coolant_temp": (-20, 140), "vibration": (0, 10), "rpm": (0, 8000), "battery_v": (6, 18),
}

# Channels the engine scores. direction: +1 = high is bad, -1 = low is bad
# severe: deviation (in real units, from this vehicle's expected level) at which ONE persistent
#         signal alone justifies High Priority
CHANNELS = {
    "coolant_temp":    {"label": "Coolant temperature", "unit": "°C",  "direction": +1, "severe": 8.0},
    "vibration":       {"label": "Vibration",           "unit": "g",   "direction": +1, "severe": 0.5},
    "battery_v":       {"label": "Charging voltage",    "unit": "V",   "direction": -1, "severe": 1.0},
    "rpm_instability": {"label": "RPM instability",     "unit": "rpm", "direction": +1, "severe": 120.0},
}

# Context features used to predict each signal's *expected* value for this vehicle
BASELINE_FEATURES = {
    "coolant_temp": ["load_ema", "ambient_temp"],
    "vibration": ["speed"],
    "rpm": ["speed"],
    "battery_v": [],
}

# Detection thresholds
Z_ANOM = 3.0            # |z| above this = anomalous sample
PERSIST_RATIO = 0.6     # >=60% of window anomalous = persistent
EMERGING_RATIO = 0.3    # >=30% = emerging (Monitor)
SEVERE_Z = 8.0          # used only to scale the risk number inside a band
MIN_COVERAGE = 0.5      # <50% of window has data = DATA UNAVAILABLE
MIN_VALID = 10          # need at least this many valid samples in window to judge
RPM_STD_WIN = 12        # rolling window for RPM instability (2 min)
HOLD = 6                # hold a raised decision for 1 min to stop flicker

DECISIONS = {
    -1: "DATA UNAVAILABLE",
    0: "Continue",
    1: "Monitor",
    2: "Inspect Soon",
    3: "High Priority",
}
ACTIONS = {
    -1: "Telemetry missing: status unknown. Check device/connection. Do NOT treat as healthy.",
    0: "Continue normal operation.",
    1: "Keep operating; re-check at end of shift.",
    2: "Schedule inspection before the next long trip (within 1–2 days).",
    3: "Inspect before next trip; consider taking the vehicle off the road.",
}

# Simple threshold system we compare against (what most basic trackers do)
THRESHOLDS = {"coolant_temp": 105.0, "vibration": 1.5, "battery_v": 12.4, "rpm": 4500.0}
