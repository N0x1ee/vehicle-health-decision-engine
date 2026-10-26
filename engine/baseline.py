"""Vehicle-specific baseline: what is *normal for this vehicle* under current conditions.

For each signal we fit a small linear model on healthy history:
    expected_signal = f(context)   e.g. coolant ~ load + ambient
and store a robust noise scale (MAD). Deviation = (actual - expected) / scale.
Simple, explainable, and per-vehicle.
"""
import json
import numpy as np
import pandas as pd

from .config import BASELINE_FEATURES, RPM_STD_WIN

MIN_FIT_ROWS = 100   # ≈ 17 min of driving at 10 s rows
from .quality import clean, add_features


def _robust_sigma(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    mad = np.median(np.abs(x - np.median(x)))
    return float(max(1.4826 * mad, 1e-6))


class VehicleBaseline:
    def __init__(self, vehicle_id=None):
        self.vehicle_id = vehicle_id
        self.models = {}        # signal -> {"features", "coef", "sigma"}
        self.rpm_instab = {}    # {"median", "sigma"}
        self.n_samples = 0
        self.skipped = []       # signals this vehicle doesn't report

    # ---------- fit ----------
    def fit(self, healthy: pd.DataFrame) -> "VehicleBaseline":
        """Signals or context features a vehicle doesn't report are skipped, not faked."""
        df = add_features(clean(healthy))
        self.models, self.rpm_instab, self.skipped = {}, {}, []
        for sig, feats in BASELINE_FEATURES.items():
            if df[sig].notna().sum() < MIN_FIT_ROWS:
                self.skipped.append(sig)
                continue
            feats = [c for c in feats if df.loc[df[sig].notna(), c].notna().mean() >= 0.8]
            d = df[[sig] + feats].dropna()
            if len(d) < MIN_FIT_ROWS:
                self.skipped.append(sig)
                continue
            X = np.column_stack([np.ones(len(d))] + [d[c].values for c in feats])
            coef, *_ = np.linalg.lstsq(X, d[sig].values, rcond=None)
            resid = d[sig].values - X @ coef
            self.models[sig] = {"features": feats, "coef": coef.tolist(), "sigma": _robust_sigma(resid)}
        if not any(s in self.models for s in ("coolant_temp", "vibration", "battery_v", "rpm")):
            raise ValueError(f"not enough healthy data for any signal (need {MIN_FIT_ROWS}+ rows ≈ "
                             f"{MIN_FIT_ROWS // 6} min of driving)")
        if "rpm" in self.models:
            instab = self._rpm_instability(df)
            if instab.notna().sum() >= MIN_FIT_ROWS:
                self.rpm_instab = {"median": float(np.nanmedian(instab)), "sigma": _robust_sigma(instab)}
        self.n_samples = int(len(df))
        return self

    def has(self, channel: str) -> bool:
        if channel == "rpm_instability":
            return bool(self.rpm_instab)
        return channel in self.models

    # ---------- predict ----------
    def expected(self, sig: str, df: pd.DataFrame) -> pd.Series:
        m = self.models[sig]
        out = pd.Series(m["coef"][0], index=df.index, dtype=float)
        for c, b in zip(m["features"], m["coef"][1:]):
            out = out + b * df[c]
        return out

    def _rpm_instability(self, df):
        resid = df["rpm"] - self.expected("rpm", df)
        return resid.rolling(RPM_STD_WIN, min_periods=RPM_STD_WIN // 2).std()

    def deviations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return expected values, residuals and z-scores for every channel."""
        out = pd.DataFrame(index=df.index)
        nan = pd.Series(np.nan, index=df.index)
        for sig in ["coolant_temp", "vibration", "battery_v"]:
            if sig not in self.models:
                out[f"{sig}_exp"] = out[f"{sig}_resid"] = out[f"{sig}_z"] = nan
                continue
            exp = self.expected(sig, df)
            out[f"{sig}_exp"] = exp
            out[f"{sig}_resid"] = df[sig] - exp
            out[f"{sig}_z"] = out[f"{sig}_resid"] / self.models[sig]["sigma"]
        if not self.rpm_instab:
            out["rpm_exp"] = out["rpm_instability_val"] = out["rpm_instability_resid"] = out["rpm_instability_z"] = nan
            return out
        out["rpm_exp"] = self.expected("rpm", df)
        instab = self._rpm_instability(df)
        out["rpm_instability_val"] = instab
        out["rpm_instability_resid"] = instab - self.rpm_instab["median"]
        out["rpm_instability_z"] = out["rpm_instability_resid"] / self.rpm_instab["sigma"]
        return out

    # ---------- persistence ----------
    def to_dict(self):
        return {"vehicle_id": self.vehicle_id, "models": self.models,
                "rpm_instab": self.rpm_instab, "n_samples": self.n_samples,
                "skipped": getattr(self, "skipped", [])}

    @classmethod
    def from_dict(cls, d):
        b = cls(d.get("vehicle_id"))
        b.models, b.rpm_instab, b.n_samples = d["models"], d["rpm_instab"], d["n_samples"]
        b.skipped = d.get("skipped", [])
        return b

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls.from_dict(json.load(fh))
