"""Deterministic data-quality rules (run before any modelling)."""
import numpy as np
import pandas as pd

from .config import VALID_RANGE, ALL_COLS, ENGINE_ON_RPM

WARMUP_TEMP = 65.0     # °C


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numbers and blank out physically impossible values."""
    df = df.copy()
    for col in ALL_COLS:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
        lo, hi = VALID_RANGE[col]
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    # engine off: charging voltage is meaningless (battery rests ~12.5 V)
    off = df["rpm"] < ENGINE_ON_RPM
    df.loc[off, "battery_v"] = np.nan
    # cold engine warming up: coolant far below operating temp is normal, not a fault
    df["warming_up"] = df["coolant_temp"] < WARMUP_TEMP
    df.loc[df["warming_up"], "coolant_temp"] = np.nan
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Context features (causal: only use past + present)."""
    df = df.copy()
    df["load_ema"] = df["engine_load"].ewm(alpha=0.08, adjust=False, ignore_na=True).mean()
    return df
