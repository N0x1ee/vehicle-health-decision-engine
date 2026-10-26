"""SQLite persistence for real (non-demo) vehicles: telemetry rows + learned baselines.

Default file: data/vhde.db (override with env VHDE_DB). Plain SQL, so moving to PostgreSQL later
means swapping the connection (psycopg) and the upsert syntax, nothing else.
"""
import json
import os
import sqlite3
import threading

import pandas as pd

from engine import VehicleBaseline
from engine.config import ALL_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Store:
    def __init__(self, path=None):
        self.path = path or os.environ.get("VHDE_DB") or os.path.join(ROOT, "data", "vhde.db")
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.lock = threading.Lock()
        cols = ", ".join(f"{c} REAL" for c in ALL_COLS)
        with self.lock, self.conn:
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS telemetry (vehicle_id TEXT NOT NULL, "
                              f"timestamp TEXT NOT NULL, {cols}, PRIMARY KEY (vehicle_id, timestamp))")
            self.conn.execute("CREATE TABLE IF NOT EXISTS baselines (vehicle_id TEXT PRIMARY KEY, "
                              "model TEXT NOT NULL, updated TEXT DEFAULT CURRENT_TIMESTAMP)")

    def append(self, vehicle_id: str, df: pd.DataFrame):
        rows = df.copy()
        rows["timestamp"] = pd.to_datetime(rows["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        rows["vehicle_id"] = vehicle_id
        cols = ["vehicle_id", "timestamp"] + ALL_COLS
        for c in ALL_COLS:
            if c not in rows:
                rows[c] = None
        vals = rows[cols].astype(object).where(rows[cols].notna(), None).values.tolist()
        q = f"INSERT OR REPLACE INTO telemetry ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
        with self.lock, self.conn:
            self.conn.executemany(q, vals)

    def save_baseline(self, b: VehicleBaseline):
        with self.lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO baselines (vehicle_id, model) VALUES (?, ?)",
                              (b.vehicle_id, json.dumps(b.to_dict())))

    def load_all(self) -> dict:
        """{vehicle_id: {"data": DataFrame, "baseline": VehicleBaseline|None}}"""
        with self.lock:
            tel = pd.read_sql_query("SELECT * FROM telemetry ORDER BY vehicle_id, timestamp", self.conn)
            bl = dict(self.conn.execute("SELECT vehicle_id, model FROM baselines").fetchall())
        out = {}
        for vid in sorted(set(tel["vehicle_id"]) | set(bl)):
            d = tel[tel.vehicle_id == vid].drop(columns="vehicle_id").reset_index(drop=True)
            d["timestamp"] = pd.to_datetime(d["timestamp"])
            out[vid] = {"data": d if len(d) else None,
                        "baseline": VehicleBaseline.from_dict(json.loads(bl[vid])) if vid in bl else None}
        return out

    def delete(self, vehicle_id: str):
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM telemetry WHERE vehicle_id = ?", (vehicle_id,))
            self.conn.execute("DELETE FROM baselines WHERE vehicle_id = ?", (vehicle_id,))
