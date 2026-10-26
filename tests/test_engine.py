import os
import tempfile
import warnings
import pytest

os.environ.setdefault("VHDE_DB", os.path.join(tempfile.mkdtemp(), "test.db"))   # never touch the real DB
from fastapi.testclient import TestClient

from engine import VehicleBaseline, assess, timeline, threshold_alerts, generate_trip, calibration_data

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def base():
    return VehicleBaseline("T01").fit(calibration_data("T01"))


def end(base, sc):
    return assess(generate_trip("T01", sc, seed="test"), base)


def test_normal_continue(base):
    assert end(base, "normal")["decision"] == "Continue"


def test_pothole_not_escalated_but_threshold_fires(base):
    df = generate_trip("T01", "pothole", seed="test")
    assert (timeline(df, base)["level"] < 2).all()
    assert threshold_alerts(df).any()


def test_hill_climb_explained_by_load(base):
    assert (timeline(generate_trip("T01", "hill_climb", seed="test"), base)["level"] < 2).all()


@pytest.mark.parametrize("sc", ["cooling_fault", "mount_wear", "overheat_severe", "battery_degradation"])
def test_faults_escalate(base, sc):
    r = end(base, sc)
    assert r["level"] >= 2
    assert any("persistent" in e for e in r["evidence"])


def test_multi_signal_high_priority(base):
    r = end(base, "overheat_severe")
    assert r["decision"] == "High Priority"
    assert any("signals agree" in e for e in r["evidence"])


def test_dropout_never_healthy(base):
    r = end(base, "sensor_dropout")
    assert r["decision"] == "DATA UNAVAILABLE"
    assert r["health"] is None


def test_stuck_sensor_flagged(base):
    r = end(base, "stuck_sensor")
    assert r["decision"] == "Monitor"
    assert any("frozen" in e for e in r["evidence"])


def test_baseline_roundtrip(base, tmp_path):
    p = tmp_path / "b.json"
    base.save(p)
    b2 = VehicleBaseline.load(p)
    assert end(b2, "cooling_fault")["decision"] == end(base, "cooling_fault")["decision"]


def test_api_smoke():
    from backend.main import app
    with TestClient(app) as c:
        f = c.get("/api/fleet").json()
        assert len(f["vehicles"]) == 12
        c.post("/api/demo/advance", json={"minutes": 60})
        f = c.get("/api/fleet").json()
        assert f["vehicles"][0]["decision"] == "High Priority"
        v = c.get("/api/vehicles/V03").json()
        assert "series" in v and len(v["series"]["t"]) > 0
        # real-mode ingestion
        rows = generate_trip("R01", "normal", minutes=120).drop(columns=["vehicle_id", "scenario", "fault_active", "onset_index"])
        rows["timestamp"] = rows["timestamp"].astype(str)
        r = c.post("/api/telemetry", json={"vehicle_id": "R01", "rows": rows.to_dict("records")}).json()
        assert r["decision"] == "CALIBRATING"
        assert c.post("/api/vehicles/R01/calibrate").json()["ok"]
        assert c.get("/api/vehicles/R01").json()["decision"] == "Continue"
    # restart: real vehicle + baseline must come back from the database
    with TestClient(app) as c:
        assert c.get("/api/vehicles/R01").json()["decision"] == "Continue"
        assert c.delete("/api/vehicles/R01").json()["ok"]
