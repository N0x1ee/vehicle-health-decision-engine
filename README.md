# Vehicle Health Decision Engine

Turns noisy vehicle telemetry into one explainable decision per vehicle:
**Continue · Monitor · Inspect Soon · High Priority** (or **DATA UNAVAILABLE**, never "healthy" when data is missing).

Built for small and mid-sized commercial fleets (10–100 vehicles). It works on top of the data they already have instead of replacing their tracker. See `ROADMAP.md` for status and next steps.

## Run it (Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m pytest -q                       # tests
uvicorn backend.main:app --reload         # dashboard at http://127.0.0.1:8000
```
In the dashboard, click **Auto-play** or **+10 min** to watch faults develop. They start at 09:00 on the demo clock.

## Evaluations
```powershell
python -m eval.evaluate          # engine vs thresholds, clean + noisy  -> results/eval_report.md
python -m eval.ml_compare        # ablation vs Isolation Forest          -> results/ml_compare.md
python -m eval.scania_benchmark --download   # real truck failure data   -> results/scania_benchmark.md
```
For Scania, if `--download` fails, get the [UCI zip](https://archive.ics.uci.edu/static/public/414/ida2016challenge.zip) manually and unzip the two CSVs into `data/scania/`.

## How it decides
```
telemetry -> data quality (ranges, missing, frozen sensor, cold start, engine off)
          -> vehicle baseline (expected value for THIS vehicle given load/speed/ambient)
          -> z-score deviation per signal
          -> persistence (>=60% of last 5 min anomalous)
          -> multi-signal agreement
          -> risk + confidence -> decision + evidence
```
- 1 persistent signal → **Inspect Soon**
- 2+ persistent signals, or 1 severe one (e.g. coolant +8 °C, voltage −1 V) → **High Priority**
- Emerging (30–60% of window) or a degraded sensor → **Monitor**
- Less than 50% of the last 5 min has data → **DATA UNAVAILABLE**

All rules and limits are in `engine/config.py`.

## Results so far (synthetic, 30 vehicles × 10 scenarios)
| | Engine | Thresholds | Isolation Forest (raw) |
|---|---|---|---|
| Faults detected | 100% | 78% | 97% |
| False alerts on normal trips | 1% | 39% | 95% |
| Median detection delay | 13 min | 63 min | 16 min |
| Device dropout shown as DATA UNAVAILABLE | 100% | 0% | 0% |

The ablation shows the gain comes from **vehicle baselines + persistence + per-signal evidence**, not model complexity. Synthetic data is close to the model's structure, so treat these as an upper bound: real OBD drives are the real test.

## Real car data (OBD-II)
1. Plug an ELM327 Bluetooth adapter into the car's OBD port. Log drives with **Torque Pro** or **Car Scanner** and export them as CSV into `logs/`.
2. Check which signals your car gives: `python -m tools.import_obd logs\drive1.csv --vehicle MYCAR`
3. Learn the car's normal behaviour from healthy drives (~1 h total is ideal):
   `python -m tools.import_obd logs\drive1.csv logs\drive2.csv --vehicle MYCAR --calibrate`
4. Assess any new drive: `python -m tools.import_obd logs\drive3.csv --vehicle MYCAR --assess`
5. Add `--push http://127.0.0.1:8000` to send it to the dashboard (saved in `data/vhde.db`, survives restarts).

Signals the car doesn't report (often vibration and ambient temp) are skipped automatically. Turn on phone accelerometer logging in Torque to get vibration.

## API
| Method | Path | What |
|---|---|---|
| GET | `/api/fleet` | All vehicles, sorted by priority |
| GET | `/api/vehicles/{id}` | Decision, evidence, last-60-min series |
| POST | `/api/telemetry` | `{"vehicle_id": "...", "rows": [{"timestamp": "...", "speed": 42, "engine_load": 38, "coolant_temp": 89.5, "rpm": 1900, "battery_v": 14.1, ...}]}` |
| POST | `/api/vehicles/{id}/calibrate` | Learn baseline from ingested (healthy) data |
| DELETE | `/api/vehicles/{id}` | Remove a real vehicle and its data |
| POST | `/api/demo/advance`, `/api/demo/reset` | Demo clock |

## Layout
| Path | What |
|---|---|
| `engine/` | Simulator, data-quality rules, vehicle baseline, decision engine, config |
| `eval/` | Engine vs thresholds, Isolation Forest ablation, Scania benchmark |
| `backend/` | FastAPI app + SQLite store |
| `frontend/index.html` | Fleet dashboard |
| `tools/import_obd.py` | OBD app log importer + calibrate/assess CLI |
| `docs/` | Interview script, interview log, competitor map |
| `results/` | Latest evaluation reports |
| `tests/` | pytest |

## Disclaimer
Decision support only. It is not a safety guarantee or a replacement for a mechanic's inspection.
