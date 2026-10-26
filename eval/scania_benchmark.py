"""Scania APS failure benchmark (UCI #414, IDA 2016 challenge).

Real heavy-truck data: 60k training / 16k test rows, 170 anonymised sensor features,
~1.7% positives (APS = air pressure system failure). Official cost: FP = 10, FN = 500.
This is a *benchmark of failure classification*. It is NOT proof our engine predicts failures on any vehicle.

Get the data (one time):
  1. Download https://archive.ics.uci.edu/static/public/414/ida2016challenge.zip
  2. Unzip so these exist:
       data/scania/aps_failure_training_set.csv
       data/scania/aps_failure_test_set.csv
  (or run with --download)

Run:  python -m eval.scania_benchmark            -> results/scania_benchmark.md
"""
import argparse
import io
import os
import time
import urllib.request
import warnings
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "scania")
URL = "https://archive.ics.uci.edu/static/public/414/ida2016challenge.zip"
C_FP, C_FN = 10, 500


def download():
    os.makedirs(DATA, exist_ok=True)
    print("downloading", URL)
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=120).read()))
    for n in z.namelist():
        if n.endswith(("training_set.csv", "test_set.csv")):
            with open(os.path.join(DATA, os.path.basename(n)), "wb") as fh:
                fh.write(z.read(n))
    print("saved to", DATA)


def read(path):
    """The UCI files start with a licence block; the real header line begins with 'class'."""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()
    start = next(i for i, l in enumerate(lines) if l.lower().startswith("class"))
    df = pd.read_csv(io.StringIO("".join(lines[start:])), na_values=["na"])
    y = (df.pop("class").astype(str).str.strip() == "pos").astype(int).values
    return df.astype(float), y


def cost(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return C_FP * fp + C_FN * fn, tn, fp, fn, tp


def best_threshold(y, p):
    """Pick the probability cut-off that minimises the business cost (on out-of-fold predictions)."""
    grid = np.unique(np.concatenate([np.linspace(0.001, 0.5, 500), np.quantile(p, np.linspace(0.8, 1, 200))]))
    costs = [cost(y, (p >= t).astype(int))[0] for t in grid]
    return float(grid[int(np.argmin(costs))])


MODELS = {
    "Logistic regression": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                 LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)),
    "Random forest": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                           RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                                  min_samples_leaf=2, n_jobs=-1, random_state=0)),
    "Gradient boosting (HistGB)": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                                         class_weight="balanced", random_state=0),
}


def run(train_path, test_path, max_rows=None):
    Xtr, ytr = read(train_path)
    Xte, yte = read(test_path)
    if max_rows:
        Xtr, ytr = Xtr.iloc[:max_rows], ytr[:max_rows]
    # drop features that are almost always missing
    keep = Xtr.columns[Xtr.isna().mean() < 0.7]
    Xtr, Xte = Xtr[keep], Xte[keep]
    rows = []
    naive = cost(yte, np.zeros_like(yte))[0]
    rows.append({"model": "Always 'no failure' (do nothing)", "threshold": "-", "cost": naive,
                 "fp": 0, "fn": int(yte.sum()), "recall": 0.0, "precision": float("nan"), "pr_auc": float("nan"),
                 "secs": 0})
    rows.append({"model": "Always 'inspect' (check every truck)", "threshold": "-", "cost": C_FP * int((yte == 0).sum()),
                 "fp": int((yte == 0).sum()), "fn": 0, "recall": 1.0, "precision": yte.mean(), "pr_auc": float("nan"),
                 "secs": 0})
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, make in MODELS.items():
        t0 = time.time()
        oof = cross_val_predict(make(), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        thr = best_threshold(ytr, oof)
        model = make().fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        c, tn, fp, fn, tp = cost(yte, (p >= thr).astype(int))
        rows.append({"model": name, "threshold": round(thr, 4), "cost": c, "fp": fp, "fn": fn,
                     "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
                     "pr_auc": average_precision_score(yte, p), "secs": round(time.time() - t0)})
        print(f"  {name}: cost {c}")
    return pd.DataFrame(rows), len(Xtr), len(Xte), int(yte.sum()), len(keep)


def report(res, ntr, nte, npos, nfeat):
    best = res.iloc[2:].sort_values("cost").iloc[0]
    lines = ["# Scania APS Failure Benchmark", "",
             f"Train {ntr:,} rows · test {nte:,} rows ({npos} failures) · {nfeat} features used.",
             f"Business cost = {C_FP} × false alarm + {C_FN} × missed failure. The decision threshold is tuned on "
             "5-fold out-of-fold training predictions, never on the test set.", "",
             "| Model | Threshold | Test cost | False alarms | Missed failures | Recall | Precision | PR-AUC |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        f = lambda v, pct=False: "–" if pd.isna(v) else (f"{v:.0%}" if pct else f"{v:.3f}")
        lines.append(f"| {r.model} | {r.threshold} | {int(r.cost):,} | {int(r.fp)} | {int(r.fn)} | "
                     f"{f(r.recall, True)} | {f(r.precision, True)} | {f(r.pr_auc)} |")
    ref = min(res.iloc[0].cost, res.iloc[1].cost)
    delta = 1 - best.cost / ref
    verdict = f"{delta:.0%} lower" if delta > 0 else f"{-delta:.0%} HIGHER — model adds no value here"
    lines += ["", f"**Best: {best.model}, cost {int(best.cost):,}** vs {int(ref):,} for the best trivial policy "
              f"({verdict}).", "",
              "Why this matters for the product:",
              "- Accuracy would be ~98% even for 'do nothing', so we report cost, recall and PR-AUC instead.",
              "- The cost-tuned threshold deliberately accepts more false alarms because a missed failure costs 50× more. "
              "That is the same trade-off a fleet makes between an extra inspection and a breakdown.",
              "- Features are anonymised and there's no time dimension, so this checks the classification part only. "
              "Persistence, vehicle baselines and explanations are evaluated in `eval_report.md`."]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--train", default=os.path.join(DATA, "aps_failure_training_set.csv"))
    ap.add_argument("--test", default=os.path.join(DATA, "aps_failure_test_set.csv"))
    ap.add_argument("--max-rows", type=int, help="subsample training rows for a quick run")
    a = ap.parse_args()
    if a.download:
        download()
    if not (os.path.exists(a.train) and os.path.exists(a.test)):
        raise SystemExit(f"Scania files not found in {DATA}. See the docstring at the top of this file, "
                         f"or run with --download.")
    res, ntr, nte, npos, nf = run(a.train, a.test, a.max_rows)
    md = report(res, ntr, nte, npos, nf)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "scania_benchmark.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
