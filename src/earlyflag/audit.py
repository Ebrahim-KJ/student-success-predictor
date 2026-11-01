from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def summary(y, p, probabilities: bool = True) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, float)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p) if probabilities else float("nan"), "base_rate": float(y.mean())}


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df.p, bins, labels=False, duplicates="drop")
    return df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()


def flag_top(p, capacity: float) -> np.ndarray:
    """Flag the `capacity` share of students with the highest risk (ties broken by order)."""
    p = np.asarray(p, float)
    k = int(round(capacity * p.size))
    flags = np.zeros(p.size, bool)
    flags[np.argsort(-p, kind="mergesort")[:k]] = True
    return flags


def capacity_curve(y, p, capacities) -> pd.DataFrame:
    y = np.asarray(y)
    rows = []
    for c in capacities:
        f = flag_top(p, c)
        tp = (f & (y == 1)).sum()
        rows.append({"capacity": c, "flagged": int(f.sum()), "dropouts_reached": int(tp),
                     "recall": tp / max((y == 1).sum(), 1), "precision": tp / max(f.sum(), 1)})
    return pd.DataFrame(rows)


def group_audit(y, p, flags, groups: pd.Series) -> pd.DataFrame:
    """Error rates and calibration by group at a fixed flagging rule."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, float), "f": np.asarray(flags, bool),
                       "g": np.asarray(groups)})
    rows = []
    for g, d in df.groupby("g"):
        pos, neg = d[d.y == 1], d[d.y == 0]
        rows.append({
            "group": g, "n": len(d), "dropout_rate": d.y.mean(), "flag_rate": d.f.mean(),
            "recall": pos.f.mean() if len(pos) else np.nan,
            "false_positive_rate": neg.f.mean() if len(neg) else np.nan,
            "precision": d[d.f].y.mean() if d.f.any() else np.nan,
            "mean_predicted": d.p.mean(),
        })
    return pd.DataFrame(rows)
