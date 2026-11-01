"""Observational comparison with explicit assumptions: IPW, overlap and E-values."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ipw_risk_difference(df: pd.DataFrame, treatment: str, outcome: str, confounders: list[str],
                        clip: float = 0.02, n_boot: int = 500, seed: int = 0) -> dict:
    """Hajek IPW estimate of P(outcome | treated) - P(outcome | untreated), with bootstrap CI."""
    def estimate(d):
        X, t, y = d[confounders].to_numpy(float), d[treatment].to_numpy(), d[outcome].to_numpy(float)
        ps = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, t).predict_proba(X)[:, 1]
        ps = ps.clip(clip, 1 - clip)
        w1, w0 = t / ps, (1 - t) / (1 - ps)
        return (w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum(), ps

    rd, ps = estimate(df)
    rng = np.random.default_rng(seed)
    boots = [estimate(df.iloc[rng.integers(0, len(df), len(df))])[0] for _ in range(n_boot)]
    t = df[treatment].to_numpy()
    naive = df.loc[t == 1, outcome].mean() - df.loc[t == 0, outcome].mean()
    return {"naive_rd": float(naive), "ipw_rd": float(rd), "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "ps_treated_range": [float(ps[t == 1].min()), float(ps[t == 1].max())],
            "ps_control_range": [float(ps[t == 0].min()), float(ps[t == 0].max())],
            "share_clipped": float(((ps <= clip) | (ps >= 1 - clip)).mean())}


def e_value(rr: float) -> float:
    """VanderWeele & Ding (2017): minimum confounder strength (risk-ratio scale) to explain away rr."""
    rr = 1 / rr if rr < 1 else rr
    return float(rr + np.sqrt(rr * (rr - 1)))
