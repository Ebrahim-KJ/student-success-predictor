from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL


def make_model(kind: str, features: list[str], seed: int = 0) -> Pipeline:
    cats = [c for c in features if c in CATEGORICAL]
    nums = [c for c in features if c not in CATEGORICAL]
    if kind == "logistic":
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False), cats),
            ("num", StandardScaler(), nums),
        ])
        est = LogisticRegression(C=0.5, max_iter=5000)
    elif kind == "lightgbm":
        from lightgbm import LGBMClassifier

        prep = ColumnTransformer([("all", "passthrough", cats + nums)])
        est = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=seed, verbose=-1)
    else:
        raise ValueError(kind)
    return Pipeline([("prep", prep), ("model", est)])


def out_of_fold(df: pd.DataFrame, features: list[str], kind: str, y: str = "dropout",
                folds: int = 5, seed: int = 0) -> np.ndarray:
    """Probability for every student from a model that never saw them."""
    p = np.zeros(len(df))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(df, df[y]):
        m = make_model(kind, features, seed).fit(df.iloc[tr][features], df.iloc[tr][y])
        p[te] = m.predict_proba(df.iloc[te][features])[:, 1]
    return p
