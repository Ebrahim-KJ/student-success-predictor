from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CSV = DATA_DIR / "dropout.csv"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
MIRROR = "https://raw.githubusercontent.com/SinaShakoori/Student-Dropout-Prediction/master/dataset.csv"


def snake(name: str) -> str:
    name = name.strip().lower().replace("'", "")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def download() -> Path:
    import requests

    DATA_DIR.mkdir(exist_ok=True)
    try:
        resp = requests.get(UCI_ZIP, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            CSV.write_bytes(z.read(name))
    except Exception:
        resp = requests.get(MIRROR, timeout=60)
        resp.raise_for_status()
        CSV.write_bytes(resp.content)
    return CSV


def load() -> pd.DataFrame:
    if not CSV.exists():
        download()
    df = pd.read_csv(CSV, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [snake(c) for c in df.columns]
    df = df.rename(columns={"nacionality": "nationality", "daytime_evening_attendance": "evening_attendance"})
    df["evening_attendance"] = 1 - df["evening_attendance"]
    df["dropout"] = (df["target"] == "Dropout").astype(int)
    return df


# What an advisor could know at each point in time
ENROLLMENT = [
    "marital_status", "application_mode", "application_order", "course", "evening_attendance",
    "previous_qualification", "previous_qualification_grade", "nationality",
    "mothers_qualification", "fathers_qualification", "mothers_occupation", "fathers_occupation",
    "admission_grade", "displaced", "educational_special_needs", "gender", "scholarship_holder",
    "age_at_enrollment", "international", "unemployment_rate", "inflation_rate", "gdp",
]
FINANCES = ["debtor", "tuition_fees_up_to_date"]
SEMESTER_1 = [
    "curricular_units_1st_sem_credited", "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations", "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade", "curricular_units_1st_sem_without_evaluations",
]
CATEGORICAL = ["marital_status", "application_mode", "course", "previous_qualification", "nationality",
               "mothers_qualification", "fathers_qualification", "mothers_occupation", "fathers_occupation"]

FEATURE_SETS = {
    "At enrollment": ENROLLMENT,
    "After semester 1": ENROLLMENT + FINANCES + SEMESTER_1 + ["sem1_pass_rate", "sem1_no_eval"],
}


def available(features: list[str], df: pd.DataFrame) -> list[str]:
    """The Kaggle release lacks the two admission-grade columns UCI added later."""
    return [f for f in features if f in df.columns]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    enrolled = df["curricular_units_1st_sem_enrolled"].astype(float)
    approved = df["curricular_units_1st_sem_approved"].astype(float)
    df["sem1_pass_rate"] = (approved / enrolled.where(enrolled > 0)).fillna(0.0)
    df["sem1_no_eval"] = (df["curricular_units_1st_sem_evaluations"] == 0).astype(int)
    return df
