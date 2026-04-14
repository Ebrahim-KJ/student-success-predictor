"""
load_data.py
------------
Downloads the UCI Student Performance dataset and saves it to data/raw/.
Dataset: https://archive.ics.uci.edu/ml/datasets/Student+Performance
"""

import os
import zipfile
import requests
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"


def download_dataset():
    """Download UCI Student Performance dataset if not already present."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DATA_DIR, "student.zip")

    if os.path.exists(os.path.join(RAW_DATA_DIR, "student-por.csv")):
        print("✅ Dataset already downloaded.")
        return

    print("⬇️  Downloading UCI Student Performance dataset...")
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DATA_DIR)

    os.remove(zip_path)
    print(f"✅ Dataset saved to {RAW_DATA_DIR}")


def load_portuguese() -> pd.DataFrame:
    """Load the Portuguese language course student dataset."""
    path = os.path.join(RAW_DATA_DIR, "student-por.csv")
    if not os.path.exists(path):
        download_dataset()
    df = pd.read_csv(path, sep=";")
    print(f"✅ Loaded Portuguese dataset: {df.shape[0]} students, {df.shape[1]} features")
    return df


def load_math() -> pd.DataFrame:
    """Load the Math course student dataset."""
    path = os.path.join(RAW_DATA_DIR, "student-mat.csv")
    if not os.path.exists(path):
        download_dataset()
    df = pd.read_csv(path, sep=";")
    print(f"✅ Loaded Math dataset: {df.shape[0]} students, {df.shape[1]} features")
    return df


if __name__ == "__main__":
    download_dataset()
    df = load_portuguese()
    print(df.head())
    print("\nColumns:", list(df.columns))
