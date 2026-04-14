"""
Data Pipeline for Student Success Predictor
Handles loading, validation, cleaning, and preprocessing of student data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'Marital_status', 'Application_mode', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification_grade',
    'Admission_grade', 'Displaced', 'Mothers_qualification',
    'Fathers_qualification', 'Scholarship_holder', 'Age_at_enrollment',
    'International', 'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade', 'Tuition_fees_up_to_date',
    'Gender', 'Debtor', 'Unemployment_rate', 'Inflation_rate', 'GDP'
]

TARGET_COL = 'Target'

BINARY_TARGET_MAP = {
    'Dropout': 1,
    'Graduate': 0,
    'Enrolled': 0
}


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw student data from CSV."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records with {df.shape[1]} columns")
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """Run basic data quality checks."""
    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"Found {missing} missing values")
    assert TARGET_COL in df.columns, "Target column missing"
    logger.info("Data validation passed")
    return True


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that improve model performance."""
    df = df.copy()

    # Academic performance aggregates
    df['avg_grade'] = (
        df['Curricular_units_1st_sem_grade'] +
        df['Curricular_units_2nd_sem_grade']
    ) / 2

    df['total_units_approved'] = (
        df['Curricular_units_1st_sem_approved'] +
        df['Curricular_units_2nd_sem_approved']
    )

    # Grade progression (improvement from sem1 to sem2)
    df['grade_progression'] = (
        df['Curricular_units_2nd_sem_grade'] -
        df['Curricular_units_1st_sem_grade']
    )

    # Financial risk score
    df['financial_risk'] = (
        df['Debtor'].astype(int) * 2 +
        (1 - df['Tuition_fees_up_to_date'].astype(int)) * 2 +
        (1 - df['Scholarship_holder'].astype(int))
    )

    # Parental education average
    df['parental_edu_avg'] = (
        df['Mothers_qualification'] + df['Fathers_qualification']
    ) / 2

    # Admission quality relative to cohort
    df['admission_grade_zscore'] = (
        df['Admission_grade'] - df['Admission_grade'].mean()
    ) / df['Admission_grade'].std()

    # Age risk (older students statistically at higher risk)
    df['mature_student'] = (df['Age_at_enrollment'] > 23).astype(int)

    logger.info(f"Engineered {7} additional features")
    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline.
    Returns X_train, X_test, y_train, y_test, feature_names, scaler
    """
    validate_data(df)
    df = engineer_features(df)

    # Create binary target (Dropout vs Not)
    df['target_binary'] = df[TARGET_COL].map(BINARY_TARGET_MAP)

    # Extended feature list including engineered features
    engineered_features = [
        'avg_grade', 'total_units_approved', 'grade_progression',
        'financial_risk', 'parental_edu_avg',
        'admission_grade_zscore', 'mature_student'
    ]
    all_features = FEATURE_COLS + engineered_features

    X = df[all_features].fillna(df[all_features].median())
    y = df['target_binary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=all_features
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=all_features
    )

    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"Dropout rate: {y.mean():.2%}")

    return X_train_scaled, X_test_scaled, y_train, y_test, all_features, scaler


def get_risk_profile(student_data: dict, scaler, feature_names: list) -> np.ndarray:
    """Preprocess a single student record for real-time prediction."""
    df_input = pd.DataFrame([student_data])
    df_input = engineer_features(df_input)
    X = df_input[feature_names].fillna(0)
    return scaler.transform(X)
