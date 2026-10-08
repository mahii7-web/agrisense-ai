"""
Data Preprocessing Module for AgriSense AI
Handles synthetic agronomic data generation, loading, schema validation,
missing value handling, and feature extraction.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Core environmental feature columns required by the ML pipeline
FEATURE_COLUMNS = [
    "temperature",
    "rainfall",
    "humidity",
    "soil_moisture",
    "soil_ph",
    "nitrogen",
    "phosphorus",
    "potassium",
]

# Physical and agronomic validation ranges for incoming data
FEATURE_BOUNDS = {
    "temperature": (0.0, 55.0, "°C"),
    "rainfall": (0.0, 4500.0, "mm"),
    "humidity": (0.0, 100.0, "%"),
    "soil_moisture": (0.0, 100.0, "%"),
    "soil_ph": (3.5, 10.0, "pH units"),
    "nitrogen": (0.0, 300.0, "kg/ha"),
    "phosphorus": (0.0, 300.0, "kg/ha"),
    "potassium": (0.0, 300.0, "kg/ha"),
}

# Agronomic knowledge base representing realistic requirement intervals for 22 crops
CROP_PROFILES: Dict[str, Dict[str, Tuple[float, float]]] = {
    # Wetland / High Moisture Tropical Crops
    "Rice": {
        "temperature": (22.0, 33.0),
        "rainfall": (1300.0, 2400.0),
        "humidity": (78.0, 94.0),
        "soil_moisture": (72.0, 92.0),
        "soil_ph": (5.5, 6.8),
        "nitrogen": (70.0, 110.0),
        "phosphorus": (35.0, 58.0),
        "potassium": (35.0, 52.0),
    },
    "Sugarcane": {
        "temperature": (24.0, 36.0),
        "rainfall": (1200.0, 2200.0),
        "humidity": (70.0, 88.0),
        "soil_moisture": (65.0, 85.0),
        "soil_ph": (6.0, 7.5),
        "nitrogen": (105.0, 150.0),
        "phosphorus": (45.0, 70.0),
        "potassium": (60.0, 95.0),
    },
    "Banana": {
        "temperature": (24.0, 34.0),
        "rainfall": (1400.0, 2500.0),
        "humidity": (75.0, 95.0),
        "soil_moisture": (70.0, 90.0),
        "soil_ph": (5.8, 7.2),
        "nitrogen": (90.0, 130.0),
        "phosphorus": (65.0, 90.0),
        "potassium": (95.0, 145.0),
    },
    "Jute": {
        "temperature": (25.0, 35.0),
        "rainfall": (1350.0, 2250.0),
        "humidity": (72.0, 92.0),
        "soil_moisture": (68.0, 88.0),
        "soil_ph": (6.0, 7.4),
        "nitrogen": (70.0, 100.0),
        "phosphorus": (35.0, 55.0),
        "potassium": (35.0, 55.0),
    },
    "Coconut": {
        "temperature": (24.0, 33.0),
        "rainfall": (1300.0, 2350.0),
        "humidity": (70.0, 90.0),
        "soil_moisture": (60.0, 80.0),
        "soil_ph": (5.2, 7.5),
        "nitrogen": (20.0, 45.0),
        "phosphorus": (15.0, 35.0),
        "potassium": (30.0, 58.0),
    },
    # Semi-Arid & Drought-Tolerant Crops
    "Millet (Bajra)": {
        "temperature": (26.0, 38.0),
        "rainfall": (300.0, 550.0),
        "humidity": (25.0, 50.0),
        "soil_moisture": (15.0, 35.0),
        "soil_ph": (6.2, 8.2),
        "nitrogen": (35.0, 65.0),
        "phosphorus": (15.0, 35.0),
        "potassium": (15.0, 35.0),
    },
    "Sorghum (Jowar)": {
        "temperature": (25.0, 36.0),
        "rainfall": (350.0, 650.0),
        "humidity": (30.0, 55.0),
        "soil_moisture": (20.0, 40.0),
        "soil_ph": (6.0, 8.0),
        "nitrogen": (40.0, 75.0),
        "phosphorus": (20.0, 40.0),
        "potassium": (20.0, 40.0),
    },
    "Chickpea": {
        "temperature": (16.0, 28.0),
        "rainfall": (320.0, 600.0),
        "humidity": (32.0, 55.0),
        "soil_moisture": (22.0, 42.0),
        "soil_ph": (6.0, 8.0),
        "nitrogen": (22.0, 45.0),
        "phosphorus": (55.0, 80.0),
        "potassium": (65.0, 90.0),
    },
    "Pigeon Pea": {
        "temperature": (20.0, 34.0),
        "rainfall": (450.0, 800.0),
        "humidity": (35.0, 60.0),
        "soil_moisture": (25.0, 45.0),
        "soil_ph": (5.5, 7.5),
        "nitrogen": (18.0, 40.0),
        "phosphorus": (48.0, 75.0),
        "potassium": (22.0, 45.0),
    },
    "Groundnut": {
        "temperature": (22.0, 33.0),
        "rainfall": (450.0, 750.0),
        "humidity": (45.0, 70.0),
        "soil_moisture": (30.0, 50.0),
        "soil_ph": (5.8, 7.2),
        "nitrogen": (22.0, 48.0),
        "phosphorus": (42.0, 70.0),
        "potassium": (22.0, 48.0),
    },
    # Temperate & Sub-tropical Cereals
    "Wheat": {
        "temperature": (12.0, 24.0),
        "rainfall": (380.0, 750.0),
        "humidity": (45.0, 70.0),
        "soil_moisture": (42.0, 65.0),
        "soil_ph": (6.0, 7.5),
        "nitrogen": (90.0, 130.0),
        "phosphorus": (40.0, 70.0),
        "potassium": (35.0, 65.0),
    },
    "Barley": {
        "temperature": (11.0, 22.0),
        "rainfall": (320.0, 650.0),
        "humidity": (40.0, 65.0),
        "soil_moisture": (35.0, 60.0),
        "soil_ph": (6.2, 7.8),
        "nitrogen": (60.0, 100.0),
        "phosphorus": (30.0, 55.0),
        "potassium": (30.0, 55.0),
    },
    "Maize": {
        "temperature": (18.0, 30.0),
        "rainfall": (580.0, 950.0),
        "humidity": (55.0, 75.0),
        "soil_moisture": (45.0, 70.0),
        "soil_ph": (5.8, 7.5),
        "nitrogen": (75.0, 120.0),
        "phosphorus": (42.0, 70.0),
        "potassium": (38.0, 68.0),
    },
    # Commercial / Cash & Oilseed Crops
    "Cotton": {
        "temperature": (22.0, 35.0),
        "rainfall": (500.0, 920.0),
        "humidity": (45.0, 70.0),
        "soil_moisture": (38.0, 62.0),
        "soil_ph": (6.0, 8.0),
        "nitrogen": (100.0, 140.0),
        "phosphorus": (38.0, 65.0),
        "potassium": (38.0, 65.0),
    },
    "Soybean": {
        "temperature": (20.0, 32.0),
        "rainfall": (600.0, 1000.0),
        "humidity": (55.0, 75.0),
        "soil_moisture": (45.0, 70.0),
        "soil_ph": (6.0, 7.2),
        "nitrogen": (22.0, 50.0),
        "phosphorus": (58.0, 85.0),
        "potassium": (35.0, 65.0),
    },
    "Mustard": {
        "temperature": (12.0, 25.0),
        "rainfall": (300.0, 600.0),
        "humidity": (45.0, 70.0),
        "soil_moisture": (35.0, 55.0),
        "soil_ph": (6.0, 7.5),
        "nitrogen": (65.0, 100.0),
        "phosphorus": (30.0, 55.0),
        "potassium": (25.0, 50.0),
    },
    "Sunflower": {
        "temperature": (18.0, 30.0),
        "rainfall": (450.0, 800.0),
        "humidity": (40.0, 65.0),
        "soil_moisture": (32.0, 55.0),
        "soil_ph": (6.2, 7.8),
        "nitrogen": (50.0, 85.0),
        "phosphorus": (40.0, 70.0),
        "potassium": (30.0, 60.0),
    },
    "Coffee": {
        "temperature": (16.0, 26.0),
        "rainfall": (1300.0, 2200.0),
        "humidity": (70.0, 88.0),
        "soil_moisture": (60.0, 80.0),
        "soil_ph": (5.0, 6.5),
        "nitrogen": (85.0, 120.0),
        "phosphorus": (28.0, 48.0),
        "potassium": (75.0, 115.0),
    },
    # Horticultural & Tuber Crops
    "Potato": {
        "temperature": (14.0, 23.0),
        "rainfall": (450.0, 800.0),
        "humidity": (60.0, 85.0),
        "soil_moisture": (55.0, 75.0),
        "soil_ph": (5.0, 6.5),
        "nitrogen": (90.0, 140.0),
        "phosphorus": (52.0, 80.0),
        "potassium": (105.0, 150.0),
    },
    "Tomato": {
        "temperature": (18.0, 29.0),
        "rainfall": (520.0, 900.0),
        "humidity": (55.0, 80.0),
        "soil_moisture": (50.0, 70.0),
        "soil_ph": (6.0, 7.2),
        "nitrogen": (80.0, 125.0),
        "phosphorus": (50.0, 80.0),
        "potassium": (70.0, 110.0),
    },
    "Onion": {
        "temperature": (15.0, 28.0),
        "rainfall": (400.0, 750.0),
        "humidity": (50.0, 70.0),
        "soil_moisture": (40.0, 65.0),
        "soil_ph": (6.0, 7.5),
        "nitrogen": (60.0, 100.0),
        "phosphorus": (40.0, 70.0),
        "potassium": (40.0, 80.0),
    },
    "Garlic": {
        "temperature": (13.0, 24.0),
        "rainfall": (400.0, 700.0),
        "humidity": (50.0, 70.0),
        "soil_moisture": (40.0, 60.0),
        "soil_ph": (6.0, 7.5),
        "nitrogen": (60.0, 95.0),
        "phosphorus": (35.0, 65.0),
        "potassium": (40.0, 75.0),
    },
}


def generate_synthetic_crop_data(
    filepath: str = "data/crop_environment.csv",
    samples_per_crop: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic agronomic dataset based on ICAR/FAO agro-climatic standards.
    Contains at least 220 samples (22 crops x 10 realistic micro-climate observations).
    """
    filepath = str(filepath)
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    rng = np.random.default_rng(random_state)
    records: List[Dict[str, Any]] = []

    for crop, ranges in CROP_PROFILES.items():
        for i in range(samples_per_crop):
            row: Dict[str, Any] = {"crop_name": crop}
            for feat, (low, high) in ranges.items():
                # Uniform base with slight natural Gaussian perturbation
                base_val = rng.uniform(low, high)
                std_noise = (high - low) * 0.04
                val = base_val + rng.normal(0, std_noise)

                # Clamp to plausible physiological boundaries
                min_bound, max_bound, _ = FEATURE_BOUNDS[feat]
                val = float(np.clip(val, min_bound, max_bound))

                if feat in ["rainfall", "nitrogen", "phosphorus", "potassium"]:
                    row[feat] = round(val, 1)
                elif feat == "soil_ph":
                    row[feat] = round(val, 2)
                else:
                    row[feat] = round(val, 1)

            records.append(row)

    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df.to_csv(filepath, index=False)
    return df


def load_and_validate_data(
    filepath: str = "data/crop_environment.csv",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads dataset, checks schema, checks missing values, handles duplicates,
    and returns cleaned DataFrame along with a comprehensive validation report.
    """
    filepath = str(filepath)
    report: Dict[str, Any] = {
        "status": "success",
        "message": "Dataset validated successfully.",
        "original_rows": 0,
        "clean_rows": 0,
        "duplicates_removed": 0,
        "missing_values_handled": 0,
        "features_present": [],
        "warnings": [],
    }

    if not os.path.exists(filepath):
        # Auto-generate synthetic dataset if missing
        report["warnings"].append(
            f"Dataset at '{filepath}' not found. Generating default synthetic dataset."
        )
        df = generate_synthetic_crop_data(filepath=filepath)
    else:
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            report["status"] = "error"
            report["message"] = f"Failed to read CSV dataset: {str(e)}"
            return pd.DataFrame(), report

    report["original_rows"] = len(df)

    # Validate essential crop_name column
    if "crop_name" not in df.columns:
        report["status"] = "error"
        report["message"] = "Missing required column 'crop_name' in dataset."
        return pd.DataFrame(), report

    # Check for feature columns
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        report["status"] = "error"
        report["message"] = f"Missing required environmental feature columns: {missing_cols}"
        return pd.DataFrame(), report

    report["features_present"] = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Handle duplicates
    dups_count = int(df.duplicated().sum())
    if dups_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        report["duplicates_removed"] = dups_count
        report["warnings"].append(f"Removed {dups_count} duplicate row(s).")

    # Handle missing values in numerical features (median imputation if any)
    missing_total = int(df[FEATURE_COLUMNS].isna().sum().sum())
    if missing_total > 0:
        for col in FEATURE_COLUMNS:
            if df[col].isna().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        report["missing_values_handled"] = missing_total
        report["warnings"].append(f"Imputed {missing_total} missing values using column medians.")

    # Remove rows where crop_name is null
    if df["crop_name"].isna().sum() > 0:
        df = df.dropna(subset=["crop_name"]).reset_index(drop=True)

    # Validate feature data types
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clip out-of-range physical anomalies
    for feat, (low, high, unit) in FEATURE_BOUNDS.items():
        anomalies = (df[feat] < low) | (df[feat] > high)
        if anomalies.sum() > 0:
            df[feat] = df[feat].clip(lower=low, upper=high)
            report["warnings"].append(
                f"Clipped {anomalies.sum()} out-of-range values in feature '{feat}'."
            )

    report["clean_rows"] = len(df)
    return df, report


def validate_user_input(input_dict: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validates user-submitted environmental conditions for range and consistency.
    Returns (is_valid, error_messages_list).
    """
    errors: List[str] = []
    for feat in FEATURE_COLUMNS:
        if feat not in input_dict:
            errors.append(f"Missing parameter '{feat}'.")
            continue

        val = input_dict[feat]
        if not isinstance(val, (int, float)) or np.isnan(val):
            errors.append(f"Value for '{feat}' must be a valid number.")
            continue

        low, high, unit = FEATURE_BOUNDS[feat]
        if val < low or val > high:
            errors.append(
                f"{feat.replace('_', ' ').title()} ({val} {unit}) is outside realistic bounds [{low}, {high} {unit}]."
            )

    return len(errors) == 0, errors
