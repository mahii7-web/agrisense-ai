"""
Model Training & Initial Pipeline Setup Script
Generates data (if missing), trains K-Means, computes evaluation metrics, and persists artifacts.
"""

import os
import sys
from pathlib import Path

# Base directory for cross-platform and cloud deployment resilience
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "crop_environment.csv"
MODELS_DIR = BASE_DIR / "models"

# Ensure project root is on sys.path
sys.path.insert(0, str(BASE_DIR))

from src.data_preprocessing import (
    generate_synthetic_crop_data,
    load_and_validate_data,
    FEATURE_COLUMNS,
)
from src.clustering import train_crop_clustering, load_trained_models
from src.recommendation import recommend_crops

def run_pipeline():
    print("=" * 70)
    print("AgriSense AI: Machine Learning Training Pipeline")
    print("=" * 70)

    data_path = str(DATA_PATH)
    models_dir = str(MODELS_DIR)

    if not os.path.exists(data_path):
        print(f"[1/4] Generating realistic synthetic agronomic dataset at '{data_path}'...")
        df = generate_synthetic_crop_data(filepath=data_path, samples_per_crop=10)
        print(f"      -> Created {len(df)} samples across {df['crop_name'].nunique()} crop varieties.")
    else:
        print(f"[1/4] Existing dataset found at '{data_path}'.")

    # Step 2: Validate Data
    print("[2/4] Validating dataset schema, bounds, and null values...")
    df, report = load_and_validate_data(filepath=data_path)
    print(f"      -> Status: {report['status'].upper()} ({len(df)} records valid)")
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"      * Note: {w}")

    # Step 3: Train Clustering Model
    print("[3/4] Training K-Means clustering pipeline & standardizing features...")
    training_output = train_crop_clustering(
        df=df,
        n_clusters=None, # Automatically determined by evaluation
        models_dir=models_dir,
        random_state=42,
    )

    meta = training_output["metadata"]
    print(f"      -> Selected Optimal K: {meta['n_clusters']}")
    print(f"      -> Silhouette Score: {meta['silhouette_score']:.4f}")
    print(f"      -> Model Inertia: {meta['inertia']:.2f}")
    print(f"      -> PCA Variance Explained: {meta['pca_variance_ratio']} (Total: {meta['pca_total_variance']*100:.1f}%)")

    print("\n   [Identified Cluster Profiles]:")
    for cid, prof in meta["cluster_profiles"].items():
        crops_preview = ", ".join(prof["unique_crops"][:3]) + ("..." if len(prof["unique_crops"]) > 3 else "")
        print(f"      Cluster {cid} ({prof['group_name']}):")
        print(f"         - Crops: {crops_preview}")
        print(f"         - Summary: {prof['summary']}")
        print(f"         - Avg Rain: {prof['means']['rainfall']}mm | Avg Temp: {prof['means']['temperature']}°C | Avg pH: {prof['means']['soil_ph']}")

    # Step 4: Test Recommendation Engine
    print("\n[4/4] Testing recommendation module with test agronomic inputs...")
    test_inputs = {
        "temperature": 27.5,
        "rainfall": 1600.0,
        "humidity": 82.0,
        "soil_moisture": 78.0,
        "soil_ph": 6.2,
        "nitrogen": 85.0,
        "phosphorus": 45.0,
        "potassium": 45.0,
    }
    rec_res = recommend_crops(
        user_inputs=test_inputs,
        df_clustered=training_output["df_clustered"],
        kmeans=training_output["kmeans"],
        scaler=training_output["scaler"],
        pca=training_output["pca"],
        top_n=3,
    )
    print(f"      -> Predicted Cluster: {rec_res['predicted_cluster']}")
    print(f"      -> Recommended Crops:")
    for r in rec_res["recommendations"]:
        print(f"         #{r['rank']}: {r['crop']} (Suitability: {r['suitability_score']}% - {r['tier']})")

    print("\n" + "=" * 70)
    print("Pipeline Execution Completed Successfully! Artifacts saved to models/")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
