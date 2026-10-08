"""
Clustering Module for AgriSense AI
Performs feature standardization, K-Means clustering, Elbow and Silhouette evaluation,
PCA dimensionality reduction, dynamic agronomic cluster interpretation, and model persistence.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import FEATURE_COLUMNS, FEATURE_BOUNDS


def evaluate_cluster_range(
    X_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Computes Inertia (Elbow method) and Silhouette Scores for a range of K values.
    Returns metrics and suggested optimal K based on maximum silhouette score.
    """
    inertias: List[float] = []
    silhouettes: List[float] = []
    k_values = list(range(k_min, k_max + 1))

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        score = float(silhouette_score(X_scaled, labels))
        silhouettes.append(round(score, 4))

    # Suggested optimal K based on maximum silhouette score
    best_idx = int(np.argmax(silhouettes))
    suggested_k = k_values[best_idx]

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "suggested_k": suggested_k,
        "best_silhouette": silhouettes[best_idx],
    }


def interpret_cluster_profile(
    cluster_means: pd.Series,
    global_quantiles: pd.DataFrame,
) -> Dict[str, str]:
    """
    Dynamically generates agronomic interpretation for a cluster by comparing
    its centroid means against the global 33rd and 66th percentiles of the dataset.
    Never relies on hard-coded cluster numbers.
    """
    # Rainfall & Moisture Profile
    rain = cluster_means["rainfall"]
    moist = cluster_means["soil_moisture"]
    rain_q33 = global_quantiles.loc[0.33, "rainfall"]
    rain_q66 = global_quantiles.loc[0.66, "rainfall"]

    if rain >= rain_q66 or moist >= 65:
        water_desc = "High Rainfall & High Moisture Wetland"
        water_tag = "Wetland / High Moisture"
    elif rain <= rain_q33 or moist <= 35:
        water_desc = "Low Rainfall & Arid (Drought Tolerant)"
        water_tag = "Arid / Semi-Arid"
    else:
        water_desc = "Moderate Rainfall & Balanced Moisture"
        water_tag = "Mesic / Moderate"

    # Temperature Profile
    temp = cluster_means["temperature"]
    temp_q33 = global_quantiles.loc[0.33, "temperature"]
    temp_q66 = global_quantiles.loc[0.66, "temperature"]

    if temp >= temp_q66:
        thermal_desc = "Warm Tropical / Sub-tropical Climate"
    elif temp <= temp_q33:
        thermal_desc = "Cool Temperate / Winter (Rabi) Season"
    else:
        thermal_desc = "Moderate Thermal Conditions"

    # Soil pH Profile
    ph = cluster_means["soil_ph"]
    if ph < 5.8:
        ph_desc = "Slightly Acidic Soils"
    elif ph > 7.5:
        ph_desc = "Alkaline / Calcareous Soils"
    else:
        ph_desc = "Neutral Loam Soils"

    # Nutrient Profile (NPK)
    n = cluster_means["nitrogen"]
    p = cluster_means["phosphorus"]
    k_val = cluster_means["potassium"]

    n_q66 = global_quantiles.loc[0.66, "nitrogen"]
    n_q33 = global_quantiles.loc[0.33, "nitrogen"]
    k_q66 = global_quantiles.loc[0.66, "potassium"]

    if k_val >= k_q66 and n >= n_q66:
        nutrient_desc = "Heavy Feeder (High Nitrogen & Potassium)"
    elif n <= n_q33:
        nutrient_desc = "Low Nitrogen Demand (Legume / Drought Hardy)"
    elif k_val >= k_q66:
        nutrient_desc = "High Potash Requirement (Tuber & Fruit Demanding)"
    else:
        nutrient_desc = "Balanced Nutrient Consumption"

    # Synthesized Crop Group Label
    group_name = f"{water_tag} {thermal_desc.split('/')[0].strip()} Group"

    return {
        "group_name": group_name,
        "water_condition": water_desc,
        "thermal_condition": thermal_desc,
        "soil_condition": ph_desc,
        "nutrient_condition": nutrient_desc,
        "summary": f"{water_desc}, {thermal_desc.lower()}, {ph_desc.lower()}, {nutrient_desc.lower()}.",
    }


def train_crop_clustering(
    df: pd.DataFrame,
    n_clusters: Optional[int] = None,
    models_dir: str = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Full training pipeline:
    1. Standardizes numerical environmental features using StandardScaler
    2. Runs evaluation curve (Elbow + Silhouette)
    3. Fits K-Means model with optimal or chosen K
    4. Performs PCA for 2D visual projection
    5. Computes dynamic cluster interpretations
    6. Serializes model, scaler, and metadata
    """
    models_dir = str(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    X = df[FEATURE_COLUMNS].values

    # Step 1: Feature Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Evaluation curve for diagnostic reporting
    eval_results = evaluate_cluster_range(X_scaled, k_min=2, k_max=10, random_state=random_state)

    if n_clusters is None:
        # Default to evaluated optimal or 4 if evaluation is near tie
        n_clusters = eval_results["suggested_k"]
        # Ensure at least 4 clusters for rich agricultural diversity if dataset permits
        if n_clusters < 4 and len(df["crop_name"].unique()) >= 15:
            n_clusters = 4

    # Step 3: Train K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Step 4: Metrics
    overall_silhouette = float(silhouette_score(X_scaled, cluster_labels))
    sample_silhouettes = silhouette_samples(X_scaled, cluster_labels)

    # Step 5: PCA 2D Reduction
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance_ratio = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    # Append results to a working copy of DataFrame
    df_clustered = df.copy()
    df_clustered["cluster_id"] = cluster_labels
    df_clustered["pca_1"] = np.round(X_pca[:, 0], 3)
    df_clustered["pca_2"] = np.round(X_pca[:, 1], 3)
    df_clustered["sample_silhouette"] = np.round(sample_silhouettes, 4)

    # Step 6: Cluster Dynamic Interpretation
    global_quantiles = df[FEATURE_COLUMNS].quantile([0.33, 0.5, 0.66])
    cluster_profiles: Dict[int, Dict[str, Any]] = {}
    cluster_summary_rows: List[Dict[str, Any]] = []

    for cid in range(n_clusters):
        c_df = df_clustered[df_clustered["cluster_id"] == cid]
        c_means = c_df[FEATURE_COLUMNS].mean()
        interpretation = interpret_cluster_profile(c_means, global_quantiles)
        crops_in_cluster = sorted(c_df["crop_name"].unique().tolist())

        profile = {
            "cluster_id": cid,
            "crop_count": len(c_df),
            "unique_crops": crops_in_cluster,
            "group_name": interpretation["group_name"],
            "summary": interpretation["summary"],
            "conditions": interpretation,
            "means": {feat: round(float(c_means[feat]), 2) for feat in FEATURE_COLUMNS},
            "centroid_scaled": kmeans.cluster_centers_[cid].tolist(),
        }
        cluster_profiles[cid] = profile

        summary_row = {
            "Cluster ID": f"Cluster {cid}",
            "Group Name": interpretation["group_name"],
            "Crops": ", ".join(crops_in_cluster[:4]) + ("..." if len(crops_in_cluster) > 4 else ""),
            "Total Samples": len(c_df),
            "Avg Temp (°C)": round(float(c_means["temperature"]), 1),
            "Avg Rainfall (mm)": round(float(c_means["rainfall"]), 1),
            "Avg Humidity (%)": round(float(c_means["humidity"]), 1),
            "Avg Moisture (%)": round(float(c_means["soil_moisture"]), 1),
            "Avg pH": round(float(c_means["soil_ph"]), 2),
            "General Condition": interpretation["water_condition"],
        }
        cluster_summary_rows.append(summary_row)

    # Map cluster group names back to df_clustered
    df_clustered["cluster_name"] = df_clustered["cluster_id"].map(
        lambda cid: cluster_profiles[cid]["group_name"]
    )
    df_clustered["cluster_description"] = df_clustered["cluster_id"].map(
        lambda cid: cluster_profiles[cid]["summary"]
    )

    # Step 7: Persistence
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(pca, os.path.join(models_dir, "pca_model.pkl"))

    metadata = {
        "n_clusters": n_clusters,
        "silhouette_score": round(overall_silhouette, 4),
        "inertia": round(float(kmeans.inertia_), 2),
        "pca_variance_ratio": explained_variance_ratio,
        "pca_total_variance": round(sum(explained_variance_ratio), 4),
        "features": FEATURE_COLUMNS,
        "cluster_profiles": cluster_profiles,
        "eval_curve": eval_results,
    }

    with open(os.path.join(models_dir, "cluster_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "kmeans": kmeans,
        "scaler": scaler,
        "pca": pca,
        "metadata": metadata,
        "df_clustered": df_clustered,
        "summary_table": pd.DataFrame(cluster_summary_rows),
        "eval_results": eval_results,
    }


def load_trained_models(models_dir: str = "models") -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    """
    Loads saved K-Means model, StandardScaler, PCA model, and cluster metadata.
    Returns (kmeans, scaler, pca, metadata) or None elements if missing/corrupted.
    """
    models_dir = str(models_dir)
    km_path = os.path.join(models_dir, "kmeans_model.pkl")
    sc_path = os.path.join(models_dir, "scaler.pkl")
    pca_path = os.path.join(models_dir, "pca_model.pkl")
    meta_path = os.path.join(models_dir, "cluster_metadata.json")

    if not (os.path.exists(km_path) and os.path.exists(sc_path) and os.path.exists(meta_path)):
        return None, None, None, None

    try:
        kmeans = joblib.load(km_path)
        scaler = joblib.load(sc_path)
        pca = joblib.load(pca_path) if os.path.exists(pca_path) else None
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            # Convert string keys back to int for cluster_profiles
            if "cluster_profiles" in metadata:
                metadata["cluster_profiles"] = {
                    int(k): v for k, v in metadata["cluster_profiles"].items()
                }
        return kmeans, scaler, pca, metadata
    except Exception:
        return None, None, None, None
