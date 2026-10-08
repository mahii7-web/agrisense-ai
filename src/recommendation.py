"""
Recommendation Module for AgriSense AI
Performs condition-based crop suitability ranking, distance computation in standardized feature space,
suitability scoring, dynamic explainability, and PCA user-point projection.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import FEATURE_COLUMNS, FEATURE_BOUNDS


def calculate_suitability_score(distance: float, sigma: float = 2.0) -> float:
    """
    Computes a non-linear Gaussian suitability percentage (0 to 100%)
    based on Euclidean distance in standardized feature space.
    """
    # Gaussian similarity kernel: score = 100 * exp(-d^2 / (2 * sigma^2))
    raw_score = 100.0 * np.exp(-(distance ** 2) / (2.0 * (sigma ** 2)))
    return round(float(np.clip(raw_score, 1.0, 99.9)), 1)


def generate_crop_explanation(
    user_inputs: Dict[str, float],
    crop_means: pd.Series,
    crop_name: str,
) -> Dict[str, Any]:
    """
    Generates explainable insights comparing user conditions to crop agronomic norms.
    Identifies high-alignment factors and potential limiting environmental factors.
    """
    aligned_factors: List[str] = []
    caution_factors: List[str] = []

    labels = {
        "temperature": ("Temperature", "°C", 3.0),
        "rainfall": ("Rainfall", "mm", 180.0),
        "humidity": ("Humidity", "%", 10.0),
        "soil_moisture": ("Soil Moisture", "%", 10.0),
        "soil_ph": ("Soil pH", "units", 0.5),
        "nitrogen": ("Nitrogen (N)", "kg/ha", 20.0),
        "phosphorus": ("Phosphorus (P)", "kg/ha", 15.0),
        "potassium": ("Potassium (K)", "kg/ha", 18.0),
    }

    for feat, (label, unit, tolerance) in labels.items():
        u_val = user_inputs[feat]
        c_val = crop_means[feat]
        diff = u_val - c_val

        if abs(diff) <= tolerance:
            aligned_factors.append(f"{label} ({u_val}{unit} matches optimal ~{round(c_val, 1)}{unit})")
        else:
            direction = "higher" if diff > 0 else "lower"
            caution_factors.append(
                f"{label} ({u_val}{unit}) is {direction} than typical average ({round(c_val, 1)}{unit})"
            )

    return {
        "crop": crop_name,
        "aligned_factors": aligned_factors,
        "caution_factors": caution_factors,
        "summary": f"{crop_name} thrives under your local {' and '.join(aligned_factors[:2]) if aligned_factors else 'conditions'}."
    }


def recommend_crops(
    user_inputs: Dict[str, float],
    df_clustered: pd.DataFrame,
    kmeans: KMeans,
    scaler: StandardScaler,
    pca: Optional[PCA] = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Core Recommendation Pipeline:
    1. Formats and scales user input vector
    2. Identifies nearest cluster centroid
    3. Computes distance to all crops in dataset
    4. Ranks crops within the predicted cluster (and overall)
    5. Calculates dynamic suitability scores
    6. Generates feature alignment explanations
    7. Projects user point onto 2D PCA space for visualization
    """
    # Vectorize input in correct feature column order
    input_vector = np.array([[user_inputs[col] for col in FEATURE_COLUMNS]], dtype=float)
    scaled_vector = scaler.transform(input_vector)

    # Predict closest cluster
    predicted_cluster = int(kmeans.predict(scaled_vector)[0])
    cluster_centers = kmeans.cluster_centers_

    # Distance to predicted cluster centroid
    centroid_dist = float(np.linalg.norm(scaled_vector - cluster_centers[predicted_cluster]))

    # Project user point onto PCA 2D coordinates
    user_pca: Optional[Tuple[float, float]] = None
    if pca is not None:
        pca_coords = pca.transform(scaled_vector)[0]
        user_pca = (round(float(pca_coords[0]), 3), round(float(pca_coords[1]), 3))

    # Compute crop-level averages across all features
    crop_stats = (
        df_clustered.groupby(["crop_name", "cluster_id"])[FEATURE_COLUMNS]
        .mean()
        .reset_index()
    )

    # Scale crop averages to compute standardized Euclidean distances
    crop_feature_matrix = crop_stats[FEATURE_COLUMNS].values
    crop_scaled_matrix = scaler.transform(crop_feature_matrix)

    # Euclidean distance from user input to each crop's mean profile
    diffs = crop_scaled_matrix - scaled_vector
    distances = np.linalg.norm(diffs, axis=1)

    crop_stats["distance"] = distances
    crop_stats["suitability_score"] = crop_stats["distance"].apply(calculate_suitability_score)

    # Filter to crops in predicted cluster first
    cluster_crops = crop_stats[crop_stats["cluster_id"] == predicted_cluster].copy()
    cluster_crops = cluster_crops.sort_values(by="suitability_score", ascending=False)

    # Also compute global top crops across any cluster (for comparison and nuance)
    global_top = crop_stats.sort_values(by="suitability_score", ascending=False).head(top_n)

    # Build recommendations list
    recommendations: List[Dict[str, Any]] = []
    top_cluster_crops = cluster_crops.head(top_n)

    for idx, row in top_cluster_crops.iterrows():
        c_name = row["crop_name"]
        score = row["suitability_score"]
        dist = round(float(row["distance"]), 3)
        c_means = row[FEATURE_COLUMNS]

        explanation = generate_crop_explanation(user_inputs, c_means, c_name)

        # Suitability Tier Badge
        if score >= 85:
            tier = "Optimal Match"
            badge_color = "#10b981"  # Emerald
        elif score >= 70:
            tier = "Highly Favorable"
            badge_color = "#3b82f6"  # Blue
        elif score >= 50:
            tier = "Moderate Match"
            badge_color = "#f59e0b"  # Amber
        else:
            tier = "Marginal Match"
            badge_color = "#6b7280"  # Gray

        recommendations.append({
            "rank": len(recommendations) + 1,
            "crop": c_name,
            "suitability_score": score,
            "distance": dist,
            "tier": tier,
            "badge_color": badge_color,
            "explanation": explanation,
            "optimal_conditions": {feat: round(float(c_means[feat]), 1) for feat in FEATURE_COLUMNS},
        })

    return {
        "predicted_cluster": predicted_cluster,
        "centroid_distance": round(centroid_dist, 3),
        "user_pca": user_pca,
        "recommendations": recommendations,
        "global_top_crops": global_top[["crop_name", "cluster_id", "suitability_score", "distance"]].to_dict(
            orient="records"
        ),
    }
