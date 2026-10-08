"""
Unit & Integration Test Suite for AgriSense AI
Validates all ML pipeline components, recommendation engine, and error handling.
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd

from src.data_preprocessing import (
    FEATURE_COLUMNS,
    FEATURE_BOUNDS,
    generate_synthetic_crop_data,
    load_and_validate_data,
    validate_user_input,
)
from src.clustering import (
    evaluate_cluster_range,
    interpret_cluster_profile,
    train_crop_clustering,
    load_trained_models,
)
from src.recommendation import (
    calculate_suitability_score,
    recommend_crops,
    generate_crop_explanation,
)
from src.visualization import (
    plot_elbow_curve,
    plot_silhouette_scores,
    plot_pca_clusters,
    plot_radar_cluster_profiles,
    plot_correlation_heatmap,
    plot_cluster_feature_comparison,
    plot_crop_distribution,
    plot_feature_distributions,
)


@pytest.fixture(scope="session")
def sample_dataset():
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, "crop_test.csv")
    df = generate_synthetic_crop_data(filepath=csv_path, samples_per_crop=5, random_state=42)
    yield df, csv_path, temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_data_generation_and_validation(sample_dataset):
    df, csv_path, _ = sample_dataset
    assert not df.empty
    assert "crop_name" in df.columns
    for feat in FEATURE_COLUMNS:
        assert feat in df.columns
        assert df[feat].min() >= FEATURE_BOUNDS[feat][0]
        assert df[feat].max() <= FEATURE_BOUNDS[feat][1]

    # Validate loader
    loaded_df, report = load_and_validate_data(csv_path)
    assert report["status"] == "success"
    assert len(loaded_df) == len(df)


def test_input_validation():
    # Valid input
    valid_input = {
        "temperature": 25.0,
        "rainfall": 1000.0,
        "humidity": 70.0,
        "soil_moisture": 60.0,
        "soil_ph": 6.5,
        "nitrogen": 80.0,
        "phosphorus": 45.0,
        "potassium": 45.0,
    }
    is_valid, errors = validate_user_input(valid_input)
    assert is_valid is True
    assert len(errors) == 0

    # Invalid input (out of bounds pH and temperature)
    invalid_input = valid_input.copy()
    invalid_input["temperature"] = 99.0
    invalid_input["soil_ph"] = 1.0
    is_valid, errors = validate_user_input(invalid_input)
    assert is_valid is False
    assert len(errors) >= 2


def test_clustering_training_and_serialization(sample_dataset):
    df, _, temp_dir = sample_dataset
    models_dir = os.path.join(temp_dir, "models")

    output = train_crop_clustering(
        df=df,
        n_clusters=4,
        models_dir=models_dir,
        random_state=42,
    )

    assert output["kmeans"] is not None
    assert output["scaler"] is not None
    assert output["metadata"]["n_clusters"] == 4
    assert output["metadata"]["silhouette_score"] > 0.15

    # Verify persistence
    km, sc, pca, meta = load_trained_models(models_dir=models_dir)
    assert km is not None
    assert sc is not None
    assert meta["n_clusters"] == 4


def test_recommendation_logic(sample_dataset):
    df, _, temp_dir = sample_dataset
    models_dir = os.path.join(temp_dir, "models")
    km, sc, pca, meta = load_trained_models(models_dir=models_dir)

    # Prepare df_clustered
    X_scaled = sc.transform(df[FEATURE_COLUMNS].values)
    df_clustered = df.copy()
    df_clustered["cluster_id"] = km.predict(X_scaled)

    # Test Wetland Query -> Expect Rice / Sugarcane / Banana / Jute
    wetland_inputs = {
        "temperature": 28.0,
        "rainfall": 1800.0,
        "humidity": 85.0,
        "soil_moisture": 80.0,
        "soil_ph": 6.2,
        "nitrogen": 85.0,
        "phosphorus": 45.0,
        "potassium": 45.0,
    }
    res = recommend_crops(
        user_inputs=wetland_inputs,
        df_clustered=df_clustered,
        kmeans=km,
        scaler=sc,
        pca=pca,
        top_n=3,
    )

    assert "predicted_cluster" in res
    assert len(res["recommendations"]) > 0
    top_crop = res["recommendations"][0]["crop"]
    # Should recommend wetland crop
    assert top_crop in ["Rice", "Sugarcane", "Banana", "Jute", "Coconut"]
    assert res["recommendations"][0]["suitability_score"] > 50.0


def test_visualizations_render(sample_dataset):
    df, _, temp_dir = sample_dataset
    models_dir = os.path.join(temp_dir, "models")
    km, sc, pca, meta = load_trained_models(models_dir=models_dir)

    # Prepare df_clustered with PCA
    X_scaled = sc.transform(df[FEATURE_COLUMNS].values)
    df_clustered = df.copy()
    df_clustered["cluster_id"] = km.predict(X_scaled)
    pca_pts = pca.transform(X_scaled)
    df_clustered["pca_1"] = pca_pts[:, 0]
    df_clustered["pca_2"] = pca_pts[:, 1]

    # Test all visualizers generate valid Plotly figures without exception
    fig_elbow = plot_elbow_curve(meta["eval_curve"])
    assert fig_elbow is not None

    fig_sil = plot_silhouette_scores(meta["eval_curve"])
    assert fig_sil is not None

    fig_pca = plot_pca_clusters(df_clustered, pca_variance=meta["pca_variance_ratio"], user_pca=(0.5, -0.2))
    assert fig_pca is not None

    fig_radar = plot_radar_cluster_profiles(meta["cluster_profiles"], df)
    assert fig_radar is not None

    fig_corr = plot_correlation_heatmap(df)
    assert fig_corr is not None

    fig_bar = plot_cluster_feature_comparison(meta["cluster_profiles"], selected_feature="rainfall")
    assert fig_bar is not None
