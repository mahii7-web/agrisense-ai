"""
AgriSense AI - Agricultural Crop Condition Analysis
Clustering-Based Crop Suitability Recommendation System
Main Streamlit Application
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Base directory for cross-platform and cloud deployment resilience
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "crop_environment.csv"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
CSS_PATH = ASSETS_DIR / "custom.css"
BANNER_PATH = ASSETS_DIR / "agriculture_banner.png"

from src.data_preprocessing import (
    FEATURE_COLUMNS,
    FEATURE_BOUNDS,
    load_and_validate_data,
    generate_synthetic_crop_data,
    validate_user_input,
)
from src.clustering import (
    train_crop_clustering,
    load_trained_models,
)
from src.recommendation import (
    recommend_crops,
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

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriSense AI | Crop Condition Analysis & Suitability",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom CSS Loader
# -----------------------------------------------------------------------------
def load_css(css_path: str = str(CSS_PATH)):
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------------------------------------------------------
# Data & Model Cache
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset(filepath: str = str(DATA_PATH)):
    if not os.path.exists(filepath):
        df = generate_synthetic_crop_data(filepath=filepath)
    df, report = load_and_validate_data(filepath=filepath)
    return df, report

@st.cache_resource(show_spinner=False)
def get_models():
    kmeans, scaler, pca, metadata = load_trained_models(models_dir=str(MODELS_DIR))
    if kmeans is None or scaler is None or metadata is None:
        # Auto-train if artifacts are missing
        df, _ = get_dataset()
        output = train_crop_clustering(df, n_clusters=None, models_dir=str(MODELS_DIR))
        kmeans, scaler, pca, metadata = (
            output["kmeans"],
            output["scaler"],
            output["pca"],
            output["metadata"],
        )
    return kmeans, scaler, pca, metadata

# Initialize session state for navigation if not set
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

def navigate_to(page_name: str):
    st.session_state["page"] = page_name

# -----------------------------------------------------------------------------
# Sidebar Navigation & Model Info
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style='text-align: center; padding: 10px 0 16px 0;'>
        <h2 style='color: #FFFFFF; margin: 0; font-size: 1.6rem; font-weight: 800;'>🌾 AgriSense AI</h2>
        <span style='font-size: 0.82rem; color: #E8F5EC; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;'>
            Crop Clustering Engine
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

pages = ["Home", "Crop Analysis", "Cluster Analysis", "Data Exploration", "ML Model"]
selected_page = st.sidebar.radio(
    "Navigation Menu",
    pages,
    index=pages.index(st.session_state["page"]) if st.session_state["page"] in pages else 0,
    key="nav_radio",
)
st.session_state["page"] = selected_page

st.sidebar.markdown("---")

# Quick model summary in sidebar
kmeans, scaler, pca, metadata = get_models()
df, validation_report = get_dataset()

if metadata:
    st.sidebar.markdown(
        f"""
        <div style='background: rgba(255,255,255,0.08); padding: 14px; border-radius: 10px; font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.18); color: #FFFFFF;'>
            <div style='color: #3FA66B; font-weight: 800; margin-bottom: 8px; font-size: 0.88rem; letter-spacing: 0.05em;'>SYSTEM DIAGNOSTICS</div>
            <div style='color: #FFFFFF; line-height: 1.7;'>• <b>Algorithm:</b> K-Means Clustering</div>
            <div style='color: #FFFFFF; line-height: 1.7;'>• <b>Optimal K:</b> {metadata.get('n_clusters', 4)} Clusters</div>
            <div style='color: #FFFFFF; line-height: 1.7;'>• <b>Silhouette Score:</b> {metadata.get('silhouette_score', 0):.3f}</div>
            <div style='color: #FFFFFF; line-height: 1.7;'>• <b>PCA Variance:</b> {metadata.get('pca_total_variance', 0)*100:.1f}%</div>
            <div style='color: #FFFFFF; line-height: 1.7;'>• <b>Dataset Samples:</b> {len(df)} records</div>
            <div style='color: #FFFFFF; line-height: 1.7;'>• <b>Crops Cataloged:</b> {df['crop_name'].nunique()} types</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='font-size: 0.78rem; color: #E8F5EC; text-align: center; line-height: 1.5;'>
        Academic Project Implementation<br>
        <b style='color: #FFFFFF;'>AI & Data Science Domain</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# PAGE 1: HOME / OVERVIEW
# -----------------------------------------------------------------------------
if st.session_state["page"] == "Home":
    # Hero Card
    st.markdown(
        """
        <div class="agri-hero">
            <h1>AGRICULTURAL CROP CONDITION ANALYSIS</h1>
            <p>
                <b>AgriSense AI:</b> An Unsupervised Machine Learning platform that groups agricultural crops 
                by multi-dimensional environmental requirements and delivers data-driven crop suitability recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Optional banner display
    banner_path = str(BANNER_PATH)
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI Statistics Row
    total_crops = df["crop_name"].nunique() if not df.empty else 22
    total_samples = len(df) if not df.empty else 220
    n_clusters = metadata.get("n_clusters", 4) if metadata else 4
    sil_score = metadata.get("silhouette_score", 0.24) if metadata else 0.24
    n_params = len(FEATURE_COLUMNS)

    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-label">Total Crops Evaluated</div>
                <div class="kpi-value">{total_crops}</div>
                <div class="kpi-desc">Diverse agronomic varieties</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Clusters Identified (K)</div>
                <div class="kpi-value">{n_clusters}</div>
                <div class="kpi-desc">Optimized via Silhouette & Elbow</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Environmental Features</div>
                <div class="kpi-value">{n_params}</div>
                <div class="kpi-desc">Temp, Rain, Moisture, pH, NPK</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Clustering Quality</div>
                <div class="kpi-value">{sil_score:.3f}</div>
                <div class="kpi-desc">Silhouette Coefficient</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Academic Project Context & CTA
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
            <div class="custom-card">
                <div class="custom-card-title">🌱 Academic Project Overview</div>
                <p>
                    Farmers and agricultural planners frequently face challenges in matching crops to regional micro-climates. 
                    Traditional agronomy often relies on static manual lookup tables that struggle with 
                    multi-factorial interactions between thermal, hydrological, and soil chemistry requirements.
                </p>
                <p>
                    <b>AgriSense AI</b> formulates crop suitability as an <b>unsupervised clustering problem</b>. 
                    By mapping multi-dimensional environmental conditions (Temperature, Rainfall, Humidity, Soil Moisture, pH, 
                    and Nitrogen, Phosphorus, Potassium ratios) into a standardized Euclidean feature space, K-Means clustering 
                    discovers natural agronomic clusters without human labelling bias.
                </p>
                <div class="academic-callout">
                    <b>🎯 Core Academic Objective:</b> Demonstrate how unsupervised machine learning (K-Means, PCA, 
                    and Euclidean distance ranking) enables explainable, condition-based crop recommendations 
                    for sustainable precision agriculture.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="custom-card">
                <div class="custom-card-title">⚡ Quick Demonstration</div>
                <p style="font-size: 0.95rem; color: #16231B;">
                    Ready to simulate soil and weather conditions? Test live recommendations across 
                    wetland, arid, temperate, and vegetable agro-climatic scenarios.
                </p>
                <br>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚀 Analyze Crop Conditions Now", use_container_width=True):
            navigate_to("Crop Analysis")
            st.rerun()

        st.markdown(
            """
            <div class="viva-tip" style="margin-top: 1rem;">
                <b>💡 Viva Tip:</b> Point out that K-Means operates in an 8-dimensional standardized space, 
                and PCA is applied strictly for 2D human visualization.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ML Pipeline Flowchart Display
    st.markdown(
        """
        <div class="custom-card">
            <div class="custom-card-title">🔬 Machine Learning Architecture Pipeline</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.8rem; margin-top: 0.8rem;">
                <div class="pipeline-step">
                    <div class="pipeline-step-num">1</div>
                    <div style="font-weight: 700; font-size: 0.88rem; color: #123524;">Data Validation</div>
                    <div style="font-size: 0.78rem; color: #526057;">Missing values & bound checks</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-step-num">2</div>
                    <div style="font-weight: 700; font-size: 0.88rem; color: #123524;">StandardScaler</div>
                    <div style="font-size: 0.78rem; color: #526057;">Zero mean, unit variance</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-step-num">3</div>
                    <div style="font-weight: 700; font-size: 0.88rem; color: #123524;">Elbow & Silhouette</div>
                    <div style="font-size: 0.78rem; color: #526057;">Determines optimal K</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-step-num">4</div>
                    <div style="font-weight: 700; font-size: 0.88rem; color: #123524;">K-Means Fit</div>
                    <div style="font-size: 0.78rem; color: #526057;">Cluster assignment</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-step-num">5</div>
                    <div style="font-weight: 700; font-size: 0.88rem; color: #123524;">PCA Projection</div>
                    <div style="font-size: 0.78rem; color: #526057;">2D coordinate mapping</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-step-num">6</div>
                    <div style="font-weight: 700; font-size: 0.88rem; color: #123524;">Recommendation</div>
                    <div style="font-size: 0.78rem; color: #526057;">Similarity & ranking</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# PAGE 2: CROP ANALYSIS (RECOMMENDATION MODULE)
# -----------------------------------------------------------------------------
elif st.session_state["page"] == "Crop Analysis":
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
            <h2 style="color: #123524; margin-bottom: 0.2rem;">🌱 Crop Condition Analysis & Recommendation</h2>
            <p style="color: #526057; font-size: 0.95rem;">
                Input target field conditions below or select a viva demonstration scenario to predict 
                the agro-climatic cluster and obtain ranked crop suitability recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Viva Demo Presets
    st.markdown("<b style='color: #123524;'>⚡ Quick Load Scenario (Ideal for Demonstration / Viva):</b>", unsafe_allow_html=True)
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    preset_values = None

    if p_col1.button("🌧️ Monsoon Wetland (Rice / Sugarcane)"):
        preset_values = {
            "temperature": 28.0,
            "rainfall": 1750.0,
            "humidity": 85.0,
            "soil_moisture": 82.0,
            "soil_ph": 6.2,
            "nitrogen": 85.0,
            "phosphorus": 45.0,
            "potassium": 45.0,
        }
    if p_col2.button("🏜️ Semi-Arid Dryland (Millet / Sorghum)"):
        preset_values = {
            "temperature": 32.0,
            "rainfall": 420.0,
            "humidity": 35.0,
            "soil_moisture": 25.0,
            "soil_ph": 7.4,
            "nitrogen": 45.0,
            "phosphorus": 25.0,
            "potassium": 25.0,
        }
    if p_col3.button("❄️ Cool Temperate (Wheat / Barley)"):
        preset_values = {
            "temperature": 16.5,
            "rainfall": 520.0,
            "humidity": 55.0,
            "soil_moisture": 52.0,
            "soil_ph": 6.8,
            "nitrogen": 105.0,
            "phosphorus": 55.0,
            "potassium": 48.0,
        }
    if p_col4.button("🥔 High-Potash Vegetable (Potato / Tomato)"):
        preset_values = {
            "temperature": 18.5,
            "rainfall": 650.0,
            "humidity": 70.0,
            "soil_moisture": 64.0,
            "soil_ph": 5.8,
            "nitrogen": 110.0,
            "phosphorus": 65.0,
            "potassium": 125.0,
        }

    # Store presets in session state if clicked
    if preset_values:
        for k, v in preset_values.items():
            st.session_state[f"input_{k}"] = v

    # User Input Forms - Grouped logically in Cards
    with st.form("crop_condition_form"):
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<div class='custom-card-title'>📊 Environmental Parameters Input</div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("<b style='color: #123524;'>🌡️ Climate Factors</b>", unsafe_allow_html=True)
            temp = st.number_input(
                "Temperature (°C)",
                min_value=0.0,
                max_value=55.0,
                value=float(st.session_state.get("input_temperature", 25.0)),
                step=0.5,
                help="Mean ambient temperature during crop vegetative period.",
            )
            rainfall = st.number_input(
                "Annual/Seasonal Rainfall (mm)",
                min_value=0.0,
                max_value=4500.0,
                value=float(st.session_state.get("input_rainfall", 1100.0)),
                step=50.0,
                help="Total effective rainfall across the cropping cycle.",
            )

        with c2:
            st.markdown("<b style='color: #123524;'>💧 Moisture Factors</b>", unsafe_allow_html=True)
            humidity = st.number_input(
                "Relative Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get("input_humidity", 70.0)),
                step=1.0,
                help="Atmospheric relative humidity percentage.",
            )
            soil_moist = st.number_input(
                "Soil Moisture Content (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get("input_soil_moisture", 60.0)),
                step=1.0,
                help="Volumetric or tensiometric moisture percentage in root zone.",
            )

        with c3:
            st.markdown("<b style='color: #123524;'>🧪 Soil Chemistry</b>", unsafe_allow_html=True)
            soil_ph = st.number_input(
                "Soil Reaction (pH)",
                min_value=3.5,
                max_value=10.0,
                value=float(st.session_state.get("input_soil_ph", 6.5)),
                step=0.1,
                help="Soil acidity / alkalinity scale (4.5-8.5 is typical).",
            )
            nitrogen = st.number_input(
                "Nitrogen (N) (kg/ha)",
                min_value=0.0,
                max_value=300.0,
                value=float(st.session_state.get("input_nitrogen", 80.0)),
                step=5.0,
                help="Available soil Nitrogen in kilograms per hectare.",
            )

        with c4:
            st.markdown("<b style='color: #123524;'>🌾 Macro Nutrients</b>", unsafe_allow_html=True)
            phosphorus = st.number_input(
                "Phosphorus (P) (kg/ha)",
                min_value=0.0,
                max_value=300.0,
                value=float(st.session_state.get("input_phosphorus", 45.0)),
                step=5.0,
                help="Available soil Phosphorus in kilograms per hectare.",
            )
            potassium = st.number_input(
                "Potassium (K) (kg/ha)",
                min_value=0.0,
                max_value=300.0,
                value=float(st.session_state.get("input_potassium", 50.0)),
                step=5.0,
                help="Available soil Potassium in kilograms per hectare.",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        submit_btn = st.form_submit_button(
            "🔍 ANALYZE CROP CONDITIONS",
            use_container_width=True,
        )

    # Execute Analysis
    if submit_btn:
        inputs = {
            "temperature": float(temp),
            "rainfall": float(rainfall),
            "humidity": float(humidity),
            "soil_moisture": float(soil_moist),
            "soil_ph": float(soil_ph),
            "nitrogen": float(nitrogen),
            "phosphorus": float(phosphorus),
            "potassium": float(potassium),
        }

        # Validation step
        is_valid, validation_errors = validate_user_input(inputs)
        if not is_valid:
            for err in validation_errors:
                st.error(f"⚠️ Validation Error: {err}")
        else:
            # Reconstruct df_clustered if needed
            cluster_labels = kmeans.predict(scaler.transform(df[FEATURE_COLUMNS].values))
            df_clustered = df.copy()
            df_clustered["cluster_id"] = cluster_labels
            if pca is not None:
                pca_pts = pca.transform(scaler.transform(df[FEATURE_COLUMNS].values))
                df_clustered["pca_1"] = pca_pts[:, 0]
                df_clustered["pca_2"] = pca_pts[:, 1]

            # Generate Recommendation
            rec_results = recommend_crops(
                user_inputs=inputs,
                df_clustered=df_clustered,
                kmeans=kmeans,
                scaler=scaler,
                pca=pca,
                top_n=4,
            )

            pred_cid = rec_results["predicted_cluster"]
            cluster_info = metadata["cluster_profiles"].get(pred_cid, {})
            group_name = cluster_info.get("group_name", f"Cluster {pred_cid}")
            cluster_summary = cluster_info.get("summary", "Standard agronomic cluster.")

            st.markdown("<br>", unsafe_allow_html=True)

            # Results Section
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown(
                    f"""
                    <div class="custom-card" style="border-left: 6px solid #1F6B45;">
                        <span class="rec-badge" style="background-color: #1F6B45; color: #FFFFFF; margin-bottom: 8px;">
                            PRIMARY CLUSTER MATCH
                        </span>
                        <h3 style="color: #123524; margin: 0.4rem 0 0.6rem 0;">
                            Cluster {pred_cid}: {group_name}
                        </h3>
                        <p style="font-size: 0.95rem; color: #16231B; margin-bottom: 0.8rem; line-height: 1.5;">
                            <b style="color: #123524;">Environmental Profile:</b> {cluster_summary}
                        </p>
                        <div style="font-size: 0.88rem; color: #16231B; background-color: #F5F7F2; padding: 12px; border-radius: 8px; border: 1px solid #D8E2DA; line-height: 1.6;">
                            • <b>Distance to Cluster Centroid:</b> {rec_results['centroid_distance']:.3f} std units<br>
                            • <b>Cluster Size:</b> {cluster_info.get('crop_count', 'N/A')} samples in knowledge base<br>
                            • <b>Crops in Cluster:</b> {', '.join(cluster_info.get('unique_crops', []))}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.markdown(
                    """
                    <div class="custom-card">
                        <div class="custom-card-title">📌 Environmental Suitability Matrix</div>
                    """,
                    unsafe_allow_html=True,
                )
                matrix_df = pd.DataFrame({
                    "Parameter": [c.replace("_", " ").title() for c in FEATURE_COLUMNS],
                    "Your Value": [f"{inputs[c]} {FEATURE_BOUNDS[c][2]}" for c in FEATURE_COLUMNS],
                    "Cluster Avg": [f"{cluster_info['means'][c]} {FEATURE_BOUNDS[c][2]}" for c in FEATURE_COLUMNS],
                })
                st.dataframe(matrix_df, hide_index=True, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Top Recommended Crops Cards
            st.markdown("<h3 style='color: #123524; margin-top: 1rem;'>🏆 Recommended Suitable Crops</h3>", unsafe_allow_html=True)

            recs = rec_results["recommendations"]
            if recs:
                rec_cols = st.columns(len(recs))
                for idx, r in enumerate(recs):
                    with rec_cols[idx]:
                        st.markdown(
                            f"""
                            <div class="rec-card" style="border-top: 5px solid {r['badge_color']}; min-height: 280px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                    <span style="font-size: 0.82rem; font-weight: 700; color: #526057;">RANK #{r['rank']}</span>
                                    <span class="rec-badge" style="background-color: {r['badge_color']}; color: #FFFFFF;">{r['tier']}</span>
                                </div>
                                <h3 style="color: #123524; margin: 0.3rem 0; font-size: 1.35rem;">🌾 {r['crop']}</h3>
                                <div style="font-size: 1.9rem; font-weight: 800; color: {r['badge_color']}; margin-bottom: 0.5rem;">
                                    {r['suitability_score']}%
                                    <span style="font-size: 0.8rem; color: #526057; font-weight: 600;">suitability</span>
                                </div>
                                <div style="font-size: 0.85rem; color: #16231B; line-height: 1.5;">
                                    <b style="color: #123524;">Aligned Factors:</b><br>
                                    {'<br>'.join(['• ' + a for a in r['explanation']['aligned_factors'][:2]]) if r['explanation']['aligned_factors'] else '• Balanced overall similarity'}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Explainability & 2D PCA Localization
            st.markdown("<br>", unsafe_allow_html=True)
            exp_c1, exp_c2 = st.columns([1, 1])

            with exp_c1:
                st.markdown(
                    """
                    <div class="custom-card">
                        <div class="custom-card-title">📖 Recommendation Explainability Details</div>
                    """,
                    unsafe_allow_html=True,
                )
                for r in recs:
                    with st.expander(f"Why {r['crop']}? (Score: {r['suitability_score']}%)", expanded=(r['rank'] == 1)):
                        st.markdown(f"<b style='color: #123524;'>Agronomic Alignment for {r['crop']}:</b>", unsafe_allow_html=True)
                        if r["explanation"]["aligned_factors"]:
                            for a in r["explanation"]["aligned_factors"]:
                                st.markdown(f"<span style='color: #16231B;'>✅ <b>Optimal Match:</b> {a}</span>", unsafe_allow_html=True)
                        if r["explanation"]["caution_factors"]:
                            for c in r["explanation"]["caution_factors"]:
                                st.markdown(f"<span style='color: #16231B;'>⚠️ <b>Field Consideration:</b> {c}</span>", unsafe_allow_html=True)
                        st.markdown(
                            f"<small style='color: #526057;'><i>Standardized Euclidean distance to crop profile: <b>{r['distance']}</b></i></small>",
                            unsafe_allow_html=True,
                        )
                st.markdown("</div>", unsafe_allow_html=True)

            with exp_c2:
                # 2D PCA Projection with User Star Marker
                pca_fig = plot_pca_clusters(
                    df_clustered,
                    pca_variance=metadata.get("pca_variance_ratio"),
                    user_pca=rec_results.get("user_pca"),
                    user_label="Your Field Condition",
                )
                st.plotly_chart(pca_fig, use_container_width=True)

            # Explicit Academic Disclaimer
            st.markdown(
                """
                <div class="academic-callout" style="background-color: #F5F7F2; border: 1px solid #D8E2DA; border-left: 5px solid #1F6B45; color: #16231B;">
                    <b style="color: #123524;">⚖️ Academic & Agronomic Notice:</b> This recommendation is computed strictly through mathematical similarity 
                    in standardized multi-variable feature space using K-Means clustering and Euclidean distance ranking. 
                    It reflects agro-climatic suitability and does <b>not</b> constitute an absolute guarantee of crop yield, 
                    disease immunity, or commercial profitability. Real-world farming requires localized soil lab tests 
                    and irrigation planning.
                </div>
                """,
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# PAGE 3: CLUSTER ANALYSIS
# -----------------------------------------------------------------------------
elif st.session_state["page"] == "Cluster Analysis":
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
            <h2 style="color: #123524; margin-bottom: 0.2rem;">🔍 Cluster Profiles & Environmental Groups</h2>
            <p style="color: #526057; font-size: 0.95rem;">
                Examine how the K-Means algorithm partitions the agricultural domain into distinct 
                agro-ecological zones based on natural environmental boundaries.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cluster_profiles = metadata.get("cluster_profiles", {})

    # Radar Chart
    r_col1, r_col2 = st.columns([3, 2])
    with r_col1:
        radar_fig = plot_radar_cluster_profiles(cluster_profiles, df)
        st.plotly_chart(radar_fig, use_container_width=True)

    with r_col2:
        st.markdown(
            """
            <div class="custom-card">
                <div class="custom-card-title">🧭 Understanding Cluster Profiles</div>
                <p style="font-size: 0.92rem; color: #16231B; line-height: 1.6;">
                    The Radar Chart standardizes all 8 environmental features from 0% to 100% of the dataset range. 
                    Notice the stark morphological divergence:
                </p>
                <ul style="font-size: 0.88rem; color: #16231B; line-height: 1.7;">
                    <li><b style="color: #123524;">Wetland Clusters:</b> Extended outwards on Rainfall, Humidity, and Moisture axes.</li>
                    <li><b style="color: #123524;">Arid / Semi-Arid Clusters:</b> Collapsed inward on Moisture/Rainfall, elevated on Temperature.</li>
                    <li><b style="color: #123524;">Temperate Cereal Clusters:</b> Shifted lower on Temperature, balanced on Moisture.</li>
                    <li><b style="color: #123524;">Tuber / Vegetable Clusters:</b> Spike heavily along the Potassium (K) & Phosphorus (P) nutrient radii.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Detailed Cluster Breakdown Table
    st.markdown("<h3 style='color: #123524;'>📋 Comprehensive Cluster Characteristics</h3>", unsafe_allow_html=True)

    summary_rows = []
    for cid, prof in cluster_profiles.items():
        summary_rows.append({
            "Cluster ID": f"Cluster {cid}",
            "Group Designation": prof["group_name"],
            "Crops Cataloged": ", ".join(prof["unique_crops"]),
            "Sample Count": prof["crop_count"],
            "Avg Temp (°C)": prof["means"]["temperature"],
            "Avg Rainfall (mm)": prof["means"]["rainfall"],
            "Avg Humidity (%)": prof["means"]["humidity"],
            "Avg Moisture (%)": prof["means"]["soil_moisture"],
            "Avg pH": prof["means"]["soil_ph"],
            "Avg N-P-K (kg/ha)": f"{prof['means']['nitrogen']}-{prof['means']['phosphorus']}-{prof['means']['potassium']}",
            "Condition Profile": prof["conditions"]["water_condition"],
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # Interactive Feature Comparison Bar Chart
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_feat = st.selectbox(
            "Select Environmental Feature to Compare Across Clusters:",
            FEATURE_COLUMNS,
            format_func=lambda x: f"{x.replace('_', ' ').title()} ({FEATURE_BOUNDS[x][2]})",
            index=1,
        )
        st.markdown(
            f"""
            <div style="font-size: 0.88rem; color: #526057; margin-top: 1rem;">
                Comparing average <b style="color: #123524;">{selected_feat.replace('_', ' ').title()}</b> values across the {len(cluster_profiles)} discovered clusters.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        bar_fig = plot_cluster_feature_comparison(cluster_profiles, selected_feature=selected_feat)
        st.plotly_chart(bar_fig, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 4: DATA EXPLORATION
# -----------------------------------------------------------------------------
elif st.session_state["page"] == "Data Exploration":
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
            <h2 style="color: #123524; margin-bottom: 0.2rem;">📊 Agronomic Data Exploration & Validation</h2>
            <p style="color: #526057; font-size: 0.95rem;">
                Explore raw and validated agronomic data, summary statistics, feature distributions, 
                and Pearson correlation matrices.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Data Validation Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", len(df))
    m2.metric("Unique Crop Varieties", df["crop_name"].nunique())
    m3.metric("Missing Values", int(df.isna().sum().sum()))
    m4.metric("Validation Status", validation_report.get("status", "VALID").upper())

    tab1, tab2, tab3 = st.tabs(["📑 Dataset Browser", "📈 Feature Distributions", "🔥 Correlation Heatmap"])

    with tab1:
        st.markdown("<b style='color: #123524;'>Filter by Crop Variety:</b>", unsafe_allow_html=True)
        crop_filter = st.multiselect(
            "Select Crops to Filter (Leave empty for all):",
            options=sorted(df["crop_name"].unique()),
            default=[],
        )
        filtered_df = df[df["crop_name"].isin(crop_filter)] if crop_filter else df
        st.dataframe(filtered_df, use_container_width=True)

        st.markdown("<br><b style='color: #123524;'>Statistical Summary:</b>", unsafe_allow_html=True)
        st.dataframe(df[FEATURE_COLUMNS].describe().round(2), use_container_width=True)

    with tab2:
        d_col1, d_col2 = st.columns([1, 2])
        with d_col1:
            dist_feat = st.selectbox(
                "Select Feature for Distribution Plot:",
                FEATURE_COLUMNS,
                format_func=lambda x: f"{x.replace('_', ' ').title()} ({FEATURE_BOUNDS[x][2]})",
            )
            st.markdown(
                """
                <div style="font-size: 0.88rem; color: #526057; margin-top: 1rem;">
                    Examines the spread, skewness, and interquartile range (IQR) of physiological parameters across all crop records.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d_col2:
            hist_fig = plot_feature_distributions(df, feature=dist_feat)
            st.plotly_chart(hist_fig, use_container_width=True)

        crop_dist_fig = plot_crop_distribution(
            df.assign(cluster_id=kmeans.predict(scaler.transform(df[FEATURE_COLUMNS].values)))
        )
        st.plotly_chart(crop_dist_fig, use_container_width=True)

    with tab3:
        corr_fig = plot_correlation_heatmap(df)
        st.plotly_chart(corr_fig, use_container_width=True)
        st.markdown(
            """
            <div class="academic-callout">
                <b style="color: #123524;">Agronomic Insight from Correlation Analysis:</b> Notice strong positive correlations 
                between <i>Rainfall</i>, <i>Soil Moisture</i>, and <i>Relative Humidity</i>. 
                Standardization via <code>StandardScaler</code> ensures that rainfall (values up to 2500 mm) 
                does not artificially dominate pH (values around 6.5) during distance calculations.
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# PAGE 5: ML MODEL & METHODOLOGY
# -----------------------------------------------------------------------------
elif st.session_state["page"] == "ML Model":
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
            <h2 style="color: #123524; margin-bottom: 0.2rem;">🧠 Machine Learning Model & Technical Evaluation</h2>
            <p style="color: #526057; font-size: 0.95rem;">
                Rigorous evaluation metrics, Elbow Method inertia curve, Silhouette score analysis, 
                and mathematical foundation for viva presentations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eval_data = metadata.get("eval_curve", {})

    col_el, col_sil = st.columns([1, 1])

    with col_el:
        if eval_data:
            elbow_fig = plot_elbow_curve(eval_data)
            st.plotly_chart(elbow_fig, use_container_width=True)
        else:
            st.info("Evaluation data not found. Please train model.")

    with col_sil:
        if eval_data:
            sil_fig = plot_silhouette_scores(eval_data)
            st.plotly_chart(sil_fig, use_container_width=True)

    # Mathematical & Algorithmic Explanations for Viva
    st.markdown(
        """
        <div class="custom-card">
            <div class="custom-card-title">📐 Algorithmic Formulation & Mathematical Foundations</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.2rem;">
                <div style="background-color: #FFFFFF; padding: 1.2rem; border-radius: 10px; border: 1px solid #D8E2DA;">
                    <b style="color: #123524; font-size: 1rem;">1. Feature Standardization (Z-Score)</b><br>
                    <code style="display: block; margin: 10px 0; padding: 8px 10px; background-color: #F5F7F2; border: 1px solid #D8E2DA; border-radius: 6px; color: #123524; font-weight: 600;">
                        z = (x - μ) / σ
                    </code>
                    <span style="font-size: 0.85rem; color: #16231B; line-height: 1.5; display: block;">
                        Prevents high-magnitude variables (Rainfall: 300-2500 mm) from dominating 
                        small-scale features (Soil pH: 5-8) in distance-based clustering.
                    </span>
                </div>
                <div style="background-color: #FFFFFF; padding: 1.2rem; border-radius: 10px; border: 1px solid #D8E2DA;">
                    <b style="color: #123524; font-size: 1rem;">2. K-Means Objective Function</b><br>
                    <code style="display: block; margin: 10px 0; padding: 8px 10px; background-color: #F5F7F2; border: 1px solid #D8E2DA; border-radius: 6px; color: #123524; font-weight: 600;">
                        J = Σ Σ ||x - μ_k||²
                    </code>
                    <span style="font-size: 0.85rem; color: #16231B; line-height: 1.5; display: block;">
                        Minimizes the Within-Cluster Sum of Squares (Inertia) across all K partitions 
                        via iterative expectation-maximization.
                    </span>
                </div>
                <div style="background-color: #FFFFFF; padding: 1.2rem; border-radius: 10px; border: 1px solid #D8E2DA;">
                    <b style="color: #123524; font-size: 1rem;">3. Silhouette Coefficient</b><br>
                    <code style="display: block; margin: 10px 0; padding: 8px 10px; background-color: #F5F7F2; border: 1px solid #D8E2DA; border-radius: 6px; color: #123524; font-weight: 600;">
                        s(i) = (b(i) - a(i)) / max(a(i), b(i))
                    </code>
                    <span style="font-size: 0.85rem; color: #16231B; line-height: 1.5; display: block;">
                        Quantifies cluster cohesion versus nearest-neighbor cluster separation. 
                        Ranges from -1 to +1, validating cluster compactness.
                    </span>
                </div>
                <div style="background-color: #FFFFFF; padding: 1.2rem; border-radius: 10px; border: 1px solid #D8E2DA;">
                    <b style="color: #123524; font-size: 1rem;">4. Principal Component Analysis (PCA)</b><br>
                    <code style="display: block; margin: 10px 0; padding: 8px 10px; background-color: #F5F7F2; border: 1px solid #D8E2DA; border-radius: 6px; color: #123524; font-weight: 600;">
                        PC_k = argmax Var(X · w_k)
                    </code>
                    <span style="font-size: 0.85rem; color: #16231B; line-height: 1.5; display: block;">
                        Projects the 8-dimensional standardized space into orthogonal 2D eigenvectors 
                        retaining maximum shared agronomic variance for visualization.
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Retrain Pipeline Control
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙️ Pipeline Management & Retraining"):
        st.write("Retrain the K-Means clustering model dynamically on the current dataset:")
        col_k, col_act = st.columns([1, 1])
        with col_k:
            k_choice = st.selectbox(
                "Cluster Count (K) Override (Default: Auto-determined optimal)",
                ["Auto", 3, 4, 5, 6],
            )
        with col_act:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Retrain Clustering Model"):
                chosen_k = None if k_choice == "Auto" else int(k_choice)
                with st.spinner("Retraining K-Means model and updating cluster profiles..."):
                    train_crop_clustering(df, n_clusters=chosen_k, models_dir="models")
                    st.cache_resource.clear()
                    st.success("Model successfully retrained and persisted!")
                    st.rerun()

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="agri-footer">
        <div><b style="color: #123524;">AgriSense AI</b> • Agricultural Crop Condition Analysis & Suitability Recommendation</div>
        <div style="color: #526057;">Unsupervised Machine Learning System • K-Means & PCA Pipeline</div>
    </div>
    """,
    unsafe_allow_html=True,
)
