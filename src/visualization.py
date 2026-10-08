"""
Visualization Module for AgriSense AI
Builds academic-grade interactive charts (Plotly) with strict high-contrast theme:
- Light background (#FFFFFF)
- Chart text & axis labels (#16231B)
- Headings (#123524)
- Tick labels (#526057)
- Borders & Gridlines (#D8E2DA)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_preprocessing import FEATURE_COLUMNS

# High-contrast Agricultural theme palette
PALETTE = ["#1F6B45", "#2563EB", "#D97706", "#7C3AED", "#DB2777", "#0D9488", "#EA580C"]

# Universal high-contrast styling constants
CHART_BG = "#FFFFFF"
TEXT_COLOR = "#16231B"
TITLE_COLOR = "#123524"
TICK_COLOR = "#526057"
GRID_COLOR = "#D8E2DA"
BORDER_COLOR = "#D8E2DA"


def apply_chart_theme(fig: go.Figure, title_text: str = "") -> go.Figure:
    """Applies unified high-contrast agricultural theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Inter, sans-serif", color=TEXT_COLOR, size=12),
        title=dict(
            text=title_text,
            font=dict(color=TITLE_COLOR, size=16, family="Outfit, Inter, sans-serif"),
            x=0.02,
            y=0.96,
        ),
        xaxis=dict(
            title_font=dict(color=TEXT_COLOR, size=12),
            tickfont=dict(color=TICK_COLOR, size=11),
            gridcolor=GRID_COLOR,
            linecolor=BORDER_COLOR,
            zerolinecolor=GRID_COLOR,
        ),
        yaxis=dict(
            title_font=dict(color=TEXT_COLOR, size=12),
            tickfont=dict(color=TICK_COLOR, size=11),
            gridcolor=GRID_COLOR,
            linecolor=BORDER_COLOR,
            zerolinecolor=GRID_COLOR,
        ),
        legend=dict(
            font=dict(color=TEXT_COLOR, size=11),
            title_font=dict(color=TITLE_COLOR, size=12),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor=BORDER_COLOR,
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="#123524",
            font_color="#FFFFFF",
            font_size=12,
            bordercolor="#3FA66B",
        ),
        margin=dict(l=45, r=35, t=55, b=45),
    )
    return fig


def plot_elbow_curve(eval_results: Dict[str, Any]) -> go.Figure:
    """Creates an interactive Plotly Elbow Method chart with optimal K highlighted."""
    k_vals = eval_results["k_values"]
    inertias = eval_results["inertias"]
    suggested_k = eval_results.get("suggested_k", 4)

    fig = go.Figure()

    # Inertia line
    fig.add_trace(
        go.Scatter(
            x=k_vals,
            y=inertias,
            mode="lines+markers",
            name="Inertia (WCSS)",
            line=dict(color="#1F6B45", width=3),
            marker=dict(size=9, color="#123524"),
            hovertemplate="<b>K = %{x}</b><br>Inertia: %{y:,.1f}<extra></extra>",
        )
    )

    # Optimal K highlight
    opt_inertia = inertias[k_vals.index(suggested_k)]
    fig.add_trace(
        go.Scatter(
            x=[suggested_k],
            y=[opt_inertia],
            mode="markers",
            name=f"Optimal K ({suggested_k})",
            marker=dict(symbol="star", size=18, color="#DC2626", line=dict(width=2, color="#991B1B")),
            hovertemplate="<b>Optimal K = %{x}</b><br>Inertia: %{y:,.1f}<extra></extra>",
        )
    )

    apply_chart_theme(fig, "<b>Elbow Method for Optimal K</b> (Within-Cluster Sum of Squares)")
    fig.update_layout(
        xaxis=dict(title="Number of Clusters (K)", dtick=1),
        yaxis=dict(title="Inertia (WCSS)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_silhouette_scores(eval_results: Dict[str, Any]) -> go.Figure:
    """Plots Silhouette Scores across varying K values with optimal indicator."""
    k_vals = eval_results["k_values"]
    silhouettes = eval_results["silhouettes"]
    suggested_k = eval_results.get("suggested_k", 4)

    colors = ["#1F6B45" if k == suggested_k else "#94A3B8" for k in k_vals]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=k_vals,
            y=silhouettes,
            marker_color=colors,
            text=[f"{s:.3f}" for s in silhouettes],
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=11),
            hovertemplate="<b>K = %{x}</b><br>Silhouette Score: %{y:.4f}<extra></extra>",
            name="Silhouette Score",
        )
    )

    apply_chart_theme(fig, "<b>Silhouette Analysis Across Cluster Values</b> (Separation Quality)")
    fig.update_layout(
        xaxis=dict(title="Number of Clusters (K)", dtick=1),
        yaxis=dict(title="Mean Silhouette Coefficient", range=[0, max(silhouettes) * 1.25]),
    )
    return fig


def plot_pca_clusters(
    df_clustered: pd.DataFrame,
    pca_variance: Optional[List[float]] = None,
    user_pca: Optional[Tuple[float, float]] = None,
    user_label: str = "Your Input Condition",
) -> go.Figure:
    """
    Creates an interactive 2D PCA scatter plot showing clusters, crop points,
    hover details, and optionally the user's localized query coordinates.
    """
    var_text = ""
    if pca_variance and len(pca_variance) >= 2:
        var_text = f" (PC1: {pca_variance[0]*100:.1f}%, PC2: {pca_variance[1]*100:.1f}% Variance Explained)"

    df_plot = df_clustered.copy()
    df_plot["Cluster Label"] = df_plot["cluster_id"].apply(lambda c: f"Cluster {c}")

    fig = px.scatter(
        df_plot,
        x="pca_1",
        y="pca_2",
        color="Cluster Label",
        hover_name="crop_name",
        hover_data={
            "pca_1": False,
            "pca_2": False,
            "Cluster Label": True,
            "temperature": ":.1f °C",
            "rainfall": ":.0f mm",
            "humidity": ":.0f %",
            "soil_moisture": ":.0f %",
            "soil_ph": ":.2f",
        },
        color_discrete_sequence=PALETTE,
        title=f"<b>2D PCA Projection of Crop Environmental Requirements</b>{var_text}",
    )

    fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color="#123524")))

    # Overlay user input if provided
    if user_pca is not None:
        fig.add_trace(
            go.Scatter(
                x=[user_pca[0]],
                y=[user_pca[1]],
                mode="markers+text",
                name=user_label,
                text=["★ User Condition"],
                textposition="top center",
                textfont=dict(color="#DC2626", size=12, family="Inter, sans-serif"),
                marker=dict(
                    symbol="star",
                    size=22,
                    color="#DC2626",
                    line=dict(width=2, color="#FFFFFF"),
                ),
                hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
            )
        )

    apply_chart_theme(fig, f"<b>2D PCA Projection of Crop Environmental Requirements</b>{var_text}")
    fig.update_layout(
        xaxis=dict(title="Principal Component 1 (PC1)", zeroline=True),
        yaxis=dict(title="Principal Component 2 (PC2)", zeroline=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_radar_cluster_profiles(
    cluster_profiles: Dict[int, Dict[str, Any]],
    df_raw: pd.DataFrame,
) -> go.Figure:
    """
    Renders an interactive multi-cluster Radar (Spider) chart comparing normalized
    environmental parameters across all identified clusters.
    """
    fig = go.Figure()
    categories = [
        "Temperature",
        "Rainfall",
        "Humidity",
        "Soil Moisture",
        "Soil pH",
        "Nitrogen (N)",
        "Phosphorus (P)",
        "Potassium (K)",
    ]

    # Compute min-max bounds for normalization across features
    mins = df_raw[FEATURE_COLUMNS].min()
    maxs = df_raw[FEATURE_COLUMNS].max()

    for idx, (cid, prof) in enumerate(cluster_profiles.items()):
        means = prof["means"]
        norm_vals = []
        for feat in FEATURE_COLUMNS:
            span = maxs[feat] - mins[feat]
            val = ((means[feat] - mins[feat]) / span) * 100.0 if span > 0 else 50.0
            norm_vals.append(round(float(val), 1))

        radar_values = norm_vals + [norm_vals[0]]
        radar_cats = categories + [categories[0]]

        color = PALETTE[idx % len(PALETTE)]
        fig.add_trace(
            go.Scatterpolar(
                r=radar_values,
                theta=radar_cats,
                fill="toself",
                name=f"Cluster {cid}: {prof['group_name']}",
                line=dict(color=color, width=2.5),
                opacity=0.6,
                hovertemplate="<b>%{theta}</b>: %{r:.1f}% of max<extra></extra>",
            )
        )

    apply_chart_theme(fig, "<b>Cluster Environmental Requirement Profiles</b> (Normalized Radar Comparison)")
    fig.update_layout(
        polar=dict(
            bgcolor=CHART_BG,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=GRID_COLOR,
                linecolor=BORDER_COLOR,
                tickfont=dict(size=10, color=TICK_COLOR),
            ),
            angularaxis=dict(
                gridcolor=GRID_COLOR,
                linecolor=BORDER_COLOR,
                tickfont=dict(size=11, color=TEXT_COLOR, family="Inter, sans-serif"),
            ),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Plots interactive correlation heatmap across the 8 environmental variables."""
    corr = df[FEATURE_COLUMNS].corr()
    clean_labels = [c.replace("_", " ").title() for c in FEATURE_COLUMNS]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=clean_labels,
            y=clean_labels,
            colorscale=[
                [0.0, "#0E2E1E"],
                [0.35, "#1F6B45"],
                [0.5, "#F5F7F2"],
                [0.75, "#3FA66B"],
                [1.0, "#123524"],
            ],
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11, color="#16231B"),
            colorbar=dict(
                title=dict(text="Pearson r", font=dict(color=TEXT_COLOR)),
                tickfont=dict(color=TICK_COLOR),
                outlinecolor=BORDER_COLOR,
            ),
            hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    apply_chart_theme(fig, "<b>Environmental Feature Correlation Matrix</b>")
    fig.update_layout(
        xaxis=dict(tickangle=-25, tickfont=dict(color=TEXT_COLOR)),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR)),
    )
    return fig


def plot_cluster_feature_comparison(
    cluster_profiles: Dict[int, Dict[str, Any]],
    selected_feature: str = "rainfall",
) -> go.Figure:
    """Compares average values of a selected environmental feature across clusters."""
    labels = []
    values = []
    group_names = []

    clean_name = selected_feature.replace("_", " ").title()

    for cid, prof in cluster_profiles.items():
        labels.append(f"Cluster {cid}")
        values.append(prof["means"][selected_feature])
        group_names.append(prof["group_name"])

    fig = go.Figure(
        data=go.Bar(
            x=labels,
            y=values,
            text=values,
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=11),
            customdata=group_names,
            marker_color=PALETTE[:len(labels)],
            hovertemplate="<b>%{x} (%{customdata})</b><br>" + f"{clean_name}: %{{y}}<extra></extra>",
        )
    )

    apply_chart_theme(fig, f"<b>Average {clean_name} Comparison by Cluster</b>")
    fig.update_layout(
        xaxis=dict(title="Cluster"),
        yaxis=dict(title=clean_name),
    )
    return fig


def plot_crop_distribution(df_clustered: pd.DataFrame) -> go.Figure:
    """Visualizes sample and crop count distribution per cluster."""
    counts = df_clustered.groupby("cluster_id")["crop_name"].nunique().reset_index()
    counts.columns = ["cluster_id", "distinct_crops"]
    counts["label"] = counts["cluster_id"].apply(lambda c: f"Cluster {c}")

    fig = go.Figure(
        data=go.Bar(
            x=counts["label"],
            y=counts["distinct_crops"],
            text=counts["distinct_crops"],
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=11),
            marker_color="#1F6B45",
            hovertemplate="<b>%{x}</b><br>Distinct Crops: %{y}<extra></extra>",
        )
    )

    apply_chart_theme(fig, "<b>Number of Unique Crop Varieties per Cluster</b>")
    fig.update_layout(
        xaxis=dict(title="Cluster"),
        yaxis=dict(title="Distinct Crops", dtick=1),
    )
    return fig


def plot_feature_distributions(df: pd.DataFrame, feature: str) -> go.Figure:
    """Plots histogram with KDE/box for a selected feature across the whole dataset."""
    clean_name = feature.replace("_", " ").title()
    fig = px.histogram(
        df,
        x=feature,
        nbins=25,
        marginal="box",
        title=f"<b>Distribution of {clean_name}</b> (All Crops)",
        color_discrete_sequence=["#1F6B45"],
    )
    apply_chart_theme(fig, f"<b>Distribution of {clean_name}</b> (All Crops)")
    fig.update_layout(
        xaxis=dict(title=clean_name),
        yaxis=dict(title="Frequency"),
    )
    return fig
