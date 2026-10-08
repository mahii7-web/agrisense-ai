# MINI PROJECT EXPERIMENT RECORD

---

| **EX. NO: 11**<br>**DATE:** | **MINI PROJECT - Agricultural Crop Condition Analysis** |
| :--- | :--- |

---

### Aim:
To develop a machine learning-based Agricultural Crop Condition Analysis system that groups crops according to environmental conditions and recommends suitable crops based on parameters such as temperature, rainfall, humidity, soil moisture, soil pH, nitrogen, phosphorus, and potassium.

---

### Define:
In the Define phase, the main problem identified during the empathy phase is clearly stated.

#### Problem Statement
"How might we develop a simple machine learning application that analyzes environmental conditions and groups crops according to their suitability, helping users identify crops that are better suited to specific agricultural conditions?"

#### Existing Problems
- Farmers may find it difficult to select suitable crops based on multiple environmental parameters.
- Environmental conditions vary between agricultural regions and fields.
- Considering temperature, rainfall, humidity, soil moisture, pH, and nutrients simultaneously can be difficult manually.
- Traditional crop selection may not effectively identify similarities between different crop requirements.
- There is a need for a simple data-driven crop suitability analysis tool.

#### Objective
The main objectives are:
1. To collect and analyze crop environmental condition data.
2. To preprocess and standardize environmental parameters.
3. To apply K-Means clustering to group crops with similar environmental requirements.
4. To evaluate the quality of the generated clusters using Elbow Method and Silhouette Score.
5. To analyze user-provided agricultural conditions and identify the closest crop group.
6. To recommend suitable crops through an interactive Streamlit application.

---

### Empathy:
The empathy phase focuses on understanding the difficulties faced by farmers and agricultural users in identifying environmentally suitable crops.

#### Problem Understanding
Farmers and agricultural users need an easier way to understand which crops are environmentally compatible with their field conditions. Crop suitability depends on multiple complex parameters rather than a single factor, including:
- Temperature
- Rainfall
- Humidity
- Soil moisture
- Soil pH
- Nitrogen
- Phosphorus
- Potassium

Traditional agricultural decision-making often relies on regional generalities or single-variable heuristics, which fail to account for micro-climatic variations and multi-dimensional interactions among soil chemistry and weather variables.

#### User Needs
- Users need a simple way to analyze agricultural conditions.
- The system should require understandable environmental inputs.
- The analysis should be generated quickly.
- Crop groups should be presented clearly.
- Recommended crops should be easy to understand.
- The system should provide an explanation of why crops are recommended.
- The interface should be easy to use for demonstration and practical analysis.

---

### Ideate:
During the Ideate phase, different possible solutions are considered.

#### Proposed Ideas
1. Develop a machine learning system for crop condition analysis.
2. Use environmental and soil parameters as input features.
3. Apply preprocessing and feature scaling before clustering.
4. Use K-Means clustering to group crops based on environmental similarity.
5. Use Elbow Method and Silhouette Score to evaluate cluster quality.
6. Use PCA for visualizing high-dimensional crop clusters.
7. Develop a Streamlit-based interactive dashboard.
8. Provide crop suitability recommendations based on the user's environmental conditions.

---

### Prototype:
The proposed prototype is an Agricultural Crop Condition Analysis and Crop Suitability Recommendation Dashboard developed using Python, Machine Learning, and Streamlit. The system provides an interactive interface where users can enter 8 environmental and soil parameters:
1. Temperature (°C)
2. Rainfall (mm)
3. Relative Humidity (%)
4. Soil Moisture (%)
5. Soil pH
6. Nitrogen (N)
7. Phosphorus (P)
8. Potassium (K)

After entering the details, the user can click the **"ANALYZE CROP CONDITIONS"** button. The trained machine learning model processes the input values, predicts the closest crop cluster, and generates ranked crop suitability recommendations.

The current project contains a dataset with **22 crop varieties and 220 samples** categorized across distinct agronomic categories:
- **Wetland / High Moisture:** Rice, Sugarcane, Banana, Jute, Coconut
- **Semi-Arid / Drought-Tolerant:** Millet (Bajra), Sorghum (Jowar), Chickpea, Pigeon Pea, Groundnut
- **Temperate & Sub-Tropical Cereals:** Wheat, Barley, Maize
- **Commercial & Cash Crops:** Cotton, Soybean, Mustard, Sunflower, Coffee
- **Horticultural & Vegetable Crops:** Potato, Tomato, Onion, Garlic

#### Major Sections
The prototype contains five major dashboard sections:

1. **Home / Overview:**
   Displays the project title, project overview, key statistics (total crops cataloged, number of clusters, environmental parameters evaluated, Silhouette quality score), machine learning pipeline architecture, and quick navigation call-to-actions.

2. **Crop Analysis:**
   Allows users to enter the 8 environmental parameters or select quick demonstration scenarios (e.g., Monsoon Wetland, Semi-Arid Dryland, Cool Temperate, High-Potash Vegetable), click "ANALYZE CROP CONDITIONS", and view the predicted crop cluster, ranked crop recommendations with suitability percentage scores, and feature alignment explanations.

3. **Cluster Analysis:**
   Displays cluster profiles, mean environmental characteristics, a multi-dimensional normalized radar chart comparing all clusters, and feature-by-feature comparisons across clusters.

4. **Data Exploration:**
   Provides an interactive dataset preview, crop variety filtering, descriptive statistics, individual feature distribution plots (histograms and boxplots), and a Pearson correlation heatmap.

5. **ML Model & Technical Evaluation:**
   Explains the complete machine learning process, displays the Elbow Method curve, Silhouette score analysis, 2D PCA cluster projection, and mathematical formulations for feature standardization and K-Means clustering.

---

### Machine Learning Methodology:

```
Environmental Crop Dataset
        ↓
Data Validation
        ↓
Feature Selection
        ↓
StandardScaler (z = (x - μ) / σ)
        ↓
K-Means Clustering (J = Σ ||x - μ||²)
        ↓
Elbow Method (WCSS Evaluation)
        ↓
Silhouette Evaluation (Cohesion & Separation)
        ↓
Cluster Dynamic Profiling
        ↓
PCA Dimensionality Reduction (2D Projection)
        ↓
Condition-Based Crop Suitability Recommendation
```

#### 1. Standardization
$$z = \frac{x - \mu}{\sigma}$$
Standardization ensures all features have zero mean ($\mu = 0$) and unit variance ($\sigma = 1$). This prevents parameters with large numerical ranges, such as rainfall (300–2500 mm), from dominating smaller-scale features such as soil pH (5.0–8.0) during Euclidean distance computations.

#### 2. K-Means Clustering
K-Means partitions the crop records into clusters based on environmental similarity by minimizing the Within-Cluster Sum of Squares (Inertia):
$$J = \sum_{k=1}^{K} \sum_{x \in S_k} ||x - \mu_k||^2$$
where $\mu_k$ is the centroid of cluster $S_k$. K-Means identifies natural ecological groupings rather than predicting crop yields.

#### 3. Cluster Evaluation
- **Elbow Method:** Examines Within-Cluster Sum of Squares (WCSS / Inertia) across $K \in [2, 10]$ to identify the inflection point.
- **Silhouette Score:** Evaluates cluster cohesion $a(i)$ relative to nearest cluster separation $b(i)$:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  The current project determines $K = 4$ as optimal (Silhouette Score = 0.2425, with 62.7% of total variance explained by the first two Principal Components). The Silhouette Score is a cluster separation quality metric, not a classification accuracy percentage.

---

### Code:

```python
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="AgriSense AI | Crop Condition Analysis",
    page_icon="🌱",
    layout="wide"
)

# 2. Load Dataset
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "data" / "crop_environment.csv"
df = pd.read_csv(data_path)

# 3. Select Environmental Features
features = [
    "temperature",
    "rainfall",
    "humidity",
    "soil_moisture",
    "soil_ph",
    "nitrogen",
    "phosphorus",
    "potassium",
]

X = df[features].values

# 4. Feature Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. K-Means Training (Optimal K = 4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["cluster_id"] = kmeans.fit_predict(X_scaled)

# 6. Cluster Evaluation
inertia = kmeans.inertia_
sil_score = silhouette_score(X_scaled, df["cluster_id"])

# 7. Dimensionality Reduction (PCA for 2D Projection)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df["pca_1"] = X_pca[:, 0]
df["pca_2"] = X_pca[:, 1]

# 8. User Input Interface
st.title("🌾 AgriSense AI - Crop Condition Analysis")
st.write("Enter environmental and soil parameters to analyze suitability:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=55.0, value=25.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=4500.0, value=1100.0)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
    moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=60.0)

with col3:
    soil_ph = st.number_input("Soil pH", min_value=3.5, max_value=10.0, value=6.5)
    nitrogen = st.number_input("Nitrogen (N) (kg/ha)", min_value=0.0, max_value=300.0, value=80.0)

with col4:
    phosphorus = st.number_input("Phosphorus (P) (kg/ha)", min_value=0.0, max_value=300.0, value=45.0)
    potassium = st.number_input("Potassium (K) (kg/ha)", min_value=0.0, max_value=300.0, value=50.0)

# 9. Recommendation Logic
if st.button("ANALYZE CROP CONDITIONS"):
    user_inputs = [temp, rainfall, humidity, moisture, soil_ph, nitrogen, phosphorus, potassium]
    user_df = pd.DataFrame([user_inputs], columns=features)
    
    # Scale user input
    user_scaled = scaler.transform(user_df)
    
    # Predict closest cluster
    predicted_cluster = int(kmeans.predict(user_scaled)[0])
    
    # Calculate distance to crop profile centroids
    crop_profiles = df.groupby(["crop_name", "cluster_id"])[features].mean().reset_index()
    crop_scaled = scaler.transform(crop_profiles[features].values)
    
    distances = np.linalg.norm(crop_scaled - user_scaled, axis=1)
    crop_profiles["distance"] = distances
    
    # Gaussian suitability score formula: S = 100 * exp(-d^2 / (2 * sigma^2))
    crop_profiles["suitability_score"] = np.round(
        100.0 * np.exp(-(distances ** 2) / (2.0 * (2.0 ** 2))), 1
    )
    
    # Rank crops within the predicted cluster
    cluster_crops = crop_profiles[crop_profiles["cluster_id"] == predicted_cluster]
    ranked_crops = cluster_crops.sort_values(by="suitability_score", ascending=False)
    
    # Display Results
    st.subheader(f"Predicted Crop Group: Cluster {predicted_cluster}")
    st.write(f"Cluster Cohesion & Quality Metric (Silhouette Score): **{sil_score:.3f}**")
    
    st.markdown("### Top Recommended Crops:")
    for rank, (_, row) in enumerate(ranked_crops.head(3).iterrows(), 1):
        st.write(
            f"**{rank}. {row['crop_name']}** — Suitability Score: "
            f"**{row['suitability_score']}%** (Distance: {row['distance']:.3f})"
        )
    
    st.info(
        "Notice: This recommendation is data-driven based on environmental similarity "
        "and is intended for educational and decision-support purposes."
    )
```

---

### App:

```
+-----------------------------------------------------------------------------------+
|                                                                                   |
|                   [ SCREENSHOT 1: AgriSense AI Home Dashboard ]                   |
|                                                                                   |
|  * High-resolution Precision Agriculture Banner                                   |
|  * Key KPI Metrics: 22 Crops, 4 Clusters, 8 Parameters, 0.243 Silhouette Score    |
|  * Project Overview and 6-Step Machine Learning Pipeline Diagram                  |
|                                                                                   |
+-----------------------------------------------------------------------------------+
Fig. 1: AgriSense AI Home Dashboard
```

```
+-----------------------------------------------------------------------------------+
|                                                                                   |
|             [ SCREENSHOT 2: Agricultural Condition Input Interface ]              |
|                                                                                   |
|  * 4 Quick-Load Viva Presets: Monsoon Wetland, Semi-Arid, Temperate, Vegetable    |
|  * Form Input Cards: Temperature, Rainfall, Humidity, Moisture, pH, N, P, K      |
|  * Primary Action Button: "ANALYZE CROP CONDITIONS"                               |
|                                                                                   |
+-----------------------------------------------------------------------------------+
Fig. 2: Agricultural Condition Input Interface
```

```
+-----------------------------------------------------------------------------------+
|                                                                                   |
|             [ SCREENSHOT 3: Crop Suitability Recommendation Result ]              |
|                                                                                   |
|  * Primary Cluster Match Card with Dynamic Agronomic Summary                      |
|  * Ranked Suitable Crop Cards with Suitability Badges (Rice 91.1%, Jute 87.6%)    |
|  * Feature Alignment Breakdown (Aligned Factors vs. Field Considerations)         |
|                                                                                   |
+-----------------------------------------------------------------------------------+
Fig. 3: Crop Suitability Recommendation Result
```

```
+-----------------------------------------------------------------------------------+
|                                                                                   |
|                [ SCREENSHOT 4: Cluster Analysis & Radar Dashboard ]                |
|                                                                                   |
|  * Normalized 8-Axis Multi-Cluster Radar (Spider) Chart                           |
|  * Comprehensive Cluster Characteristics Summary Table                            |
|  * Interactive Single-Feature Comparison Across Clusters                          |
|                                                                                   |
+-----------------------------------------------------------------------------------+
Fig. 4: Cluster Analysis Dashboard
```

```
+-----------------------------------------------------------------------------------+
|                                                                                   |
|           [ SCREENSHOT 5: Data Exploration & Feature Correlation Heatmap ]         |
|                                                                                   |
|  * Dataset Browser with Multi-Crop Variety Filtering                              |
|  * Descriptive Statistics Table (Mean, Std, Min, IQR, Max)                        |
|  * Pearson Correlation Matrix Heatmap across all 8 Environmental Variables        |
|                                                                                   |
+-----------------------------------------------------------------------------------+
Fig. 5: Data Exploration
```

```
+-----------------------------------------------------------------------------------+
|                                                                                   |
|             [ SCREENSHOT 6: Machine Learning Model Technical Evaluation ]          |
|                                                                                   |
|  * Elbow Method Curve Plot (Inertia / WCSS vs. Cluster Count K)                   |
|  * Silhouette Score Analysis Across K Values (Highlighting Optimal K = 4)         |
|  * Mathematical Formulations: Z-Score Standardization & K-Means Objective         |
|                                                                                   |
+-----------------------------------------------------------------------------------+
Fig. 6: Machine Learning Model Evaluation
```

```
+-----------------------------------------------------------------------------------+
|                                                                                   |
|              [ SCREENSHOT 7: 2D PCA Cluster Space Visualization ]                 |
|                                                                                   |
|  * 2D Principal Component Projection (PC1: 42.0%, PC2: 20.8% Variance Explained)  |
|  * Clustered Crop Data Points with Interactive Hover Metadata                      |
|  * Projected User Query Condition Plotted as a Distinctive Star Marker (★)       |
|                                                                                   |
+-----------------------------------------------------------------------------------+
Fig. 7: 2D PCA Cluster Visualization
```

---

### Result:
The Agricultural Crop Condition Analysis system was successfully developed using Machine Learning and Streamlit. The system analyzes environmental and soil parameters, groups crops using K-Means clustering, evaluates the clusters using Elbow Method and Silhouette Score, and recommends suitable crops based on the entered agricultural conditions. The system was tested with different environmental inputs and successfully generated crop group and suitability recommendations.
