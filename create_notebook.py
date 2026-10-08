"""
Script to create notebooks/crop_analysis.ipynb
Generates a complete, structured Jupyter Notebook suitable for academic submission.
"""

import os
from pathlib import Path
import nbformat as nbf

BASE_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

def build_notebook():
    os.makedirs(str(NOTEBOOKS_DIR), exist_ok=True)
    nb = nbf.v4.new_notebook()

    cells = []

    # Title & Header Cell
    cells.append(nbf.v4.new_markdown_cell(
        "# AGRICULTURAL CROP CONDITION ANALYSIS\n"
        "### Clustering-Based Crop Suitability Recommendation System\n"
        "**Academic Project | AI & Data Science**\n\n"
        "**Objective:**\n"
        "Develop an unsupervised machine learning pipeline using K-Means Clustering and Principal Component Analysis (PCA) "
        "to group agricultural crops according to their multi-dimensional environmental requirements and recommend suitable crops "
        "based on user-provided environmental conditions."
    ))

    # Cell 1: Imports
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.cluster import KMeans\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.decomposition import PCA\n"
        "from sklearn.metrics import silhouette_score, silhouette_samples\n"
        "import joblib\n\n"
        "# Plotting configuration\n"
        "plt.style.use('seaborn-v0_8-whitegrid')\n"
        "plt.rcParams['figure.figsize'] = (10, 6)\n"
        "print('Libraries imported successfully.')"
    ))

    # Cell 2: Dataset Loading & Validation
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Dataset Loading and Schema Inspection\n"
        "We load the agricultural crop environmental requirement dataset from `data/crop_environment.csv`. "
        "The dataset features 8 key agronomic parameters:\n"
        "- **Temperature (°C)**\n"
        "- **Rainfall (mm)**\n"
        "- **Humidity (%)**\n"
        "- **Soil Moisture (%)**\n"
        "- **Soil pH**\n"
        "- **Nitrogen (N) (kg/ha)**\n"
        "- **Phosphorus (P) (kg/ha)**\n"
        "- **Potassium (K) (kg/ha)**"
    ))

    cells.append(nbf.v4.new_code_cell(
        "data_path = '../data/crop_environment.csv' if os.path.exists('../data/crop_environment.csv') else 'data/crop_environment.csv'\n"
        "df = pd.read_csv(data_path)\n"
        "print(f'Dataset Shape: {df.shape}')\n"
        "print(f'Unique Crop Varieties: {df[\"crop_name\"].nunique()}')\n"
        "display(df.head())\n"
        "display(df.info())"
    ))

    # Cell 3: Exploratory Data Analysis (EDA)
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Exploratory Data Analysis (EDA)\n"
        "Let us examine descriptive statistics, missing values, and the Pearson correlation matrix."
    ))

    cells.append(nbf.v4.new_code_cell(
        "features = ['temperature', 'rainfall', 'humidity', 'soil_moisture', 'soil_ph', 'nitrogen', 'phosphorus', 'potassium']\n"
        "display(df[features].describe().round(2))\n\n"
        "# Correlation Heatmap\n"
        "plt.figure(figsize=(9, 7))\n"
        "sns.heatmap(df[features].corr(), annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)\n"
        "plt.title('Pearson Correlation Matrix of Environmental Features', fontsize=14, fontweight='bold')\n"
        "plt.show()"
    ))

    # Cell 4: Feature Standardization
    cells.append(nbf.v4.new_markdown_cell(
        r"## 3. Feature Standardization (StandardScaler)\n"
        r"Because environmental features operate on vastly different physical scales (Rainfall: 300–2500 mm vs. Soil pH: 5–8), "
        r"we must standardize all features to zero mean ($\mu=0$) and unit variance ($\sigma=1$). "
        r"Otherwise, high-magnitude features would dominate Euclidean distance metrics."
    ))

    cells.append(nbf.v4.new_code_cell(
        "X = df[features].values\n"
        "scaler = StandardScaler()\n"
        "X_scaled = scaler.fit_transform(X)\n"
        "print('Features standardized successfully. Mean:', np.round(X_scaled.mean(axis=0), 2))\n"
        "print('Standard Deviation:', np.round(X_scaled.std(axis=0), 2))"
    ))

    # Cell 5: Elbow Method & Silhouette Score
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Determining Optimal Number of Clusters (K)\n"
        "We evaluate K values from 2 to 10 using:\n"
        "1. **Elbow Method (Inertia / Within-Cluster Sum of Squares - WCSS)**\n"
        "2. **Silhouette Coefficient (Cohesion vs. Separation Quality)**"
    ))

    cells.append(nbf.v4.new_code_cell(
        "k_range = list(range(2, 11))\n"
        "inertias = []\n"
        "silhouette_scores = []\n\n"
        "for k in k_range:\n"
        "    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n"
        "    labels = km.fit_predict(X_scaled)\n"
        "    inertias.append(km.inertia_)\n"
        "    silhouette_scores.append(silhouette_score(X_scaled, labels))\n\n"
        "fig, ax1 = plt.subplots(figsize=(10, 5))\n"
        "color = 'tab:blue'\n"
        "ax1.set_xlabel('Number of Clusters (K)', fontsize=12)\n"
        "ax1.set_ylabel('Inertia (WCSS)', color=color, fontsize=12)\n"
        "ax1.plot(k_range, inertias, marker='o', color=color, linewidth=2)\n"
        "ax1.tick_params(axis='y', labelcolor=color)\n\n"
        "ax2 = ax1.twinx()\n"
        "color = 'tab:green'\n"
        "ax2.set_ylabel('Silhouette Score', color=color, fontsize=12)\n"
        "ax2.plot(k_range, silhouette_scores, marker='s', color=color, linewidth=2, linestyle='--')\n"
        "ax2.tick_params(axis='y', labelcolor=color)\n\n"
        "plt.title('Cluster Evaluation: Elbow Method & Silhouette Score', fontsize=14, fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "best_k = k_range[np.argmax(silhouette_scores)]\n"
        "print(f'Optimal K by Silhouette Score: {best_k} (Score: {max(silhouette_scores):.4f})')"
    ))

    # Cell 6: Training K-Means
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Training Final K-Means Model\n"
        "We fit the K-Means algorithm using the optimal cluster count $K=4$."
    ))

    cells.append(nbf.v4.new_code_cell(
        "optimal_k = 4\n"
        "kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)\n"
        "df['cluster_id'] = kmeans.fit_predict(X_scaled)\n"
        "overall_silhouette = silhouette_score(X_scaled, df['cluster_id'])\n"
        "print(f'Final Model Silhouette Score: {overall_silhouette:.4f}')"
    ))

    # Cell 7: PCA 2D Reduction
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Dimensionality Reduction with PCA for 2D Visualization\n"
        "Because our dataset has 8 dimensions, we project it onto the first two principal components."
    ))

    cells.append(nbf.v4.new_code_cell(
        "pca = PCA(n_components=2, random_state=42)\n"
        "X_pca = pca.fit_transform(X_scaled)\n"
        "df['pca_1'] = X_pca[:, 0]\n"
        "df['pca_2'] = X_pca[:, 1]\n\n"
        "print(f'Explained Variance Ratio: {pca.explained_variance_ratio_}')\n"
        "print(f'Total Variance Explained: {pca.explained_variance_ratio_.sum()*100:.2f}%')\n\n"
        "plt.figure(figsize=(11, 7))\n"
        "scatter = sns.scatterplot(\n"
        "    data=df,\n"
        "    x='pca_1',\n"
        "    y='pca_2',\n"
        "    hue='cluster_id',\n"
        "    palette='Set2',\n"
        "    s=80,\n"
        "    alpha=0.85\n"
        ")\n"
        "plt.title('2D PCA Projection of Crop Environmental Clusters', fontsize=14, fontweight='bold')\n"
        "plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')\n"
        "plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')\n"
        "plt.legend(title='Cluster ID')\n"
        "plt.show()"
    ))

    # Cell 8: Cluster Profiling
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Dynamic Cluster Interpretation & Profiling\n"
        "We calculate the mean environmental attributes for each cluster and dynamically interpret the agronomic profiles."
    ))

    cells.append(nbf.v4.new_code_cell(
        "cluster_means = df.groupby('cluster_id')[features].mean().round(2)\n"
        "display(cluster_means)\n\n"
        "for cid in range(optimal_k):\n"
        "    crops = sorted(df[df['cluster_id'] == cid]['crop_name'].unique())\n"
        "    print(f'=== Cluster {cid} ({len(crops)} crops) ===')\n"
        "    print('Crops:', ', '.join(crops))\n"
        "    print('Key Means:', dict(cluster_means.loc[cid][['temperature', 'rainfall', 'soil_moisture', 'soil_ph']]))\n"
        "    print()"
    ))

    # Cell 9: Recommendation Simulation
    cells.append(nbf.v4.new_markdown_cell(
        "## 8. Crop Recommendation Simulation\n"
        "Given user input conditions, the pipeline:\n"
        "1. Scales the query vector with `StandardScaler`\n"
        "2. Predicts the closest cluster centroid\n"
        "3. Computes Euclidean distance to crop centroids\n"
        "4. Converts distance into a 0–100% Suitability Score\n"
        "5. Ranks top crops"
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Simulated user input (e.g. wetland condition)\n"
        "user_input = np.array([[28.0, 1750.0, 85.0, 82.0, 6.2, 85.0, 45.0, 45.0]])\n"
        "user_scaled = scaler.transform(user_input)\n"
        "pred_cluster = kmeans.predict(user_scaled)[0]\n"
        "print(f'Predicted Cluster: {pred_cluster}')\n\n"
        "# Compute distance to each crop's mean profile in scaled space\n"
        "crop_profiles = df.groupby('crop_name')[features].mean()\n"
        "crop_scaled = scaler.transform(crop_profiles.values)\n"
        "distances = np.linalg.norm(crop_scaled - user_scaled, axis=1)\n\n"
        "crop_scores = pd.DataFrame({\n"
        "    'crop': crop_profiles.index,\n"
        "    'distance': distances,\n"
        "    'suitability_pct': np.round(100.0 * np.exp(-(distances ** 2) / (2.0 * (2.0 ** 2))), 1)\n"
        "}).sort_values(by='suitability_pct', ascending=False)\n\n"
        "print('Top Recommended Crops:')\n"
        "display(crop_scores.head(5))"
    ))

    # Cell 10: Model Serialization
    cells.append(nbf.v4.new_markdown_cell(
        "## 9. Model Serialization\n"
        "Models are saved using Joblib so the Streamlit application can load them without retraining."
    ))

    cells.append(nbf.v4.new_code_cell(
        "os.makedirs('../models', exist_ok=True)\n"
        "joblib.dump(kmeans, '../models/kmeans_model.pkl')\n"
        "joblib.dump(scaler, '../models/scaler.pkl')\n"
        "joblib.dump(pca, '../models/pca_model.pkl')\n"
        "print('Models persisted to models/ directory successfully.')"
    ))

    nb.cells = cells

    nb_file = NOTEBOOKS_DIR / "crop_analysis.ipynb"
    with open(str(nb_file), "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook '{nb_file}' generated successfully.")

if __name__ == "__main__":
    build_notebook()
