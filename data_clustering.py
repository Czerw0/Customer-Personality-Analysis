import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Base output directory
BASE_REPORTS_PATH = '03_reports_and_results'
CLUSTERED_DATA_OUTPUT_DIR = '02_data_split' # Save clustered data back to the splits directory

def evaluate_k_range(df, split_name, k_range=range(2, 11)):
    """
    Calculates and plots inertia and silhouette scores for a range of k values
    to find the optimal number of clusters.

    Args:
        df (pd.DataFrame): The input dataframe with numerical data for clustering.
        split_name (str): The name of the data split (e.g., 'people').
        k_range (range): The range of cluster numbers to evaluate.

    Returns:
        int: The optimal number of clusters based on the highest silhouette score.
    """
    X = df.select_dtypes(include=np.number)
    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    # Create and save the evaluation plots
    plot_output_path = os.path.join(BASE_REPORTS_PATH, 'charts')
    os.makedirs(plot_output_path, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, marker='o', linestyle='--')
    plt.title(f'Elbow Method for {split_name.capitalize()}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouettes, marker='o', linestyle='--', color='green')
    plt.title(f'Silhouette Scores for {split_name.capitalize()}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_path, f'{split_name}_k_evaluation.png'))
    plt.close()

    # Determine the best k as the one with the highest silhouette score
    optimal_k = k_range[np.argmax(silhouettes)]
    return optimal_k

def cluster_with_pca(df, split_name, n_clusters, n_components=3, method='kmeans'):
    """
    Performs PCA and clustering on the given dataframe and saves the results.

    Args:
        df (pd.DataFrame): The input dataframe.
        split_name (str): The name of the data split (e.g., 'people').
        n_clusters (int): The number of clusters to form.
        n_components (int): The number of principal components to use.
        method (str): The clustering algorithm to use ('kmeans', 'dbscan', 'agglomerative').
    """
    X = df.select_dtypes(include=np.number)
    df_copy = df.copy()

    # Define output paths
    reports_charts_path = os.path.join(BASE_REPORTS_PATH, 'charts')
    reports_scores_path = os.path.join(BASE_REPORTS_PATH, 'scores')
    os.makedirs(reports_charts_path, exist_ok=True)
    os.makedirs(reports_scores_path, exist_ok=True)

    # 1. PCA Transformation
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    # 2. Select clustering algorithm
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'dbscan':
        model = DBSCAN(eps=0.5) # Note: DBSCAN does not use n_clusters
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    # 3. Fit model and get cluster labels
    clusters = model.fit_predict(X_pca)
    df_copy['Cluster'] = clusters

    # 4. Save the dataframe with cluster labels
    clustered_output_path = os.path.join(CLUSTERED_DATA_OUTPUT_DIR, f"{split_name}_clustered.csv")
    df_copy.to_csv(clustered_output_path, index=False)

    # 5. Calculate and save silhouette score
    score_text = "N/A"
    if len(set(clusters)) > 1:
        silhouette = silhouette_score(X_pca, clusters)
        score_text = f"{silhouette:.3f}"
        with open(os.path.join(reports_scores_path, f'{split_name}_silhouette.txt'), 'w') as f:
            f.write(f"Silhouette Score for {split_name} with {n_clusters} clusters: {score_text}\n")

    # 6. Visualize and save the clustering result
    title = f'"{split_name.capitalize()}" Clusters ({n_components}D PCA)\nSilhouette Score: {score_text}'
    plot_path = os.path.join(reports_charts_path, f'{split_name}_clusters_{n_components}d.png')

    plt.figure(figsize=(10, 8))
    if n_components >= 3:
        ax = plt.axes(projection='3d')
        scatter = ax.scatter3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=clusters, cmap='viridis', alpha=0.6)
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
    else: # n_components == 2
        ax = plt.axes()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        plt.grid(True)

    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.savefig(plot_path)
    plt.close()

    return df