import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D


# Base directory for all reports and results
REPORTS_DIR = '03_reports_and_results'

# Specific subdirectories for different types of output
REPORTS_K_EVAL_DIR = os.path.join(REPORTS_DIR, 'k_evaluation')
REPORTS_CLUSTER_PLOTS_DIR = os.path.join(REPORTS_DIR, 'cluster_plots')
REPORTS_SCORES_DIR = os.path.join(REPORTS_DIR, 'scores')
# Directory for saving plots for all k-values
REPORTS_ALL_K_PLOTS_DIR = os.path.join(REPORTS_DIR, 'all_k_plots')


def evaluate_k_range(df, split_name, k_range=range(2, 11)):
    """
    Calculates and plots inertia and silhouette scores for a range of k values
    to find the optimal number of clusters.
    """
    print(f"--- Evaluating k for '{split_name}'. Data shape: {df.shape}, Columns: {df.columns.tolist()}")

    # Ensure the dataframe has numeric data to evaluate
    if 'ID' in df.columns:
        X = df.select_dtypes(include=np.number).drop(columns=['ID'])
    else:
        X = df.select_dtypes(include=np.number)

    # Ensure there's data to process
    if X.empty:
        print(f"Warning: No numeric data to evaluate for split '{split_name}'. Skipping.")
        return 2 # Return a default value

    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    os.makedirs(REPORTS_K_EVAL_DIR, exist_ok=True)
    plot_output_path = os.path.join(REPORTS_K_EVAL_DIR, f'{split_name}_k_evaluation.png')

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
    plt.savefig(plot_output_path)
    plt.close()

    # Determine the best k as the one with the highest silhouette score
    optimal_k = k_range[np.argmax(silhouettes)]
    return optimal_k

def cluster_with_pca(df, split_name, n_clusters, n_components=3, method='kmeans'):
    """
    Performs PCA and clustering on the given SCALED dataframe.
    Returns a dataframe with just the ID and the resulting Cluster label.
    """
    if 'ID' not in df.columns:
        raise ValueError("The input dataframe for clustering must contain an 'ID' column.")

    X = df.select_dtypes(include=np.number).drop(columns=['ID'])

    if X.empty:
        print(f"Warning: No numeric data to cluster for split '{split_name}'. Skipping.")
        return pd.DataFrame({'ID': df['ID'], 'Cluster': 0})

    os.makedirs(REPORTS_CLUSTER_PLOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_SCORES_DIR, exist_ok=True)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)

    clusters = model.fit_predict(X_pca)
    cluster_labels_df = pd.DataFrame({'ID': df['ID'], 'Cluster': clusters})

    if len(set(clusters)) > 1:
        score_text = f"{silhouette_score(X_pca, clusters):.3f}"
        score_file_path = os.path.join(REPORTS_SCORES_DIR, f'{split_name}_silhouette.txt')
        with open(score_file_path, 'w') as f:
            f.write(f"Silhouette Score for {split_name} with {n_clusters} clusters: {score_text}\n")

    title = f'"{split_name.capitalize()}" Clusters ({n_components}D PCA)\nSilhouette Score: {score_text}'
    plot_path = os.path.join(REPORTS_CLUSTER_PLOTS_DIR, f'{split_name}_clusters_{n_components}d.png')

    plt.figure(figsize=(10, 8))
    if n_components >= 3:
        ax = plt.axes(projection='3d')
        scatter = ax.scatter3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=clusters, cmap='viridis', alpha=0.6)
        ax.set_zlabel('PCA 3')
    else:
        ax = plt.axes()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.grid(True)
    
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.savefig(plot_path)
    plt.close()

    return cluster_labels_df

def save_all_k_means_plots(df, split_name, k_range=range(2, 11), n_components=2):
    """
    Performs PCA and K-Means clustering for a range of k values and saves a plot for each.

    Args:
        df (pd.DataFrame): The input dataframe.
        split_name (str): The name of the data split (e.g., 'people').
        k_range (range): The range of cluster numbers to visualize.
        n_components (int): The number of principal components to use for visualization.
    """
    print(f"--- Generating K-Means plots for k={min(k_range)} to k={max(k_range)} for '{split_name}' split ---")
    
    X = df.select_dtypes(include=np.number)
    
    os.makedirs(REPORTS_ALL_K_PLOTS_DIR, exist_ok=True)

    # Perform PCA once
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    # Loop through each value of k
    for k in k_range:
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(X_pca)
        
        # Create and save the plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        
        plt.title(f'"{split_name.capitalize()}" Clusters (k={k})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        
        plot_path = os.path.join(REPORTS_ALL_K_PLOTS_DIR, f'{split_name}_k_{k}_clusters.png')
        plt.savefig(plot_path)
        plt.close()
        
    print(f"All plots for '{split_name}' saved to {REPORTS_ALL_K_PLOTS_DIR}")