# analyze_clusters.py

import pandas as pd
import os

def create_cluster_profile(clustered_file_path, output_dir='03_reports_and_results/cluster_profiles'):
    """
    Loads a clustered dataset, groups by cluster, and calculates the mean for each feature.
    Saves the resulting profile to a CSV file.

    Args:
        clustered_file_path (str): The path to the input CSV file (e.g., 'people_clustered.csv').
        output_dir (str): The directory where the summary profile will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the base name for the output file (e.g., 'people')
    base_name = os.path.basename(clustered_file_path).replace('_split_clustered.csv', '')
    output_path = os.path.join(output_dir, f'{base_name}_cluster_profile.csv')

    print(f"Profiling clusters for: {base_name}")
    
    # Load the clustered data
    df = pd.read_csv(clustered_file_path)

    # We are interested in the original features, not the PCA components
    # Let's get a list of columns to analyze (exclude PCA and potentially the ID)
    cols_to_profile = [col for col in df.columns if not col.startswith('PCA_') and col != 'ID']

    if 'Cluster' not in df.columns:
        print(f"Error: 'Cluster' column not found in {clustered_file_path}")
        return

    # Group by the 'Cluster' column and calculate the mean for all other relevant columns
    # The .T transposes the result, making it easier to read (clusters as columns)
    cluster_profile = df[cols_to_profile].groupby('Cluster').mean().T

    # Save the profile to a new CSV file
    cluster_profile.to_csv(output_path)
    
    print(f"Successfully created cluster profile: {output_path}\n")