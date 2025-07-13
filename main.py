import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import data_loader
import data_processing
import data_split
import data_clustering
import EDA as eda
from analyze_clusters import analyze_and_interpret_clusters
from data_split import COL_DEFINITIONS

# Define constants for paths
RAW_DATA_PATH = '00_raw_data/marketing_campaign.csv'
SPLIT_DATA_DIR = '02_data_split'
REPORTS_DIR = '03_reports_and_results'
REPORTS_DIR_EDA = os.path.join(REPORTS_DIR, 'charts')

def main():
    print("Starting Customer Personality Cluster Pipeline")
    
    # Ensure all needed directories exist
    os.makedirs(SPLIT_DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR_EDA, exist_ok=True)

    # 1. Load, Check and Process Data
    df_raw = data_loader.load_raw_data(RAW_DATA_PATH)
    eda.simple_eda(df_raw)
    df_processed = data_processing.processing(df_raw)

    # 2. Perform and Save Full Exploratory Data Analysis
    print("\n--- Performing Exploratory Data Analysis ---")
    eda.eda(df_processed, output_dir=REPORTS_DIR_EDA)
    print("EDA completed. Charts saved.")

    # 3. Create unscaled (for analysis) and scaled (for clustering) dataframes
    df_unscaled_for_lookup = df_processed.copy()
    df_unscaled = pd.get_dummies(df_processed, columns=['Education', 'Living_With'], drop_first=True)
    df_scaled = df_unscaled.copy()
    
    numeric_cols = df_processed.select_dtypes(include=np.number).columns
    if 'ID' in numeric_cols:
        numeric_cols = numeric_cols.drop('ID')
    
    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    print("\nCreated 'unscaled' and 'scaled' dataframes.")

    # 4. Split both dataframes into 4P groups
    unscaled_dir = os.path.join(SPLIT_DATA_DIR, 'unscaled')
    scaled_dir = os.path.join(SPLIT_DATA_DIR, 'scaled')
    data_split.split_by_marketing_4ps(df_unscaled, output_dir=unscaled_dir)
    data_split.split_by_marketing_4ps(df_scaled, output_dir=scaled_dir)
    print("Data split for both scaled and unscaled sets completed.")

    # 5. Loop through splits, cluster, merge, and analyze
    print("\n--- Starting Clustering and Analysis Process for all 4P Splits ---")
    
    final_k_values = {'people': 5, 'products': 2, 'promotion': 10, 'place': 2} 

    for split_name in COL_DEFINITIONS.keys():
        print(f"\n--- Processing '{split_name}' split ---")
        
        df_split_scaled = pd.read_csv(os.path.join(scaled_dir, f"{split_name}_split.csv"))
        df_split_unscaled = pd.read_csv(os.path.join(unscaled_dir, f"{split_name}_split.csv"))
        
        suggested_k = data_clustering.evaluate_k_range(df=df_split_scaled, split_name=split_name)
        final_k = final_k_values.get(split_name, suggested_k) # Use predefined k or fallback to suggested
        print(f"Automated suggestion for '{split_name}' k = {suggested_k}. Using final k = {final_k}.")

        cluster_labels = data_clustering.cluster_with_pca(
            df=df_split_scaled, split_name=split_name, n_clusters=final_k, n_components=2
        )
        
        final_df_split = pd.merge(df_split_unscaled, cluster_labels, on='ID')

        analyze_and_interpret_clusters(
            df_split=final_df_split,
            df_full_unscaled=df_unscaled_for_lookup,
            cols_for_this_split=COL_DEFINITIONS[split_name],
            base_name=split_name
        )

    print("\nClustering pipeline and analysis finished for all groups.")

if __name__ == "__main__":
    main()