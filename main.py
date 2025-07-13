import os
import pandas as pd
import data_loader
import data_processing
import data_split
import data_clustering
from analyze_clusters import analyze_and_interpret_clusters

# --- Define Constants for Paths ---

# Data directories
RAW_DATA_PATH = '00_raw_data/marketing_campaign.csv'
PROCESSED_DATA_DIR = '01_data_processed'
SPLIT_DATA_DIR = '02_data_split'

# Report directories
BASE_REPORTS_DIR = '03_reports_and_results'
REPORTS_DIR_EDA = os.path.join(BASE_REPORTS_DIR, 'charts')
REPORTS_CLUSTER_PLOTS_DIR = os.path.join(BASE_REPORTS_DIR, 'cluster_plots')
REPORTS_K_EVAL_DIR = os.path.join(BASE_REPORTS_DIR, 'k_evaluation')
REPORTS_SCORES_DIR = os.path.join(BASE_REPORTS_DIR, 'scores')

# File paths
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'processed_marketing_data.csv')
HOLISTIC_CLUSTERED_PATH = os.path.join(PROCESSED_DATA_DIR, 'holistic_clustered_data.csv')


def main():
    """
    Main function to run the customer personality clustering pipeline.
    """
    print("Starting Customer Personality Cluster Pipeline")

    # 1. Create necessary directories
    all_dirs = [
        PROCESSED_DATA_DIR,
        SPLIT_DATA_DIR,
        REPORTS_DIR_EDA,
        REPORTS_CLUSTER_PLOTS_DIR,
        REPORTS_K_EVAL_DIR,
        REPORTS_SCORES_DIR,
    ]
    for directory in all_dirs:
        os.makedirs(directory, exist_ok=True)
    print("Output directories created/verified.")

    # 2. Load and Process Data
    # ... (code is unchanged)

    print("\n--- Starting Clustering Process ---")

    # --- Section for a single clustering on ALL data ---
    print("\n--- Processing a single cluster on all data ---")
    holistic_df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Evaluate and print the suggested k for the holistic dataset
    suggested_k_holistic = data_clustering.evaluate_k_range(holistic_df, split_name="holistic_all_data")
    print(f"Automated suggestion for holistic k = {suggested_k_holistic}")
    
    final_k_holistic = 2
    print(f"Using final k = {final_k_holistic} for holistic clustering.")
    
    holistic_clustered_df = data_clustering.cluster_with_pca(
        df=holistic_df, split_name="holistic_all_data", n_clusters=final_k_holistic, n_components=3
    )
    holistic_clustered_df.to_csv(HOLISTIC_CLUSTERED_PATH, index=False)
    print(f"Holistic clustered data saved to {HOLISTIC_CLUSTERED_PATH}")
    analyze_and_interpret_clusters(clustered_file_path=HOLISTIC_CLUSTERED_PATH)
    
    print("\n--- Starting Clustering Process for 4P Splits ---")

    # --- Cluster the 'People' data ---
    print("\n--- Processing 'people' split ---")
    people_df = pd.read_csv(os.path.join(SPLIT_DATA_DIR, "people_split.csv"))
    people_clustered_path = os.path.join(SPLIT_DATA_DIR, "people_clustered.csv")
    
    # Evaluate and print the suggested k for the 'people' split
    suggested_k_people = data_clustering.evaluate_k_range(df=people_df, split_name="people")
    print(f"Automated suggestion for 'people' k = {suggested_k_people}")

    final_k_people = 6 #Based on the silhuette score 
    print(f"Using final k = {final_k_people} for 'people' clustering.")
    people_clustered_df = data_clustering.cluster_with_pca(
        df=people_df, split_name="people", n_clusters=final_k_people, n_components=2
    )
    people_clustered_df.to_csv(people_clustered_path, index=False)
    print(f"'People' clustered data saved to {people_clustered_path}")
    analyze_and_interpret_clusters(people_clustered_path)

    # --- Cluster the 'Products' data ---
    print("\n--- Processing 'products' split ---")
    products_df = pd.read_csv(os.path.join(SPLIT_DATA_DIR, "products_split.csv"))
    products_clustered_path = os.path.join(SPLIT_DATA_DIR, "products_clustered.csv")

    # Evaluate and print the suggested k for the 'products' split
    suggested_k_products = data_clustering.evaluate_k_range(df=products_df, split_name="products")
    print(f"Automated suggestion for 'products' k = {suggested_k_products}")

    final_k_products = 2 # Based on the silhuette score
    print(f"Using final k = {final_k_products} for 'products' clustering.")
    products_clustered_df = data_clustering.cluster_with_pca(
        df=products_df, split_name="products", n_clusters=final_k_products, n_components=2
    )
    products_clustered_df.to_csv(products_clustered_path, index=False)
    print(f"'Products' clustered data saved to {products_clustered_path}")
    analyze_and_interpret_clusters(products_clustered_path)

    # --- Cluster the 'Promotion' data ---
    print("\n--- Processing 'promotion' split ---")
    promotion_df = pd.read_csv(os.path.join(SPLIT_DATA_DIR, "promotion_split.csv"))
    promotion_clustered_path = os.path.join(SPLIT_DATA_DIR, "promotion_clustered.csv")
    
    # Evaluate and print the suggested k for the 'promotion' split
    suggested_k_promotion = data_clustering.evaluate_k_range(df=promotion_df, split_name="promotion")
    print(f"Automated suggestion for 'promotion' k = {suggested_k_promotion}")

    final_k_promotion = 10
    print(f"Using final k = {final_k_promotion} for 'promotion' clustering.")
    promotion_clustered_df = data_clustering.cluster_with_pca(
        df=promotion_df, split_name="promotion", n_clusters=final_k_promotion, n_components=2
    )
    promotion_clustered_df.to_csv(promotion_clustered_path, index=False)
    print(f"'Promotion' clustered data saved to {promotion_clustered_path}")
    analyze_and_interpret_clusters(promotion_clustered_path)

    # --- Cluster the 'Place' data ---
    print("\n--- Processing 'place' split ---")
    place_df = pd.read_csv(os.path.join(SPLIT_DATA_DIR, "place_split.csv"))
    place_clustered_path = os.path.join(SPLIT_DATA_DIR, "place_clustered.csv")

    # Evaluate and print the suggested k for the 'place' split
    suggested_k_place = data_clustering.evaluate_k_range(df=place_df, split_name="place")
    print(f"Automated suggestion for 'place' k = {suggested_k_place}")

    final_k_place = 2
    print(f"Using final k = {final_k_place} for 'place' clustering.")
    place_clustered_df = data_clustering.cluster_with_pca(
        df=place_df, split_name="place", n_clusters=final_k_place, n_components=2
    )
    place_clustered_df.to_csv(place_clustered_path, index=False)
    print(f"'Place' clustered data saved to {place_clustered_path}")
    analyze_and_interpret_clusters(place_clustered_path)

    print("\nClustering pipeline and analysis finished for all groups.")

    # Performing cluser

if __name__ == "__main__":
    main()