import os
import pandas as pd
import data_loader
import data_processing
import data_split
import EDA as eda
import data_clustering

# Define constants for paths
RAW_DATA_PATH = '00_raw_data/marketing_campaign.csv'
PROCESSED_DATA_PATH = '01_data_processed/processed_marketing_data.csv'
SPLIT_DATA_DIR = '02_data_split'
REPORTS_DIR = '03_reports_and_results'

def main():
    """
    Main function to run the customer personality clustering pipeline.
    """
    print("Starting Customer Personality Cluster Pipeline")

    # 1. Create necessary directories
    # This line gets the directory name (e.g., '01_data_processed') from the full path
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("Output directories created/verified.")

    # 2. Load and Process Data
    df = data_loader.load_raw_data(RAW_DATA_PATH)
    df = data_processing.processing(df)
    df = data_processing.advanced_processing(df)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

    # 3. Split Data by Marketing 4Ps
    data_split.split_by_marketing_4ps(df, output_dir=SPLIT_DATA_DIR)
    print("Data split by Marketing 4Ps completed.")

    print("\n--- Starting Clustering Process ---")

    # --- NEW: Section for a single, holistic clustering on ALL data ---
    print("\n--- Processing a single HOLISTIC cluster on all data ---")
    
    # Use the main processed dataframe before it was split
    holistic_df = pd.read_csv(PROCESSED_DATA_PATH)
    
    suggested_k_holistic = data_clustering.evaluate_k_range(holistic_df, split_name="holistic_all_data")
    print(f"Automated suggestion for holistic k = {suggested_k_holistic}")
    

    final_k_holistic = 2
    
    print(f"Using final k = {final_k_holistic} for holistic clustering.")
    
    # We save the clustered result to its own file
    holistic_clustered_df = data_clustering.cluster_with_pca(
        df=holistic_df,
        split_name="holistic_all_data",
        n_clusters=final_k_holistic,
        n_components=3 # Use 3 components for a better model
    )
    # Save the clustered data to the main processed data folder
    holistic_clustered_df.to_csv(os.path.join('01_data_processed', 'holistic_clustered_data.csv'), index=False)
    print("Holistic clustering complete.")
    
    # NOW, you can also analyze this new holistic cluster
    from analyze_clusters import create_cluster_profile
    create_cluster_profile(
        clustered_file_path=os.path.join('01_data_processed', 'holistic_clustered_data.csv')
    )
    
    print("\n--- Starting Clustering Process for 4P Splits ---")

    # --- Section 1: Cluster the 'People' data ---
    print("\n--- Processing 'people' split ---")
    people_df = pd.read_csv(os.path.join(SPLIT_DATA_DIR, "people_split.csv"))
    
    # You can run the evaluation to see the suggestion, but it won't be used unless you want it to
    suggested_k_people = data_clustering.evaluate_k_range(people_df, split_name="people")
    print(f"Automated suggestion for 'people' k = {suggested_k_people}")
    
    final_k_people = 6 
    
    print(f"Using final k = {final_k_people} for 'people' clustering.")
    data_clustering.cluster_with_pca(
        df=people_df,
        split_name="people",
        n_clusters=final_k_people,
        n_components=2 # Using 2D for the plot
    )
    print("Clustering complete for 'people'.")

    # --- Section 2: Cluster the 'Products' data ---
    print("\n--- Processing 'products' split ---")
    products_df = pd.read_csv(os.path.join(SPLIT_DATA_DIR, "products_split.csv"))

    suggested_k_products = data_clustering.evaluate_k_range(products_df, split_name="products")
    print(f"Automated suggestion for 'products' k = {suggested_k_products}")

    final_k_products = 2 

    print(f"Using final k = {final_k_products} for 'products' clustering.")
    data_clustering.cluster_with_pca(
        df=products_df,
        split_name="products",
        n_clusters=final_k_products,
        n_components=2
    )
    print("Clustering complete for 'products'.")

    # --- Section 3: Cluster the 'Promotion' data ---
    print("\n--- Processing 'promotion' split ---")
    promotion_df = pd.read_csv(os.path.join(SPLIT_DATA_DIR, "promotion_split.csv"))
    
    suggested_k_promotion = data_clustering.evaluate_k_range(promotion_df, split_name="promotion")
    print(f"Automated suggestion for 'promotion' k = {suggested_k_promotion}")

    final_k_promotion = 10 
    
    print(f"Using final k = {final_k_promotion} for 'promotion' clustering.")
    data_clustering.cluster_with_pca(
        df=promotion_df,
        split_name="promotion",
        n_clusters=final_k_promotion,
        n_components=2
    )
    print("Clustering complete for 'promotion'.")

    # --- Section 4: Cluster the 'Place' data ---
    print("\n--- Processing 'place' split ---")
    place_df = pd.read_csv(os.path.join(SPLIT_DATA_DIR, "place_split.csv"))
    
    suggested_k_place = data_clustering.evaluate_k_range(place_df, split_name="place")
    print(f"Automated suggestion for 'place' k = {suggested_k_place}")
    
    final_k_place = 2 

    print(f"Using final k = {final_k_place} for 'place' clustering.")
    data_clustering.cluster_with_pca(
        df=place_df,
        split_name="place",
        n_clusters=final_k_place,
        n_components=2
    )
    print("Clustering complete for 'place'.")

    print("\nClustering pipeline finished for all groups.")

    # --- NEW SECTION: Analyze and Profile the Clusters ---
    print("\n--- Starting Cluster Profiling and Comparison ---")
    
    # Define the paths to your clustered files
    clustered_files = {
        "people": os.path.join(SPLIT_DATA_DIR, "people_clustered.csv"),
        "products": os.path.join(SPLIT_DATA_DIR, "products_clustered.csv"),
        "promotion": os.path.join(SPLIT_DATA_DIR, "promotion_clustered.csv"),
        "place": os.path.join(SPLIT_DATA_DIR, "place_clustered.csv")
    }
    
    # Import the new function
    from analyze_clusters import create_cluster_profile

    # Run the profiling for each clustered file
    for name, path in clustered_files.items():
        if os.path.exists(path):
            create_cluster_profile(clustered_file_path=path)
        else:
            print(f"Warning: Clustered file not found at {path}. Skipping profile creation.")

    print("All cluster profiles have been created.")


if __name__ == "__main__":
    main()