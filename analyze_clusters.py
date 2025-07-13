# analyze_clusters.py

import pandas as pd
import os

# This dictionary is our "decoder ring." It translates the technical column names
# into plain, understandable English for our final reports.
COLUMN_DECODER = {
    'ID': 'Customer ID',
    'Year_Birth': 'Birth Year',
    'Education': 'Education Level',
    'Marital_Status': 'Marital Status',
    'Income': 'Annual Income',
    'Kidhome': 'Number of Children',
    'Teenhome': 'Number of Teenagers',
    'Dt_Customer': 'Date of Enrollment',
    'Recency': 'Days Since Last Purchase',
    'MntWines': 'Spent on Wine',
    'MntFruits': 'Spent on Fruits',
    'MntMeatProducts': 'Spent on Meat',
    'MntFishProducts': 'Spent on Fish',
    'MntSweetProducts': 'Spent on Sweets',
    'MntGoldProds': 'Spent on Gold Products',
    'NumDealsPurchases': 'Purchases with Discount',
    'NumWebPurchases': 'Web Purchases',
    'NumCatalogPurchases': 'Catalog Purchases',
    'NumStorePurchases': 'In-Store Purchases',
    'NumWebVisitsMonth': 'Monthly Web Visits',
    'AcceptedCmp1': 'Accepted Campaign 1',
    'AcceptedCmp2': 'Accepted Campaign 2',
    'AcceptedCmp3': 'Accepted Campaign 3',
    'AcceptedCmp4': 'Accepted Campaign 4',
    'AcceptedCmp5': 'Accepted Campaign 5',
    'Response': 'Accepted Last Campaign',
    'Complain': 'Has Complained in 2 Years',
    'Cluster': 'Cluster',
    'Age': 'Age',
    'Spent': 'Total Spending',
    'Living_With': 'Household Status',
    'Children': 'Total Children',
    'Family_Size': 'Family Size',
    'Is_Parent': 'Is a Parent',
    'Cluster_Size': 'Number of Customers' # We will add this row
}

def analyze_and_interpret_clusters(clustered_file_path, output_dir='03_reports_and_results/cluster_profiles'):
    """
    Loads clustered data, creates a human-readable profile for each cluster,
    and prints a clear summary to the console.

    Args:
        clustered_file_path (str): Path to the CSV file with cluster assignments.
        output_dir (str): Directory to save the detailed CSV profiles.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(clustered_file_path).replace('_clustered.csv', '')
    
    print(f"\n{'='*25}")
    print(f"  Analyzing Clusters for: {base_name.upper()}")
    print(f"{'='*25}\n")

    try:
        df = pd.read_csv(clustered_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file {clustered_file_path}")
        return

    if 'Cluster' not in df.columns:
        print(f"Error: 'Cluster' column not found in {clustered_file_path}. Cannot analyze.")
        return

    # --- 1. Basic Profiling ---
    # We want to profile the original features, not PCA components
    cols_to_profile = [col for col in df.columns if not col.startswith('PCA_') and df[col].dtype in ['int64', 'float64']]
    
    # Calculate the average for each feature across the ENTIRE dataset (our baseline)
    overall_avg = df[cols_to_profile].mean().to_frame(name='Overall_Average')
    
    # Calculate the average for each feature WITHIN each cluster
    cluster_profile = df.groupby('Cluster')[cols_to_profile].mean().T
    
    # Count the number of customers in each cluster
    cluster_sizes = df['Cluster'].value_counts().sort_index()
    cluster_profile.loc['Cluster_Size'] = cluster_sizes
    
    # --- 2. Create a "Comparison to Average" View ---
    # This is the most powerful view for interpretation.
    # It shows how much higher/lower a cluster's average is compared to the overall average.
    comparison_profile = cluster_profile.iloc[:-1].div(overall_avg.iloc[:, 0], axis=0) * 100
    comparison_profile = comparison_profile.round(0) # Express as a percentage
    
    # --- 3. Make it Readable and Display It ---
    # Use the decoder to translate column names to English
    readable_profile = cluster_profile.rename(index=COLUMN_DECODER)
    readable_comparison = comparison_profile.rename(index=COLUMN_DECODER)
    
    # Add cluster size to the top of the main profile
    readable_profile = readable_profile.iloc[[-1] + list(range(len(readable_profile)-1))]
    
    print("--- Profile Summary (Average value for each feature) ---")
    # We only show the most important columns in the console for brevity
    display_cols = ['Annual Income', 'Total Spending', 'Days Since Last Purchase', 
                    'Web Purchases', 'In-Store Purchases', 'Total Children', 'Age']
    # Filter for columns that actually exist in the profile
    display_cols_exist = [col for col in display_cols if col in readable_profile.index]
    
    print(readable_profile.loc[['Number of Customers'] + display_cols_exist].to_string())
    print("\n" + "="*80 + "\n")
    
    print("--- How Each Cluster Compares to the Average Customer (as a %) ---")
    print(" (e.g., 150 means 50% higher than average; 70 means 30% lower than average)\n")
    print(readable_comparison.loc[display_cols_exist].to_string())
    print("\n")

    # --- 4. Save Detailed Profiles to CSV ---
    # Save both the raw numbers and the comparison percentages for later use
    profile_output_path = os.path.join(output_dir, f'{base_name}_profile_summary.csv')
    comparison_output_path = os.path.join(output_dir, f'{base_name}_profile_comparison_to_average.csv')
    
    readable_profile.to_csv(profile_output_path)
    readable_comparison.to_csv(comparison_output_path)
    
    print(f"Successfully saved detailed profiles to: {output_dir}")