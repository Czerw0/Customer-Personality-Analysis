import pandas as pd
import os

# Defines user-friendly names for columns in reports
COLUMN_DECODER = {
    'ID': 'Customer ID', 'Education': 'Education', 'Living_With': 'Living With', 'Income': 'Annual Income',
    'Kidhome': 'Number of Children', 'Teenhome': 'Number of Teenagers',
    'Recency': 'Days Since Last Purchase', 'MntWines': 'Spent on Wine',
    'MntFruits': 'Spent on Fruits', 'MntMeatProducts': 'Spent on Meat',
    'MntFishProducts': 'Spent on Fish', 'MntSweetProducts': 'Spent on Sweets',
    'MntGoldProds': 'Spent on Gold Products', 'NumDealsPurchases': 'Purchases with Discount',
    'NumWebPurchases': 'Web Purchases', 'NumCatalogPurchases': 'Catalog Purchases',
    'NumStorePurchases': 'In-Store Purchases', 'NumWebVisitsMonth': 'Monthly Web Visits',
    'AcceptedCmp1': 'Accepted Campaign 1', 'AcceptedCmp2': 'Accepted Campaign 2',
    'AcceptedCmp3': 'Accepted Campaign 3', 'AcceptedCmp4': 'Accepted Campaign 4',
    'AcceptedCmp5': 'Accepted Campaign 5', 'Response': 'Accepted Last Campaign',
    'Complain': 'Has Complained in 2 Years', 'Cluster': 'Cluster', 'Age': 'Age',
    'Spent': 'Total Spending', 'Children': 'Total Children', 'Family_Size': 'Family Size',
    'Is_Parent': 'Is a Parent', 'Days_Enrolled': 'Days Enrolled',
    'Cluster_Size': 'Number of Customers'
}

def analyze_and_interpret_clusters(df_split, df_full_unscaled, cols_for_this_split, base_name, output_dir='03_reports_and_results/cluster_profiles'):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Analyzing Clusters for: {base_name.upper()} ---")

    if 'Cluster' not in df_split.columns:
        print("Error: 'Cluster' column not found.")
        return

    # Merge cluster labels onto the full dataset for a complete analysis frame
    df_analysis = pd.merge(df_full_unscaled, df_split[['ID', 'Cluster']], on='ID')
    
    # Generate summary statistics
    cols_to_profile = [col for col in df_analysis.columns if df_analysis[col].dtype in ['int64', 'float64', 'uint8']]
    cluster_profile = df_analysis.groupby('Cluster')[cols_to_profile].mean().T
    cluster_sizes = df_analysis['Cluster'].value_counts().sort_index()
    cluster_profile.loc['Cluster_Size'] = cluster_sizes
    readable_profile = cluster_profile.rename(index=COLUMN_DECODER)
    readable_profile = readable_profile.iloc[[-1] + list(range(len(readable_profile)-1))]

    # Print a concise, holistic summary to the console
    print("Profile Summary:")
    key_features = ['Annual Income', 'Spent', 'Age', 'Children', 'Family_Size', 'Days_Enrolled', 'Recency']
    display_cols = [col for col in readable_profile.index if col in [COLUMN_DECODER.get(k, k) for k in key_features]]
    summary_table = readable_profile.loc[['Number of Customers'] + display_cols].round(0)
    print(summary_table.to_string())

    # Save a detailed, multi-sheet Excel report for interactive analysis
    excel_output_path = os.path.join(output_dir, f'{base_name}_cluster_analysis.xlsx')
    
    with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
        summary_table.to_excel(writer, sheet_name='Summary_Metrics')

        # Save samples for each cluster to its own sheet in the Excel file
        unique_clusters = sorted(df_analysis['Cluster'].unique())
        for cluster_id in unique_clusters:
            sample_ids = df_analysis[df_analysis['Cluster'] == cluster_id].sample(
                n=min(15, len(df_analysis[df_analysis['Cluster'] == cluster_id])),
                random_state=42
            )['ID']
            
            sample_full_profiles = df_full_unscaled[df_full_unscaled['ID'].isin(sample_ids)]
            sample_full_profiles = pd.merge(sample_full_profiles, df_split[['ID', 'Cluster']], on='ID')

            # Reorder columns for clarity
            priority_cols = ['ID', 'Cluster'] + [c for c in cols_for_this_split if c not in ['ID', 'Cluster']]
            other_cols = [col for col in sample_full_profiles.columns if col not in priority_cols]
            final_sample_table = sample_full_profiles[priority_cols + other_cols]
            final_sample_table.to_excel(writer, sheet_name=f'Cluster_{cluster_id}_Samples', index=False)
            
    print(f"\nFull interactive analysis saved to Excel file:\n--> {excel_output_path}")
    print("-" * 60)