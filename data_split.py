import os
import pandas as pd

# Define the columns for each Marketing 4P split
# These definitions are now on the "public workbench" and can be imported.

PEOPLE_COLS = [
    'ID', 'Age', 'Education', 'Living_With', 'Income', 'Kidhome', 'Teenhome',
    'Children', 'Family_Size', 'Is_Parent', 'Days_Enrolled', 'Recency', 'Complain'
]
PRODUCTS_COLS = [
    'ID', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'Spent'
]
PROMOTION_COLS = [
    'ID', 'NumDealsPurchases', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
    'AcceptedCmp4', 'AcceptedCmp5', 'Response'
]
PLACE_COLS = [
    'ID', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
    'NumWebVisitsMonth'
]

COL_DEFINITIONS = {
    "people": PEOPLE_COLS,
    "products": PRODUCTS_COLS,
    "promotion": PROMOTION_COLS,
    "place": PLACE_COLS
}

def split_by_marketing_4ps(df, output_dir):
    """
    Splits the dataframe into four smaller dataframes based on the Marketing 4Ps,
    using the globally defined COL_DEFINITIONS.
    """
    os.makedirs(output_dir, exist_ok=True)

    
    for split_name, cols in COL_DEFINITIONS.items():
        # Filter for columns that actually exist in the passed dataframe
        cols_to_keep = [col for col in cols if col in df.columns]
        
        if not cols_to_keep:
            print(f"Warning: No columns found for split '{split_name}'. Skipping.")
            continue
            
        split_df = df[cols_to_keep]
        split_df.to_csv(os.path.join(output_dir, f"{split_name}_split.csv"), index=False)

    print(f"Data split and saved to '{output_dir}'.")