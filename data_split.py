import os

def split_by_marketing_4ps(df, output_dir="02_data_split"):
    """
    Splits the DataFrame into People, Products, Promotion, and Place groups and saves each as a CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    people_cols = [
    'Income', 'Age_Category_encoded', 'Marital_Status_encoded', 'Education_encoded',
    'Days_with_Company', 'Kidhome', 'Teenhome', 'Recency', 'Complain'
    ]
    product_cols = [
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'Total_Spend', 'Total_Purchases'
    ]
    promotion_cols = [
        'NumDealsPurchases', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
        'AcceptedCmp4', 'AcceptedCmp5', 'Response'
    ]
    place_cols = [
        'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'
    ]

    # Only keep columns that exist in the DataFrame
    people = df[[col for col in people_cols if col in df.columns]]
    products = df[[col for col in product_cols if col in df.columns]]
    promotion = df[[col for col in promotion_cols if col in df.columns]]
    place = df[[col for col in place_cols if col in df.columns]]

    people.to_csv(os.path.join(output_dir, "people_split.csv"), index=False)
    products.to_csv(os.path.join(output_dir, "products_split.csv"), index=False)
    promotion.to_csv(os.path.join(output_dir, "promotion_split.csv"), index=False)
    place.to_csv(os.path.join(output_dir, "place_split.csv"), index=False)

    print(f"Data split and saved to '{output_dir}'.")