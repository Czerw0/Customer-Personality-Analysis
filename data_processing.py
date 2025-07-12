import os
import pandas as pd


#Unecessary data drop 
def processing(df):
    
    #droping columns that are not needed for analysis
    columns_to_drop = ['ID']
    if 'ID' in df.columns:
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"Dropping columns: {columns_to_drop}")
    else:
        print("No columns to drop, 'ID' not found in DataFrame.")
    
    #total amount spend
    # based on "MinWines + MinFruits + MinMeatProducts + MinFishProducts + MinSweetProducts + MinGoldProds"

    
    df['Total_Spend'] = (
        df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
        df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
    )
    print("Calculated 'Total_Spend' based on product categories.")

    
    # Total number of purcheses
    if all(col in df.columns for col in ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']):
        df['Total_Purchases'] = (df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'])
        print("Calculated 'Total_Purchases' based on purchase categories.")
    else:
        print("One or more columns for calculating 'Total_Purchases' are missing.")

    #Cretaing age column
    if 'Year_Birth' in df.columns:
        current_year = pd.Timestamp.now().year
        df['Age'] = current_year - df['Year_Birth']
        print("Calculated 'Age' from 'Year_Birth'.")
        # drop 'Year_of_Birth' column after calculating 'Age'
        df = df.drop(columns=['Year_Birth'], errors='ignore')
        print("Dropped 'Year_Birth' column after calculating 'Age'.")
    else:
        print("'Year_Birth' column not found in the DataFrame.")
    

    #Categorize 'Age' into bins

    if 'Age' in df.columns:
        bins = [0, 18, 30, 45, 60, 75, float('inf')]
        labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
        df['Age_Category'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
        print("Categorized 'Age' into bins.")


    #Z Revenue drop and Z_CostContact- not relevant - every customer has the same value
    if 'Z_Revenue' in df.columns:
        df = df.drop(columns=['Z_Revenue'], errors='ignore')
        print("Dropped 'Z_Revenue' column as it is not relevant for analysis.")
    else:
        print("'Z_Revenue' column not found in the DataFrame.")

    if 'Z_CostContact' in df.columns:
        df = df.drop(columns=['Z_CostContact'], errors='ignore')
        print("Dropped 'Z_CostContact' column as it is not relevant for analysis.")
    else:
        print("'Z_CostContact' column not found in the DataFrame.")
    
    # Change rows with missing values in 'Income'
    if 'Income' in df.columns:
        df['Income'] = df['Income'].fillna(df['Income'].mean())
        print("Filled missing values in 'Income' with the mean value.")
    else:
        print("'Income' column not found in the DataFrame.")

     # Change the date of the customer enrollment to days with the company 
    if 'Dt_Customer' in df.columns:
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce', dayfirst=True)
        df['Days_with_Company'] = (pd.Timestamp.now() - df['Dt_Customer']).dt.days
        df = df.drop(columns=['Dt_Customer'], errors='ignore')
        print("Dropped 'Dt_Customer' column after calculating 'Days_with_Company'.")
    else:
        print("'Dt_Customer' column not found in the DataFrame.")
    

    #Based on advanced EDA:
    # Drop rows where 'Marital_Status' is 'YOLO' or 'Absurd'
    if 'Marital_Status' in df.columns:
        df = df[~df['Marital_Status'].isin(['YOLO', 'Absurd'])]
        # Replace 'Alone' with 'Single' in 'Marital_Status'
        df['Marital_Status'] = df['Marital_Status'].replace('Alone', 'Single')
        print("Dropped rows with 'YOLO' or 'Absurd' in 'Marital_Status' and replaced 'Alone' with 'Single'.")
    else:
        print("'Marital_Status' column not found in the DataFrame.")
    print(df.head(), sep = "\n")
    print(df.info())

    return df

    
    
    