import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


#Unecessary data drop 
import pandas as pd

def processing(df):
    """
    Performs data cleaning and feature engineering.
    - Creates new, useful columns ('Total_Spend', 'Age', etc.).
    - Cleans up bad data.
    - Fills missing values.
    - Returns a clean, human-readable dataframe.
    - IMPORTANT: Does NOT scale or encode data.
    """
    print("--- Starting Data Processing and Feature Engineering ---")

    # Feature Engineering: Total Spending
    mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['Spent'] = df[mnt_cols].sum(axis=1)

    # Feature Engineering: Household Composition
    df['Living_With'] = df['Marital_Status'].replace({
        "Married": "Partner", "Together": "Partner",
        "Absurd": "Single", "Widow": "Single", "YOLO": "Single",
        "Divorced": "Single", "Single": "Single", "Alone": "Single"
    })
    df['Children'] = df['Kidhome'] + df['Teenhome']
    df['Family_Size'] = df['Living_With'].replace({"Single": 1, "Partner": 2}) + df['Children']
    df['Is_Parent'] = (df['Children'] > 0).astype(int)

    # Feature Engineering: Age and Enrollment Duration
    current_year = pd.Timestamp.now().year
    df['Age'] = current_year - df['Year_Birth']
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
    df['Days_Enrolled'] = (pd.Timestamp.now() - df['Dt_Customer']).dt.days

    # Data Cleaning
    df['Income'] = df['Income'].fillna(df['Income'].median())
    df = df[df['Income'] < 600000] # Remove outliers
    df = df[df['Age'] < 90] # Remove outliers

    # Drop original columns that are now redundant or not useful
    cols_to_drop = ['Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue', 'Marital_Status']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    print("Data processing complete. New features created and data cleaned.")
    return df


def advanced_processing(df):
    """
    Perform advanced data processing on the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    
    Returns:
    pd.DataFrame: The processed DataFrame.
    """    
    
    #Encoding categorical variables
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    for feature in categorical_features:
        df[feature] = df[feature].astype('category')
        df[feature + '_encoded'] = df[feature].cat.codes
        print(f"Encoded '{feature}' into '{feature}_encoded'.")

    # Now drop all original categorical columns at once
    df = df.drop(columns=categorical_features, errors='ignore')

    #Scaling numerical features
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    print("Scaled numerical features using StandardScaler.")


    print(df.head())
    print(df.info())
    print("Advanced data processing completed.")
    return df