import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def simple_eda(df_or_path):
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path


    print("Performing Basic Exploratory Data Analysis (EDA)...")

    print("Dataset")
    print(df.head())
    print("Sumamry Statistics:")
    print(df.describe(include='all').T)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData info:")
    print(df.info())



def eda(df_or_path, output_dir='03_reports_and_results/charts'):
    """
    Perform Exploratory Data Analysis (EDA) on the given DataFrame or CSV file path.
    
    Parameters:
    df_or_path (pd.DataFrame or str): The DataFrame to analyze or path to CSV file.
    
    Returns:
    None
    """
    # If a string is passed, assume it's a file path and read the CSV
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path

    os.makedirs(output_dir, exist_ok=True)
    print("Performing Exploratory Data Analysis on cleaned dataset(EDA)...")

    # Set seaborn style
    sns.set_style(style="whitegrid")
    
    # Plot distribution of numerical features
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'distribution_{feature}.png'))
        plt.close()


    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=['number'])  # Only numeric columns
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

    # Boxplots
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[feature])
        plt.title(f'Boxplot of {feature}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'boxplot_{feature}.png'))
        plt.close()

    # Count plots for categorical features
    categorical_features = df.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[feature], order=df[feature].value_counts().index)
        plt.title(f'Count Plot of {feature}')
        plt.xlabel('Count')
        plt.ylabel(feature)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'countplot_{feature}.png'))
        plt.close()
    
    print(f"EDA charts and summary saved to '{output_dir}'.")