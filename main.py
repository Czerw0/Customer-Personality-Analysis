import os
import pandas as pd
import data_loader
import data_processing
import data_split
import EDA as eda
#import model_training 

# Define constants for paths
RAW_DATA_PATH = '00_raw_data/marketing_campaign.csv'
PROCESSED_DATA_PATH = '01_data_processed/processed_marketing_data.csv'
SPLIT_DATA_PATH = '02_data_split/split_marketing_data'
REPORTS_DIR = '03_reports_and_results'
CHARTS_DIR = os.path.join(REPORTS_DIR, 'charts')
RESULTS_PATH = os.path.join(REPORTS_DIR, 'model_evaluation_report.txt')

def main():
    print("Starting customer personality cluster Pipeline")

    # 1. Create necessary directories
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    print(f"Output directories created/verified.")

    # 2. Load Data and run EDA
    df = data_loader.load_raw_data(RAW_DATA_PATH)
    eda.simple_eda(df)

    # 3 - Data Processing 
    df = data_processing.processing(df)  # Assign the returned DataFrame
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")
    eda.eda(PROCESSED_DATA_PATH, CHARTS_DIR)
    
    # 3.5 - Advanced Data Processing
    df = data_processing.advanced_processing(df)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Advanced processed data saved to {PROCESSED_DATA_PATH}")
    print("Data processing completed.")

    # 4. Split Data by Marketing 4Ps
    df = pd.read_csv(PROCESSED_DATA_PATH)  # Reload the processed data
    data_split.split_by_marketing_4ps(df)
    print("Data split by Marketing 4Ps completed.")


if __name__ == "__main__":
    main()