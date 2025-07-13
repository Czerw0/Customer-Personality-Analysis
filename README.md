# Customer Personality Analysis for Targeted Marketing

This project performs an in-depth analysis of customer data to identify distinct personality segments. By leveraging clustering algorithms, we can group customers based on their demographic, purchasing, and behavioral patterns. The ultimate goal is to provide actionable insights that can be used to develop tailored marketing strategies for each customer segment, thereby improving campaign effectiveness and customer engagement.

This pipeline is automated to process raw data, perform exploratory data analysis (EDA), evaluate optimal cluster numbers, and generate detailed analysis reports in Excel for easy interpretation.

**Data Source:** [Customer Personality Analysis on Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

---

## Project Structure

The repository is organized to ensure a clean and modular workflow, separating data, code, and reports.

├── 00_raw_data/

│ └── marketing_campaign.csv - The original, unprocessed dataset

├── 01_data_processed/

│ └── (empty by default) - Will contain processed data (not tracked by Git)

├── 02_data_split/

│ └── (empty by default) Will contain data splits (scaled and unscaled)

├── 03_reports_and_results/

│ ├── charts/ - Stores EDA visualizations

│ ├── cluster_plots/ - Stores cluster visualizations (PCA plots)

│ ├── cluster_profiles/ - Stores the final Excel analysis reports

│ └── k_evaluation/ - Stores Elbow and Silhouette score plots

├── .gitignore

├── analyze_clusters.py - Module for analyzing and interpreting clusters

├── data_clustering.py - Module for performing clustering algorithms

├── data_loader.py - Module for loading the raw data

├── data_processing.py - Module for cleaning and feature engineering

├── data_split.py - Module for splitting data into 4P categories

├── EDA.py - Module for exploratory data analysis

├── main.py - The main pipeline controller script

└── README.md

## Column Descriptions

The dataset contains the following features:

### People

-   **ID:** Unique customer identifier
-   **Year_Birth:** Customer's birth year
-   **Education:** Customer's education level
-   **Marital_Status:** Customer's marital status
-   **Income:** Customer's yearly household income
-   **Kidhome:** Number of children in the household
-   **Teenhome:** Number of teenagers in the household
-   **Dt_Customer:** Date of customer's enrollment with the company
-   **Recency:** Number of days since the last purchase
-   **Complain:** 1 if the customer complained in the last 2 years, 0 otherwise

### Products

-   **MntWines:** Amount spent on wine in the last 2 years
-   **MntFruits:** Amount spent on fruits in the last 2 years
-   **MntMeatProducts:** Amount spent on meat in the last 2 years
-   **MntFishProducts:** Amount spent on fish in the last 2 years
-   **MntSweetProducts:** Amount spent on sweets in the last 2 years
-   **MntGoldProds:** Amount spent on gold products in the last 2 years

### Promotion

-   **NumDealsPurchases:** Number of purchases made with a discount
-   **AcceptedCmp1:** 1 if the customer accepted the offer in the 1st campaign, 0 otherwise
-   **AcceptedCmp2:** 1 if the customer accepted the offer in the 2nd campaign, 0 otherwise
-   **AcceptedCmp3:** 1 if the customer accepted the offer in the 3rd campaign, 0 otherwise
-   **AcceptedCmp4:** 1 if the customer accepted the offer in the 4th campaign, 0 otherwise
-   **AcceptedCmp5:** 1 if the customer accepted the offer in the 5th campaign, 0 otherwise
-   **Response:** 1 if the customer accepted the offer in the last campaign, 0 otherwise

### Place

-   **NumWebPurchases:** Number of purchases made through the company’s website
-   **NumCatalogPurchases:** Number of purchases made using a catalogue
-   **NumStorePurchases:** Number of purchases made directly in stores
-   **NumWebVisitsMonth:** Number of visits to the company’s website in the last month

---

## How to Run the Pipeline

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Czerw0/Customer-Personality-Analysis.git
    cd Customer-Personality-Analysis
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment. Ensure you have the required libraries installed.
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
    ```

3.  **Execute the Main Script:**
    Run the `main.py` script from the root of the project directory. This will execute the entire pipeline from data loading to report generation.
    ```bash
    python main.py
    ```

---

## Pipeline Overview

The `main.py` script automates the following sequence of operations:

1.  **Load Data:** Loads the raw `marketing_campaign.csv` file.
2.  **Initial EDA:** Performs a basic check of the raw data (shape, missing values).
3.  **Process Data:** Cleans the data and engineers new features (e.g., `Age`, `Total Spending`, `Family_Size`).
4.  **Full EDA:** Generates and saves a comprehensive set of visualizations (histograms, boxplots, correlation heatmap) based on the cleaned data.
5.  **Create Datasets:** Prepares two versions of the data: an unscaled version for analysis and a scaled version for the clustering algorithms.
6.  **Split Data:** Divides both datasets into four logical groups based on the Marketing 4Ps framework: **People, Products, Promotion, and Place**.
7.  **Cluster and Analyze:** For each of the four splits, the pipeline:
    -   Evaluates the optimal number of clusters (`k`) using the Elbow Method and Silhouette Scores.
    -   Performs K-Means clustering on the scaled data.
    -   Merges the resulting cluster labels back to the unscaled data.
    -   Generates a concise summary profile in the terminal.
    -   Saves a detailed, multi-sheet **Excel analysis report** in the `03_reports_and_results/cluster_profiles` directory for deep-dive analysis.

---

## Analysis and Insights

To be performed in teh feature 
---

## Technologies Used

-   **Python 3**
-   **pandas:** For data manipulation and analysis.
-   **NumPy:** For numerical operations.
-   **scikit-learn:** For data scaling, PCA, and K-Means clustering.
-   **matplotlib & seaborn:** For data visualization and generating charts.
-   **openpyxl:** For writing data to Excel files.
