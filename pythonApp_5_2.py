#!/usr/bin/env python3
"""
Data Scraping + Cleaning + Normalization Pipeline
-------------------------------------------------
This script:
 1. Scrapes README.md from the AIBAS GitHub repo.
 2. Loads dataset03.csv (dirty).
 3. Performs:
        - Data cleaning
        - Outlier handling (IQR)
        - Normalization
 4. Stores final cleaned data in:
        UE_06_dataset04_joint_scraped_data.csv
"""

import os
import re
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------
#           SCRAPE README FROM GITHUB
# ---------------------------------------------------
RAW_README_URL = "https://github.com/MarcusGrum/AIBAS/blob/main/README.md"
OUTPUT_README = "scraped_readme.md"


def scrape_readme():
    """Download the README.md from GitHub (raw version)."""
    print("Scraping README.md...")
    response = requests.get(RAW_README_URL)

    if response.status_code != 200:
        raise Exception("Failed to fetch README.md from GitHub")

    with open(OUTPUT_README, "w", encoding="utf-8") as f:
        f.write(response.text)

    print("✓ README.md scraped and saved.\n")
    return response.text


# ---------------------------------------------------
#           OUTLIER HANDLING (IQR METHOD)
# ---------------------------------------------------
def handle_outliers(df, strategy="cap"):
    """
    strategy = "remove"  -> drop rows with outliers
    strategy = "cap"     -> cap outliers to IQR boundaries
    """
    print("Handling outliers using IQR method...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        if strategy == "remove":
            df = df[(df[col] >= lower) & (df[col] <= upper)]

        elif strategy == "cap":
            df[col] = np.where(df[col] < lower, lower, df[col])
            df[col] = np.where(df[col] > upper, upper, df[col])

    print("✓ Outliers processed.\n")
    return df


# ---------------------------------------------------
#           NORMALIZATION (MIN-MAX)
# ---------------------------------------------------
def normalize(df):
    print("Performing Min–Max normalization...")

    scaler = MinMaxScaler()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("✓ Normalization complete.\n")
    return df


# ---------------------------------------------------
#           DATA CLEANING
# ---------------------------------------------------
def clean_data(df):
    print("Cleaning dataset...")

    # remove leading/trailing whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # convert columns that look numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # handle missing values
    df = df.fillna(df.median(numeric_only=True))  # numeric NaN
    df = df.fillna("Unknown")                     # non-numeric NaN

    # drop duplicates
    df = df.drop_duplicates()

    print("✓ Data cleaning finished.\n")
    return df


# ---------------------------------------------------
#           LOAD AND PROCESS dataset03.csv
# ---------------------------------------------------
def load_and_process_dataset():
    print("Loading dataset03.csv...")

    if not os.path.exists("dataset03.csv"):
        raise FileNotFoundError("dataset03.csv not found in this directory.")

    df = pd.read_csv("dataset03.csv")

    print("Initial shape:", df.shape)

    df = clean_data(df)
    df = handle_outliers(df, strategy="cap")
    df = normalize(df)

    print("Final shape after cleaning:", df.shape, "\n")

    return df


# ---------------------------------------------------
#           MERGE SCRAPED + CLEANED DATA
# ---------------------------------------------------
def merge_data(scraped_text, df):
    """
    Joint dataset:
    - original cleaned dataset03.csv
    - plus the scraped README text, stored as a single column
    """
    print("Merging scraped README text into dataset...")

    scraped_df = pd.DataFrame({
        "readme_text": [scraped_text]
    })

    # merge by adding README text as global metadata row
    scraped_df = scraped_df.reindex(df.index.union([df.index.max() + 1]))

    joint_df = pd.concat([df, scraped_df], axis=1)

    print("✓ Merge complete.\n")
    return joint_df


# ---------------------------------------------------
#                 MAIN PIPELINE
# ---------------------------------------------------
def main():
    print("\n=== STARTING DATA PIPELINE ===\n")

    readme_text = scrape_readme()
    df = load_and_process_dataset()

    final_df = merge_data(readme_text, df)

    OUTPUT_FILE = "UE_06_dataset04_joint_scraped_data.csv"
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("✓ FINAL DATASET SAVED AS:", OUTPUT_FILE)
    print("\n=== ALL TASKS COMPLETED SUCCESSFULLY ===\n")


if __name__ == "__main__":
    main()
