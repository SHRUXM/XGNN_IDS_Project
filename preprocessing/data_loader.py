# ============================================================
# XGNN-Based Intrusion Detection System
# File: preprocessing/data_loader.py
# Author: Shreyas Santosh Shinde
# MSc Computer Science - Kirti College, Mumbai
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

def load_dataset(filepath):
    """
    Load the CICIDS2017 or UNSW-NB15 dataset from CSV file
    """
    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def clean_dataset(df):
    """
    Clean the dataset by handling missing values,
    duplicates and infinite values
    """
    print("\nCleaning dataset...")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows")

    # Remove missing values
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"Removed {before - after} rows with missing values")

    # Replace infinite values with NaN then drop
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f"Removed infinite values")

    print(f"Clean dataset shape: {df.shape}")
    return df


def encode_labels(df, label_column):
    """
    Encode attack labels into binary format
    0 = Benign (normal traffic)
    1 = Attack (malicious traffic)
    """
    print("\nEncoding labels...")

    # Strip whitespace from label column
    df[label_column] = df[label_column].str.strip()

    # Show unique labels
    print(f"Unique labels found: {df[label_column].unique()}")

    # Binary encoding - BENIGN = 0, everything else = 1
    df['label'] = df[label_column].apply(
        lambda x: 0 if x == 'BENIGN' else 1
    )

    print(f"Benign samples: {(df['label'] == 0).sum()}")
    print(f"Attack samples: {(df['label'] == 1).sum()}")

    return df


def normalize_features(df, label_column):
    """
    Normalize numerical features using MinMaxScaler
    """
    print("\nNormalizing features...")

    # Select only numerical columns
    # excluding label columns
    exclude_cols = [label_column, 'label',
                   'Source IP', 'Destination IP',
                   'source_ip', 'destination_ip',
                   'src_ip', 'dst_ip']

    # Get numerical columns only
    num_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # Remove label column from numerical columns
    num_cols = [c for c in num_cols
                if c not in exclude_cols]

    # Apply MinMax scaling
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    print(f"Normalized {len(num_cols)} numerical features")
    return df, scaler, num_cols


def preprocess_pipeline(filepath, label_column='Label'):
    """
    Complete preprocessing pipeline
    Run all steps in order
    """
    print("=" * 50)
    print("Starting Preprocessing Pipeline")
    print("=" * 50)

    # Step 1: Load
    df = load_dataset(filepath)

    # Step 2: Clean
    df = clean_dataset(df)

    # Step 3: Encode labels
    df = encode_labels(df, label_column)

    # Step 4: Normalize
    df, scaler, feature_cols = normalize_features(
        df, label_column
    )

    print("\n" + "=" * 50)
    print("Preprocessing Complete!")
    print(f"Final dataset shape: {df.shape}")
    print("=" * 50)

    return df, scaler, feature_cols


# ============================================================
# Test the data loader
# ============================================================
if __name__ == "__main__":

    # Change this path to your actual dataset location
    dataset_path = "../data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

    if os.path.exists(dataset_path):
        df, scaler, features = preprocess_pipeline(dataset_path)
        print("\nSample data:")
        print(df.head())
    else:
        print(f"Dataset not found at {dataset_path}")
        print("Please download CICIDS2017 and place it in data/ folder")