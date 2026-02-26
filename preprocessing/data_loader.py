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

def load_dataset(train_path, test_path=None):
    """
    Load the KDD Cup Network Intrusion dataset
    """
    print(f"Loading training dataset from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Training dataset loaded successfully!")
    print(f"Train Shape: {train_df.shape}")

    if test_path:
        print(f"\nLoading test dataset from: {test_path}")
        test_df = pd.read_csv(test_path)
        print(f"Test dataset loaded successfully!")
        print(f"Test Shape: {test_df.shape}")
        return train_df, test_df

    return train_df, None


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

    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f"Removed infinite values")

    print(f"Clean dataset shape: {df.shape}")
    return df


def encode_labels(df):
    """
    Encode labels into binary format
    normal = 0
    anomaly = 1
    """
    print("\nEncoding labels...")

    # Show unique labels
    print(f"Unique labels: {df['class'].unique()}")

    # Binary encoding
    df['label'] = df['class'].apply(
        lambda x: 0 if x == 'normal' else 1
    )

    print(f"Normal samples: {(df['label'] == 0).sum()}")
    print(f"Anomaly samples: {(df['label'] == 1).sum()}")

    return df


def encode_categorical(df):
    """
    Encode categorical columns into numerical values
    This dataset has 3 categorical columns:
    protocol_type, service, flag
    """
    print("\nEncoding categorical features...")

    categorical_cols = ['protocol_type', 'service', 'flag']
    le = LabelEncoder()

    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"Encoded column: {col}")

    return df


def normalize_features(df):
    """
    Normalize numerical features using MinMaxScaler
    """
    print("\nNormalizing features...")

    # Columns to exclude from normalization
    exclude_cols = ['class', 'label']

    # Get numerical columns
    num_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # Remove excluded columns
    num_cols = [c for c in num_cols
                if c not in exclude_cols]

    # Apply scaling
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    print(f"Normalized {len(num_cols)} features")
    return df, scaler, num_cols


def preprocess_pipeline(train_path, test_path=None):
    """
    Complete preprocessing pipeline
    """
    print("=" * 50)
    print("Starting Preprocessing Pipeline")
    print("=" * 50)

    # Step 1: Load
    train_df, test_df = load_dataset(train_path, test_path)

    # Step 2: Clean
    train_df = clean_dataset(train_df)

    # Step 3: Encode categorical
    train_df = encode_categorical(train_df)

    # Step 4: Encode labels
    train_df = encode_labels(train_df)

    # Step 5: Normalize
    train_df, scaler, feature_cols = normalize_features(train_df)

    print("\n" + "=" * 50)
    print("Preprocessing Complete!")
    print(f"Final training dataset shape: {train_df.shape}")
    print(f"Features available: {len(feature_cols)}")
    print("=" * 50)

    return train_df, test_df, scaler, feature_cols


# ============================================================
# Test the data loader
# ============================================================
if __name__ == "__main__":

    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "Train_data.csv")
    test_path = os.path.join(base_dir, "data", "Test_data.csv")

    if os.path.exists(train_path):
        train_df, test_df, scaler, features = preprocess_pipeline(
            train_path, test_path
        )
        print("\nSample data:")
        print(train_df.head())
        print("\nLabel distribution:")
        print(train_df['label'].value_counts())
    else:
        print(f"Dataset not found at {train_path}")
        print("Please place Train_data.csv in data/ folder")