import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_data(df):
    # Columns that often appear in Zameen datasets but are not numeric
    drop_cols = [
        "url", "link", "Location", "Address", "Title", "Description",
        "Area Type", "Property ID"
    ]

    # Remove existing non-numeric columns safely
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Keep only numeric columns
    df = df.select_dtypes(include=["number"])

    # Remove rows with missing target
    df = df.dropna(subset=["Price"])

    # Fill missing numeric values
    df = df.fillna(df.median())

    return df

def process_and_save(input_path, out_dir):
    df = pd.read_csv(input_path)

    df = clean_data(df)

    # Split
    X = df.drop(columns=["Price"])
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs(out_dir, exist_ok=True)

    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(out_dir, "y_test.csv"), index=False)

    print("✓ Processed data saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    process_and_save(args.input, args.out_dir)
