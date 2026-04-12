import pandas as pd
from sklearn.model_selection import train_test_split
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUTPUT_DIR, "feature_store.csv")

def split_data():
    print("Loading Feature Store...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Feature store not found. Creating dummy data for checks if needed, but should exist.")
        return

    print(f"Total Samples: {len(df)}")
    
    # Stratified split by derived delinquency target
    # 70% Train, 15% Val, 15% Test
    
    # First: Split into Train (70%) and Temp (30%)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['is_delinquent'], random_state=42
    )
    
    # Second: Split Temp (30%) into Val (15%) and Test (15%) -> equal 50/50 split of the remainder
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['is_delinquent'], random_state=42
    )
    
    print("\n--- Split Stats ---")
    print(f"Train Set: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"Val Set:   {len(val_df)} ({len(val_df)/len(df):.1%})")
    print(f"Test Set:  {len(test_df)} ({len(test_df)/len(df):.1%})")
    
    print("\n--- Delinquency Distribution in Test ---")
    print(test_df['is_delinquent'].value_counts())

    # Save splits
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val_data.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_data.csv"), index=False)
    print("\nData splits saved successfully.")

if __name__ == "__main__":
    split_data()
