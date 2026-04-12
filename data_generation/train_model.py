import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def load_split(name):
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}_data.csv"))

def train_and_evaluate():
    print("Loading Data Splits...")
    train_df = load_split("train")
    test_df = load_split("test")
    
    features = [
        # Lifetime transaction aggregates
        'avg_txn_amt', 'min_balance', 'avg_balance', 
        'lending_app_txns', 'gambling_txns', 'atm_txns', 
        'bounced_count', 'failed_txn_count',
        # Temporal / windowed features
        'bounce_count_last_60d', 'lending_app_last_60d',
        'days_since_last_salary', 'salary_credit_count',
        'failed_txn_last_30d', 'late_utility_payments',
        'atm_trend_ratio', 'balance_decline_pct', 'gambling_acceleration',
        # App-log features
        'total_logins', 'balance_check_count',
        'login_freq_change', 'late_night_logins',
        # Support-log features
        'support_ticket_count', 'negative_sentiment_count', 'distress_topic_count',
    ]
    
    def get_X(df):
        return df[features]

    X_train = get_X(train_df)
    X_test = get_X(test_df)
    
    # 1. Preprocessing (Fit scaler on TRAIN only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Train Isolation Forest (Unsupervised - so we use X_train only)
    print("Training Isolation Forest on Training Set...")
    # We start with a rough contamination guess
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_train_scaled)
    
    # 3. Threshold from training score distribution only (unsupervised)
    print("\nSelecting threshold from train score distribution...")
    train_scores = iso_forest.score_samples(X_train_scaled)
    best_threshold = np.quantile(train_scores, 0.10)
    print(f"Threshold selected (10th percentile): {best_threshold:.4f}")
    
    # 4. Final Evaluation on TEST Set (Unbiased)
    print("\n--- Test Set Anomaly Summary ---")
    test_scores = iso_forest.score_samples(X_test_scaled)
    y_test_pred = [ -1 if s < best_threshold else 1 for s in test_scores]
    print(f"Anomalies detected in test set: {(np.array(y_test_pred) == -1).sum()} / {len(y_test_pred)}")
    
    # 5. Predict on Full Dataset for Dashboard
    print("\nGenerating Portfolio Predictions...")
    full_df = pd.read_csv(os.path.join(DATA_DIR, "feature_store.csv"))
    X_full = get_X(full_df)
    X_full_scaled = scaler.transform(X_full)
    
    full_df['anomaly_score'] = iso_forest.score_samples(X_full_scaled)
    # Use the tuned threshold
    full_df['is_anomaly'] = [ -1 if s < best_threshold else 1 for s in full_df['anomaly_score']]
    if 'is_delinquent' in full_df.columns:
        full_df['ground_truth'] = full_df['is_delinquent'].apply(lambda value: -1 if value == 1 else 1)
    
    # Save Model & Scaler
    joblib.dump(iso_forest, os.path.join(DATA_DIR, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(DATA_DIR, "scaler.pkl"))
    # Save threshold for inference
    with open(os.path.join(DATA_DIR, "threshold.txt"), "w") as f:
        f.write(str(best_threshold))
        
    full_df.to_csv(os.path.join(DATA_DIR, "model_results.csv"), index=False)
    print("Model, Scaler, Threshold, and Portfolio Results saved.")

if __name__ == "__main__":
    train_and_evaluate()
