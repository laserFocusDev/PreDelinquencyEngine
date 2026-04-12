import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import shap
import os
import matplotlib.pyplot as plt

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def train_xgboost():
    print("Loading Data Splits...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_data.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
    
    # Features — lifetime aggregates + temporal trends + support signals
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
    
    def prepare_data(df):
        X = df[features]
        y = df['is_delinquent'].astype(int)
        return X, y

    X_train, y_train = prepare_data(train_df)
    X_val, y_val = prepare_data(val_df)
    X_test, y_test = prepare_data(test_df)
    
    # 1. Train XGBoost Classifier
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=10
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 2. Evaluation
    print("\n--- Evaluation on Test Set ---")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Delinquent', 'Delinquent']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 3. Explainability (SHAP)
    print("\nGenerating SHAP Values...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    # Save SHAP plot for the dashboard/report
    # Binary classification summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(DATA_DIR, "shap_summary_delinquency.png"), bbox_inches='tight')
    print("SHAP Summary Plot saved.")
    
    # Default probability for dashboard: class-1 delinquency probability
    
    # 4. Save Artifacts
    joblib.dump(model, os.path.join(DATA_DIR, "xgboost_model.pkl"))
    # Save the explainers can be huge, usually we just save the model and re-init explainer
    # But for dashboard speed, we might save a small version or just the values for a sample
    joblib.dump(explainer, os.path.join(DATA_DIR, "shap_explainer.pkl"))
    
    print("XGBoost Model and Explainer saved.")
    
    # 5. Generate Scores for Full Portfolio (for Dashboard)
    print("\nScoring Full Portfolio...")
    full_df = pd.read_csv(os.path.join(DATA_DIR, "feature_store.csv"))
    X_full = full_df[features]
    
    # Get probability of delinquency class (1)
    probs = model.predict_proba(X_full)
    full_df['risk_score_prob'] = probs[:, 1]
    full_df['risk_score_scaled'] = (full_df['risk_score_prob'] * 100).astype(int)
    
    # Save updated results
    full_df.to_csv(os.path.join(DATA_DIR, "model_results_xgb.csv"), index=False)
    print("Full portfolio scores saved to model_results_xgb.csv")

if __name__ == "__main__":
    train_xgboost()
