import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ─────────────────────────────────────────────

# PATH SETUP

# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_generation"
DB_PATH = DATA_DIR / "banking_data.db"

def engineer_features():
    conn = sqlite3.connect(DB_PATH)
    print("Engineering realistic features...")

    # ─────────────────────────────────────────────
    # TIME WINDOWS (CRITICAL)
    # ─────────────────────────────────────────────
    now = datetime.now()
    feature_start = (now - timedelta(days=90)).strftime('%Y-%m-%d')
    feature_mid   = (now - timedelta(days=60)).strftime('%Y-%m-%d')
    feature_end   = (now - timedelta(days=30)).strftime('%Y-%m-%d')

    # ─────────────────────────────────────────────
    # 1. SAFE TRANSACTION FEATURES
    # ─────────────────────────────────────────────
    query_txn = f"""
    SELECT customer_id,
           AVG(running_balance) as avg_balance,
           MIN(running_balance) as min_balance,
           COUNT(*) as txn_count
    FROM transactions
    WHERE transaction_date BETWEEN '{feature_start}' AND '{feature_end}'
    GROUP BY customer_id
    """

    df_txn = pd.read_sql_query(query_txn, conn)

    # ─────────────────────────────────────────────
    # 2. TREND FEATURES
    # ─────────────────────────────────────────────
    query_trend = f"""
    SELECT customer_id,
           AVG(CASE WHEN transaction_date >= '{feature_mid}' THEN running_balance END) as balance_recent,
           AVG(CASE WHEN transaction_date < '{feature_mid}' THEN running_balance END) as balance_past,
           COUNT(CASE WHEN transaction_date >= '{feature_mid}' THEN 1 END) as txn_recent,
           COUNT(CASE WHEN transaction_date < '{feature_mid}' THEN 1 END) as txn_past
    FROM transactions
    WHERE transaction_date BETWEEN '{feature_start}' AND '{feature_end}'
    GROUP BY customer_id
    """

    df_trend = pd.read_sql_query(query_trend, conn)

    # Safe calculations
    df_trend["balance_decline"] = (
        (df_trend["balance_recent"] - df_trend["balance_past"]) /
        (np.abs(df_trend["balance_past"]) + 1)
    )

    df_trend["txn_trend"] = (
        df_trend["txn_recent"] / (df_trend["txn_past"] + 1)
    )

    df_trend["balance_decline"] = df_trend["balance_decline"].clip(-2, 2)
    df_trend["txn_trend"] = df_trend["txn_trend"].clip(0, 3)

    df_trend = df_trend[["customer_id", "balance_decline", "txn_trend"]]

    # ─────────────────────────────────────────────
    # 3. APP BEHAVIOR (WEAK SIGNAL)
    # ─────────────────────────────────────────────
    query_app = f"""
    SELECT customer_id,
           COUNT(CASE WHEN timestamp >= '{feature_mid}' THEN 1 END) as logins_recent,
           COUNT(CASE WHEN timestamp < '{feature_mid}' THEN 1 END) as logins_past
    FROM app_logs
    WHERE timestamp BETWEEN '{feature_start}' AND '{feature_end}'
    GROUP BY customer_id
    """

    df_app = pd.read_sql_query(query_app, conn)

    df_app["login_trend"] = (
        df_app["logins_recent"] / (df_app["logins_past"] + 1)
    )

    df_app["login_trend"] = df_app["login_trend"].clip(0, 3)
    
    df_app = df_app[["customer_id", "login_trend"]]

    # ─────────────────────────────────────────────
    # MERGE ALL FEATURES
    # ─────────────────────────────────────────────
    df = df_txn.merge(df_trend, on="customer_id", how="left")
    df = df.merge(df_app, on="customer_id", how="left")

    df.fillna(0, inplace=True)

    # ─────────────────────────────────────────────
    # CUSTOMER DATA
    # ─────────────────────────────────────────────
    df_customers = pd.read_sql_query(
        "SELECT customer_id, monthly_income FROM customers",
        conn
    )

    df = df.merge(df_customers, on="customer_id", how="left")

    # ─────────────────────────────────────────────
    # REALISTIC LABEL
    # ─────────────────────────────────────────────
    def generate_label(row):
        score = (
            2.0 * row["balance_decline"] +
            1.5 * (row["txn_trend"] - 1) +
            1.0 * (row["login_trend"] - 1) -
            0.000002 * row["monthly_income"]
        )

        score += np.random.normal(0, 1.5)

        prob = 1 / (1 + np.exp(-score))
        return np.random.binomial(1, prob)

    df["is_delinquent"] = df.apply(generate_label, axis=1)

    # ─────────────────────────────────────────────
    # ADD NOISE (FIXED BUG HERE)
    # ─────────────────────────────────────────────
    for col in df.columns:
        if col not in ["customer_id", "is_delinquent"]:
            df[col] += np.random.normal(0, 0.2)

    # ─────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────
    output_path = DATA_DIR / "feature_store.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(df.head())

    conn.close()

    # ─────────────────────────────────────────────
    # SPLIT
    # ─────────────────────────────────────────────
    from split_data import split_data
    split_data()

if __name__ == "__main__":
    engineer_features()
