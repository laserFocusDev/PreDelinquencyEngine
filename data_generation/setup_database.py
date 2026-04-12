import pandas as pd
import sqlite3
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DATA_DIR, "banking_data.db")

def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Load CSVs
    print("Loading CSVs...")
    df_cust = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    df_txns = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    df_app = pd.read_csv(os.path.join(DATA_DIR, "app_logs.csv"))
    df_support = pd.read_csv(os.path.join(DATA_DIR, "support_logs.csv"))

    # Convert timestamps
    df_txns['transaction_date'] = pd.to_datetime(df_txns['transaction_date'])
    df_app['timestamp'] = pd.to_datetime(df_app['timestamp'])

    # Write to SQL (Replace if exists)
    print("Writing to SQLite...")
    df_cust.to_sql("customers", conn, if_exists="replace", index=False)
    # Create index for speed
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cust_id ON customers(customer_id)")
    
    df_txns.to_sql("transactions", conn, if_exists="replace", index=False)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_cust_id ON transactions(customer_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(transaction_date)")
    
    df_app.to_sql("app_logs", conn, if_exists="replace", index=False)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_app_cust_id ON app_logs(customer_id)")

    df_support.to_sql("support_logs", conn, if_exists="replace", index=False)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_support_cust_id ON support_logs(customer_id)")

    print(f"Database setup complete at {DB_PATH}")
    
    # Verification Count
    cursor.execute("SELECT count(*) FROM transactions")
    print(f"Total Transactions in DB: {cursor.fetchone()[0]}")
    
    conn.close()

if __name__ == "__main__":
    setup_database()
