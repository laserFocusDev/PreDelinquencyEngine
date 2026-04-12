import pandas as pd
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading data...")
df_cust = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
df_txns = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))

print(f"Loaded {len(df_cust)} customers and {len(df_txns)} transactions.")

# Check for Risk Signals
print("\n--- Risk Signal Verification ---")
bounced_txns = df_txns[df_txns['status'] == 'BOUNCED']
print(f"Total BOUNCED transactions: {len(bounced_txns)}")
if len(bounced_txns) > 0:
    print("Sample BOUNCED transaction:")
    print(bounced_txns.head(1)[['customer_id', 'transaction_date', 'amount', 'category', 'status']])

gambling_txns = df_txns[df_txns['category'] == 'GAMBLING']
print(f"Total GAMBLING transactions: {len(gambling_txns)}")
if len(gambling_txns) > 0:
    print("Sample GAMBLING transaction:")
    print(gambling_txns.head(1)[['customer_id', 'amount', 'running_balance']])

# Check Balance Logic
print("\n--- Balance Logic Verification ---")
# Pick a random customer 
sample_cust_id = df_cust.iloc[0]['customer_id']
sample_txns = df_txns[df_txns['customer_id'] == sample_cust_id].sort_values('transaction_date')
print(f"Checking balance flow for customer {sample_cust_id}...")
print(sample_txns[['transaction_date', 'category', 'amount', 'running_balance']].head(10))

# Verify Negative Balances (Overdrafts)
negative_balance_count = len(df_txns[df_txns['running_balance'] < 0])
print(f"\nTransactions resulting in negative balance: {negative_balance_count}")
