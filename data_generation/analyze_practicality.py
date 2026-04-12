import pandas as pd
import numpy as np
from datetime import datetime
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Data
print("Loading data...")
try:
    df_cust = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    df_txns = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    df_txns['transaction_date'] = pd.to_datetime(df_txns['transaction_date'])
except FileNotFoundError:
    print("Error: content files not found. Run generation script first.")
    exit()

print(f"Loaded {len(df_cust)} customers and {len(df_txns)} transactions.")

# --- Signal 1: Salary Delays ---
print("\n--- 1. Salary Delays Analysis ---")
salary_txns = df_txns[df_txns['category'] == 'SALARY'].copy()
salary_txns['day'] = salary_txns['transaction_date'].dt.day
# Calculate variance in salary day per customer
salary_variance = salary_txns.groupby('customer_id')['day'].std().reset_index()
salary_variance.columns = ['customer_id', 'salary_day_std']
# Merge with risk profile
merged_salary = salary_variance.merge(df_cust, on='customer_id')
print("Average Standard Deviation of Salary Day by Risk Level:")
print(merged_salary.groupby('risk_level')['salary_day_std'].mean())

# --- Signal 2: Lending Apps Usage ---
print("\n--- 2. Lending Apps Usage ---")
lending_txns = df_txns[df_txns['category'] == 'UPI_LENDING_APP']
print(f"Total Lending App Transactions: {len(lending_txns)}")
if len(lending_txns) > 0:
    lending_users = lending_txns['customer_id'].unique()
    lending_user_risk = df_cust[df_cust['customer_id'].isin(lending_users)]['risk_level'].value_counts()
    print("Risk Profile of users using Lending Apps:")
    print(lending_user_risk)

# --- Signal 3: Auto-Debit Failures ---
print("\n--- 3. Auto-Debit Failures ---")
bounced_txns = df_txns[df_txns['status'] == 'BOUNCED']
print(f"Total Bounced Auto-Debits: {len(bounced_txns)}")
if len(bounced_txns) > 0:
    bounced_users = bounced_txns['customer_id'].unique()
    bounced_user_risk = df_cust[df_cust['customer_id'].isin(bounced_users)]['risk_level'].value_counts()
    print("Risk Profile of users with Bounced Auto-Debits:")
    print(bounced_user_risk)

# --- Signal 4: Savings Drawdown (Balance Trend) ---
print("\n--- 4. Savings Drawdown Analysis ---")
# Calculate slope of balance for each customer (simple linear regression on time)
from scipy.stats import linregress

def calculate_slope(group):
    # timestamps to days from start
    if len(group) < 2: return 0
    start_date = group['transaction_date'].min()
    days = (group['transaction_date'] - start_date).dt.days
    slope, _, _, _, _ = linregress(days, group['running_balance'])
    return slope

# Sample 20 customers to save time, or do all if fast enough
sample_cust_ids = df_cust['customer_id'].sample(50, random_state=42)
sample_txns = df_txns[df_txns['customer_id'].isin(sample_cust_ids)]

slopes = sample_txns.groupby('customer_id').apply(calculate_slope).reset_index()
slopes.columns = ['customer_id', 'balance_slope']

merged_slopes = slopes.merge(df_cust, on='customer_id')
print("Average Balance Slope (Rs/day) by Risk Level (Negative = Declining):")
print(merged_slopes.groupby('risk_level')['balance_slope'].mean())

# --- Signal 5: Gambling & ATM Hoarding ---
print("\n--- 5. Behavioral Risks (Gambling & ATM) ---")
gambling_txns = df_txns[df_txns['category'] == 'GAMBLING']
print(f"Total Gambling Transactions: {len(gambling_txns)}")

atm_txns = df_txns[df_txns['category'] == 'ATM_WITHDRAWAL']
print(f"Total ATM Withdrawals: {len(atm_txns)}")
if len(atm_txns) > 0:
    atm_users = atm_txns['customer_id'].unique()
    print("Risk Profile of users making large ATM withdrawals:")
    # Calculate user risk counts
    user_risks = df_cust[df_cust['customer_id'].isin(atm_users)]['risk_level'].value_counts()
    print(user_risks)

# --- Signal 6: Utility Lateness ---
print("\n--- 6. Utility Bill Lateness ---")
try:
    df_bills = pd.read_csv(os.path.join(DATA_DIR, "bill_schedules.csv"))
    utility_txns = df_txns[df_txns['category'] == 'UTILITY'].copy()
    utility_txns['day'] = utility_txns['transaction_date'].dt.day
    
    # Merge with bill schedule to get due date
    merged_utility = utility_txns.merge(df_bills[['customer_id', 'due_date_day']], on='customer_id')
    
    # Calculate Lateness (Payment Day - Due Date)
    # Note: highly simplified logic assuming same month payment
    merged_utility['lateness'] = merged_utility['day'] - merged_utility['due_date_day']
    
    # Filter for only late payments (positive lateness) to see severity
    # Actually let's look at average lateness
    avg_lateness = merged_utility.groupby('customer_id')['lateness'].mean().reset_index()
    
    merged_lateness = avg_lateness.merge(df_cust[['customer_id', 'risk_level']], on='customer_id')
    print("Average Utility Payment Lateness (Days) by Risk Level:")
    print(merged_lateness.groupby('risk_level')['lateness'].mean())
except FileNotFoundError:
    print("Bill schedules not found.")

# --- Signal 7: Discretionary Spend Reduction ---
print("\n--- 7. Discretionary Spend Ratio ---")
# Calculate ratio of (FOOD + SHOPPING) count to Total Txns per customer
def discretionary_ratio(group):
    total = len(group)
    if total == 0: return 0
    discretionary = len(group[group['category'].isin(['FOOD', 'SHOPPING'])])
    return discretionary / total

disc_ratios = df_txns.groupby('customer_id').apply(discretionary_ratio).reset_index()
disc_ratios.columns = ['customer_id', 'disc_ratio']
merged_disc = disc_ratios.merge(df_cust[['customer_id', 'risk_level']], on='customer_id')

print("Average Discretionary Spend Ratio by Risk Level:")
print(merged_disc.groupby('risk_level')['disc_ratio'].mean())


print("\n--- Conclusion ---")
print("Does the dataset reflect the problem statement?")
