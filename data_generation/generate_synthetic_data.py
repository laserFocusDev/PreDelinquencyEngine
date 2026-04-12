import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os
from pathlib import Path

fake = Faker('en_IN')  # Use Indian locale for names/addresses
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# Configuration (Use pathlib for clean path handling)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_generation"
DATA_DIR.mkdir(parents=True, exist_ok=True)
NUM_CUSTOMERS = 5000
DAYS_HISTORY = 180  # 6 months
START_DATE = datetime.now() - timedelta(days=DAYS_HISTORY)

def generate_customers(n):
    customers = []
    for _ in range(n):
        financial_literacy = np.random.beta(2, 5)
        impulsivity = np.random.beta(2, 5)
        income_stability = np.random.beta(5, 2)

        # Base theoretical risk derived from continuous unobserved traits
        base_risk = (
            0.4 * impulsivity +
            0.4 * (1 - income_stability) +
            0.2 * (1 - financial_literacy)
        )
        
        # Introduce structural bias
        macro_shock = np.random.normal(0, 0.1)
        total_stress = base_risk + macro_shock
        prob_default = 1 / (1 + np.exp(-10 * (total_stress - 0.6)))
        is_delinquent = np.random.binomial(n=1, p=prob_default)

        # Behaviors
        gambling_prob = np.clip(impulsivity * 0.3 + np.random.normal(0, 0.05), 0, 1)
        salary_delay_prob = np.clip((1 - income_stability) * 0.4 + np.random.normal(0, 0.05), 0, 1)
        lending_app_prob = np.clip((1 - financial_literacy) * impulsivity * 0.5 + np.random.normal(0, 0.02), 0, 1)
        bounce_prob = np.clip((1 - income_stability) * 0.4 + impulsivity * 0.2 + np.random.normal(0, 0.05), 0, 1)
        spending_volatility = np.clip(impulsivity * 0.8 + np.random.normal(0, 0.1), 0, 1)
        atm_hoarding_prob = np.clip(impulsivity * 0.4 + np.random.normal(0, 0.1), 0, 1)
        discretionary_reduction = np.clip((1 - income_stability) * 0.6 + np.random.normal(0, 0.1), 0, 1)

        app_login_freq = np.random.beta(2, 2)
        balance_check_freq = np.clip(impulsivity * 0.5 + (1 - income_stability) * 0.3 + np.random.normal(0, 0.1), 0, 1)
        support_ticket_prob = np.clip((1 - financial_literacy) * 0.4 + impulsivity * 0.2 + np.random.normal(0, 0.05), 0, 1)
        
        income = np.random.randint(30000, 150000)
        
        customers.append({
            'customer_id': fake.uuid4(),
            'name': fake.name(),
            'age': np.random.randint(22, 55),
            'monthly_income': income,
            'is_delinquent': int(is_delinquent),
            'gambling_prob': gambling_prob,
            'salary_delay_prob': salary_delay_prob,
            'lending_app_prob': lending_app_prob,
            'bounce_prob': bounce_prob,
            'spending_volatility': spending_volatility,
            'atm_hoarding_prob': atm_hoarding_prob,
            'discretionary_reduction': discretionary_reduction,
            'app_login_freq': app_login_freq,
            'balance_check_freq': balance_check_freq,
            'support_ticket_prob': support_ticket_prob,
            'financial_literacy': financial_literacy,
            'impulsivity': impulsivity,
            'income_stability': income_stability,
            'bill_due_date': np.random.randint(1, 28)
        })
    return pd.DataFrame(customers)

def generate_bill_schedules(customers):
    schedules = []
    for _, cust in customers.iterrows():
        schedules.append({
            'customer_id': cust['customer_id'],
            'bill_type': 'ELECTRICITY',
            'due_date_day': cust['bill_due_date'],
            'expected_amount': np.random.randint(500, 3000)
        })
    return pd.DataFrame(schedules)

def generate_transactions(customers):
    transactions = []
    
    for _, cust in customers.iterrows():
        current_date = START_DATE
        balance = np.random.randint(10000, 50000)  # Initial balance
        overdraft_limit = -int(cust['monthly_income'] * 0.5)
        salary_date = 1  # 1st of every month usually
        
        while current_date <= datetime.now():
            # 1. Salary Credit (Monthly)
            if current_date.day == salary_date:
                # Introduce delay based on risk
                actual_salary_date = current_date
                if np.random.random() < cust['salary_delay_prob']:
                    delay_days = np.random.randint(3, 10) # 3-10 days delay
                    actual_salary_date += timedelta(days=delay_days)
                
                if actual_salary_date <= datetime.now():
                   balance += cust['monthly_income']
                   transactions.append({
                       'customer_id': cust['customer_id'],
                       'transaction_date': actual_salary_date,
                       'amount': cust['monthly_income'],
                       'type': 'CREDIT',
                       'category': 'SALARY',
                       'description': 'SALARY CREDIT',
                       'status': 'SUCCESS',
                       'running_balance': balance
                   })

            # 2. Daily Spending (UPI / Sent to Lending Apps / ATM)
            if np.random.random() < 0.7: # 70% chance of transaction any day
                num_txns = np.random.randint(1, 4)
                for _ in range(num_txns):
                    # Default probabilities
                    probs = [0.34, 0.33, 0.33]  # FOOD, SHOPPING, UPI_P2P
                    categories = ['FOOD', 'SHOPPING', 'UPI_P2P']
                    
                    # Risk Logic 1: Reduced Discretionary Spend (Stress)
                    if cust['discretionary_reduction'] > 0:
                        # Reduce prob of FOOD/SHOPPING, increase UPI_P2P (borrowing) or nothing
                        factor = cust['discretionary_reduction']
                        probs = [0.34 * (1-factor), 0.33 * (1-factor), 0.33 + (0.67 * factor)]
                        # Normalize
                        total = sum(probs)
                        probs = [p/total for p in probs]

                    amount = np.random.randint(100, 2000)
                    category = np.random.choice(categories, p=probs)

                    # Risk Signal: UPI to lending apps
                    if np.random.random() < cust['lending_app_prob']:
                         category = 'UPI_LENDING_APP'
                         amount = np.random.randint(500, 5000)
                    
                    # Risk Signal: ATM Cash Hoarding
                    if np.random.random() < cust['atm_hoarding_prob']:
                         category = 'ATM_WITHDRAWAL'
                         amount = np.random.randint(2000, 10000) # Large withdrawals

                    projected_balance = balance - amount
                    if projected_balance < overdraft_limit:
                        transactions.append({
                            'customer_id': cust['customer_id'],
                            'transaction_date': current_date,
                            'amount': 0,
                            'type': 'DEBIT',
                            'category': category,
                            'description': f'UPI/{fake.first_name()}/{category}',
                            'status': 'FAILED',
                            'running_balance': balance
                        })
                        continue

                    balance = projected_balance
                    transactions.append({
                        'customer_id': cust['customer_id'],
                        'transaction_date': current_date,
                        'amount': -amount,
                        'type': 'DEBIT',
                        'category': category,
                        'description': f'UPI/{fake.first_name()}/{category}',
                        'status': 'SUCCESS',
                        'running_balance': balance
                    })

            # 2b. Explicit Utility Payments (Correct Logic)
            due_day = cust['bill_due_date']
            # Basic logic: Check if we are in the payment window
            # We want to simulate ONE utility payment per month
            # Let's say we pay exactly on the (due_day + delay)
            delay = int(np.clip(np.random.normal((1 - cust['income_stability']) * 12, 5), -2, 15))
            
            payment_date = datetime(current_date.year, current_date.month, 1) + timedelta(days=due_day + delay - 1)
            # Handle month rollover issues roughly (clipping)
            
            if current_date.date() == payment_date.date():
                amount = np.random.randint(500, 3000)
                projected_balance = balance - amount
                if projected_balance < overdraft_limit:
                    transactions.append({
                        'customer_id': cust['customer_id'],
                        'transaction_date': current_date,
                        'amount': 0,
                        'type': 'DEBIT',
                        'category': 'UTILITY',
                        'description': 'ELECTRICITY BILL',
                        'status': 'FAILED',
                        'running_balance': balance
                    })
                else:
                    balance = projected_balance
                    transactions.append({
                        'customer_id': cust['customer_id'],
                        'transaction_date': current_date,
                        'amount': -amount,
                        'type': 'DEBIT',
                        'category': 'UTILITY',
                        'description': 'ELECTRICITY BILL',
                        'status': 'SUCCESS',
                        'running_balance': balance
                    })

            # 3. High Risk Signal: Gambling/Lottery
            if np.random.random() < cust['gambling_prob']:
                amount = np.random.randint(1000, 10000)
                projected_balance = balance - amount
                if projected_balance < overdraft_limit:
                    transactions.append({
                        'customer_id': cust['customer_id'],
                        'transaction_date': current_date,
                        'amount': 0,
                        'type': 'DEBIT',
                        'category': 'GAMBLING',
                        'description': 'ONLINE BETTING',
                        'status': 'FAILED',
                        'running_balance': balance
                    })
                else:
                    balance = projected_balance
                    transactions.append({
                        'customer_id': cust['customer_id'],
                        'transaction_date': current_date,
                        'amount': -amount,
                        'type': 'DEBIT',
                        'category': 'GAMBLING',
                        'description': 'ONLINE BETTING',
                        'status': 'SUCCESS',
                        'running_balance': balance
                    })

            # 4. Auto-Debit (Loan EMI) - usually 5th of month
            if current_date.day == 5:
                emi_amount = int(cust['monthly_income'] * 0.3) # 30% of income
                status = 'SUCCESS'
                
                # Check for bounce
                if np.random.random() < cust['bounce_prob']:
                    status = 'BOUNCED'
                    # Balance doesn't decrease on bounce
                else:
                    projected_balance = balance - emi_amount
                    if projected_balance < overdraft_limit:
                        status = 'BOUNCED'
                    else:
                        balance = projected_balance
                
                transactions.append({
                    'customer_id': cust['customer_id'],
                    'transaction_date': current_date,
                    'amount': -emi_amount,
                    'type': 'DEBIT',
                    'category': 'LOAN_EMI',
                    'description': 'AUTO-DEBIT EMI',
                    'status': status,
                    'running_balance': balance
                })

            current_date += timedelta(days=1)
            
    df = pd.DataFrame(transactions)
    return df.sort_values(by=['customer_id', 'transaction_date'])


def generate_app_logs(customers):
    logs = []
    for _, cust in customers.iterrows():
        device_pool = [fake.uuid4() for _ in range(np.random.randint(1, 4))]
        current_date = START_DATE
        while current_date <= datetime.now():
            # Check if user logs in today
            if np.random.random() < cust['app_login_freq']:
                # Number of sessions
                num_sessions = np.random.randint(1, 4)
                if np.random.random() < cust['impulsivity']: num_sessions += np.random.randint(0, 5) # Increased frequency
                
                for _ in range(num_sessions):
                    # Time of day
                    hour = np.random.randint(8, 22)
                    if cust['impulsivity'] > 0.7 and np.random.random() < 0.3:
                         hour = np.random.randint(0, 5) # Late night logins
                    
                    timestamp = current_date + timedelta(hours=hour, minutes=np.random.randint(0, 59))
                    
                    # Events in session
                    events = ['LOGIN', 'VIEW_DASHBOARD']
                    if np.random.random() < cust['balance_check_freq']:
                         events.append('CHECK_BALANCE')
                         # Disorganized finance check checks balance multiple times
                         if cust['impulsivity'] > 0.6:
                             events.extend(['CHECK_BALANCE'] * np.random.randint(1, 4))
                    
                    for event in events:
                        logs.append({
                            'customer_id': cust['customer_id'],
                            'timestamp': timestamp,
                            'event_type': event,
                            'device_id': random.choice(device_pool)
                        })
                        timestamp += timedelta(seconds=np.random.randint(5, 60))
            
            current_date += timedelta(days=1)
    return pd.DataFrame(logs)

def generate_support_logs(customers):
    tickets = []
    topics_low = ['Info Request', 'Address Change', 'Card Replacement']
    topics_high = ['Fees Dispute', 'Limit Increase', 'Payment Extension', 'Overdraft Query']
    
    for _, cust in customers.iterrows():
        current_month = datetime(START_DATE.year, START_DATE.month, 1)
        end_month = datetime(datetime.now().year, datetime.now().month, 1)

        while current_month <= end_month:
            if np.random.random() < cust['support_ticket_prob']:
                date = current_month + timedelta(days=np.random.randint(0, 28), hours=np.random.randint(8, 21))
                if date > datetime.now():
                    current_month = (current_month + timedelta(days=32)).replace(day=1)
                    continue
            
                if cust['financial_literacy'] < 0.3 or cust['impulsivity'] > 0.7:
                    topic = np.random.choice(topics_high)
                    sentiment = np.random.choice(['Negative', 'Anxious', 'Aggressive'], p=[0.5, 0.4, 0.1])
                    description = f"Issue with {topic.lower()}. Please help immediately."
                else:
                    topic = np.random.choice(topics_low)
                    sentiment = np.random.choice(['Neutral', 'Positive'], p=[0.8, 0.2])
                    description = f"Request regarding {topic.lower()}."

                tickets.append({
                    'customer_id': cust['customer_id'],
                    'date': date,
                    'topic': topic,
                    'sentiment': sentiment,
                    'description': description,
                    'channel': np.random.choice(['Chat', 'Call', 'Email'])
                })

            current_month = (current_month + timedelta(days=32)).replace(day=1)
            
    return pd.DataFrame(tickets)

if __name__ == "__main__":
    print("Generating Customers...")
    df_customers = generate_customers(NUM_CUSTOMERS)
    df_customers[['customer_id', 'name', 'age', 'monthly_income', 'is_delinquent', 'bill_due_date']].to_csv(DATA_DIR / "customers.csv", index=False)
    print(f"Saved {len(df_customers)} customers.")

    print("Generating Bill Schedules...")
    df_bills = generate_bill_schedules(df_customers)
    df_bills.to_csv(DATA_DIR / "bill_schedules.csv", index=False)
    print(f"Saved {len(df_bills)} bill schedules.")

    print("Generating Transactions...")
    df_transactions = generate_transactions(df_customers)
    df_transactions.to_csv(DATA_DIR / "transactions.csv", index=False)
    print(f"Saved {len(df_transactions)} transactions.")
    
    print("Generating App Interaction Logs...")
    df_app = generate_app_logs(df_customers)
    df_app.to_csv(DATA_DIR / "app_logs.csv", index=False)
    print(f"Saved {len(df_app)} app logs.")
    
    print("Generating Support Tickets...")
    df_support = generate_support_logs(df_customers)
    df_support.to_csv(DATA_DIR / "support_logs.csv", index=False)
    print(f"Saved {len(df_support)} support tickets.")

    # Preview
    print("\n--- Preview Data ---")
    print(f"Transactions: {len(df_transactions)}")
    print(f"App Logs: {len(df_app)}")
    print(f"Support Tickets: {len(df_support)}")

