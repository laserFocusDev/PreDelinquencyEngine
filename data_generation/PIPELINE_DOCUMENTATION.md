# Pre-Delinquency Intervention Engine — Pipeline Documentation

> Comprehensive technical reference for the synthetic data generation, feature engineering, model training, and inference pipeline.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Pipeline Execution Order](#2-pipeline-execution-order)
3. [Configuration & Constants](#3-configuration--constants)
4. [Data Generation (`generate_synthetic_data.py`)](#4-data-generation)
5. [Database Setup (`setup_database.py`)](#5-database-setup)
6. [Feature Engineering (`engineer_features.py`)](#6-feature-engineering)
7. [Data Splitting (`split_data.py`)](#7-data-splitting)
8. [XGBoost Training (`train_xgboost.py`)](#8-xgboost-training)
9. [Isolation Forest Training (`train_model.py`)](#9-isolation-forest-training)
10. [Dashboard (`dashboard.py`)](#10-dashboard)
11. [Utility Scripts](#11-utility-scripts)
12. [CSV Schemas (All Files)](#12-csv-schemas)
13. [Feature Catalog (25 Features)](#13-feature-catalog)
14. [Ground-Truth Labeling Logic](#14-ground-truth-labeling-logic)
15. [Model Output Schemas](#15-model-output-schemas)
16. [Design Decisions & Rationale](#16-design-decisions--rationale)
17. [Known Limitations & Future Work](#17-known-limitations--future-work)

---

## 1. System Overview

The engine generates synthetic Indian banking customer data, engineers behavioral features from transaction/app/support histories, derives a ground-truth delinquency label, and trains two complementary models:

| Model | Type | Purpose |
|-------|------|---------|
| **XGBoost** | Supervised binary classifier | Primary risk scorer — outputs calibrated delinquency probability (0–100) |
| **Isolation Forest** | Unsupervised anomaly detector | Safety net — catches "unknown unknowns" the supervised model may miss |

A Streamlit dashboard consumes both model outputs for portfolio monitoring, SHAP-based explainability, customer-level drill-down, and intervention queue management.

---

## 2. Pipeline Execution Order

```
Step 1: generate_synthetic_data.py    → customers.csv, transactions.csv, app_logs.csv, 
                                        support_logs.csv, bill_schedules.csv
Step 2: setup_database.py             → banking_data.db (SQLite)
Step 3: engineer_features.py          → feature_store.csv  (calls split_data.py internally)
                                      → train_data.csv, val_data.csv, test_data.csv
Step 4: train_xgboost.py              → xgboost_model.pkl, shap_explainer.pkl,
                                        shap_summary_delinquency.png, model_results_xgb.csv
Step 5: train_model.py                → isolation_forest.pkl, scaler.pkl,
                                        threshold.txt, model_results.csv
Step 6: dashboard.py                  → Streamlit app (reads model outputs + DB)
```

Each step must complete before the next begins. All scripts use relative paths via `os.path.dirname(os.path.abspath(__file__))` — no hardcoded absolute paths.

---

## 3. Configuration & Constants

Defined at the top of `generate_synthetic_data.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `NUM_CUSTOMERS` | 5,000 | Total synthetic customers |
| `DAYS_HISTORY` | 180 | 6-month simulation window |
| `START_DATE` | `now() - 180 days` | Simulation start date |
| Random seeds | `42` (Faker, NumPy, random) | Full reproducibility |
| Locale | `en_IN` | Indian names, addresses |

### Risk Profile Distribution

Customers are randomly assigned to a generation tier (60/30/10 split). This tier controls behavioral probabilities during data generation but is **not used as the model's target label** — delinquency is derived post-generation from actual outcomes.

| Profile Parameter | LOW | MEDIUM | HIGH |
|------------------|-----|--------|------|
| `salary_delay_prob` | 0.01 | 0.10 | 0.40 |
| `gambling_prob` | 0.005 | 0.03 | 0.20 |
| `lending_app_prob` | 0.01 | 0.05 | 0.30 |
| `bounce_prob` | 0.01 | 0.10 | 0.40 |
| `spending_volatility` | 0.1 | 0.3 | 0.6 |
| `atm_hoarding_prob` | 0.05 | 0.20 | 0.60 |
| `discretionary_reduction` | 0.0 | 0.3 | 0.7 |
| `app_login_freq` | 0.2 | 0.5 | 0.9 |
| `balance_check_freq` | 0.1 | 0.4 | 0.9 |
| `support_ticket_prob` | 0.05 | 0.20 | 0.50 |

**Cross-contamination:** LOW/MEDIUM customers are given non-zero gambling and lending app probabilities (0.5–3%) to ensure feature overlap between classes and prevent trivial classification.

---

## 4. Data Generation

**Script:** `generate_synthetic_data.py` (367 lines)

Generates five raw CSV files by simulating 180 days of banking activity per customer.

### 4.1 Customer Generation (`generate_customers()`)

For each of the 5,000 customers:
- UUID v4 `customer_id` (primary key across all tables)
- Fake Indian name, random age (22–55), random monthly income (₹30,000–₹1,50,000)
- Internal `risk_level` (LOW/MEDIUM/HIGH) and `profile` dict — used only for generation
- The `profile` column is **dropped before saving** to prevent data leakage
- Random `bill_due_date` (1–27) for utility payments

### 4.2 Transaction Generation (`generate_transactions()`)

Day-by-day simulation for each customer from `START_DATE` to `now()`. Each day can generate multiple transaction types:

| Transaction Type | Timing | Amount Range | Logic |
|------------------|--------|-------------|-------|
| **SALARY** | 1st of month | `monthly_income` | Delayed by 3–10 days with `salary_delay_prob` |
| **Daily Spending** (FOOD/SHOPPING/UPI_P2P) | 70% chance/day, 1–3 txns | ₹100–₹2,000 | Discretionary reduction shifts probability toward UPI_P2P |
| **UPI_LENDING_APP** | Sampled via `lending_app_prob` | ₹500–₹5,000 | Overlaps all tiers |
| **ATM_WITHDRAWAL** | Sampled via `atm_hoarding_prob` | ₹2,000–₹10,000 | Cash hoarding signal |
| **GAMBLING** | Sampled via `gambling_prob` | ₹1,000–₹10,000 | Overlaps all tiers |
| **UTILITY** | Once/month on `bill_due_date + delay` | ₹500–₹3,000 | HIGH: +5–15 day delay, MEDIUM: -2 to +5 |
| **LOAN_EMI** | 5th of month | 30% of income | Bounces with `bounce_prob` or if overdraft exceeded |

**Overdraft Guard:** Every debit checks `projected_balance < -monthly_income * 0.5`. If exceeded, the transaction is recorded with `amount=0, status='FAILED'` and the balance remains unchanged. This prevents unrealistic infinite overdrafts while still tracking blocked-transaction counts as a stress signal.

**EMI Bounce Logic:** Auto-debit EMIs bounce either randomly (via `bounce_prob`) or deterministically when the overdraft limit would be breached. Bounced EMIs do not deduct from the balance.

### 4.3 App Log Generation (`generate_app_logs()`)

Each customer is assigned a **stable pool of 1–3 device UUIDs** at creation, then sampled consistently across all sessions (enabling multi-device tracking as a potential fraud signal).

| Aspect | LOW | MEDIUM | HIGH |
|--------|-----|--------|------|
| Login probability/day | 20% | 50% | 90% |
| Sessions when active | 1–3 | 1–3 | 1–8 (+0–5 bonus) |
| Late-night logins (00:00–05:00) | No | No | 30% chance |
| Balance checks per session | 0–1 | 0–1 | 1–4 (repeated) |

Events per session: `LOGIN` → `VIEW_DASHBOARD` → optional `CHECK_BALANCE` (×1–4 for HIGH).

### 4.4 Support Log Generation (`generate_support_logs()`)

Generates tickets on a **monthly probability loop** (not just one ticket total per customer):

| Aspect | LOW/MEDIUM | HIGH |
|--------|-----------|------|
| Monthly ticket probability | 5% / 20% | 50% |
| Topics | Info Request, Address Change, Card Replacement | Fees Dispute, Limit Increase, Payment Extension, Overdraft Query |
| Sentiment | Neutral (80%) / Positive (20%) | Negative (50%) / Anxious (40%) / Aggressive (10%) |

### 4.5 Bill Schedule Generation (`generate_bill_schedules()`)

One row per customer. Maps `customer_id` → `bill_type` (always ELECTRICITY), `due_date_day`, `expected_amount` (₹500–₹3,000).

---

## 5. Database Setup

**Script:** `setup_database.py` (50 lines)

Loads four CSVs into SQLite (`banking_data.db`):

| Table | Source CSV | Indexes Created |
|-------|-----------|-----------------|
| `customers` | customers.csv | `idx_cust_id` on `customer_id` |
| `transactions` | transactions.csv | `idx_txn_cust_id` on `customer_id`, `idx_txn_date` on `transaction_date` |
| `app_logs` | app_logs.csv | `idx_app_cust_id` on `customer_id` |
| `support_logs` | support_logs.csv | `idx_support_cust_id` on `customer_id` |

All tables use `if_exists="replace"` — safe to re-run. Database size is approximately 700 MB at 5,000 customers.

---

## 6. Feature Engineering

**Script:** `engineer_features.py` (~160 lines)

Reads from SQLite and produces `feature_store.csv` with **29 columns** (25 model features + `customer_id`, `risk_level`, `monthly_income`, `is_delinquent`).

Features are computed via four SQL aggregation groups, then merged and enhanced with derived ratios. See [Feature Catalog](#13-feature-catalog) for the full list.

### 6.1 Feature Groups

**Group 1 — Lifetime Transaction Aggregates** (10 features)
SQL aggregation over the full `transactions` table: counts, averages, category breakdowns, bounce/failure counts.

**Group 2 — Temporal / Windowed Features** (10 features)
Time-windowed SQL queries using cutoff dates (30d, 60d, 90d from `now()`). Captures behavioral **change** — the key signal in pre-delinquency detection.

Three derived ratios computed in Python after the SQL query:
- `atm_trend_ratio` = ATM withdrawals last 30d / (ATM prior 30d + 1)
- `balance_decline_pct` = (avg_balance_last_30d - avg_balance_prior) / (|avg_balance_prior| + 1)
- `gambling_acceleration` = gambling_last_30d / (gambling_prior + 1)

**Group 3 — App-Log Features** (5 features)
Login counts (total, last 30d, prior 30d), balance check frequency, late-night login count. One derived ratio: `login_freq_change`.

**Group 4 — Support-Log Features** (3 features)
Ticket count, negative sentiment count (Negative/Anxious/Aggressive), distress topic count (Fees Dispute/Limit Increase/Payment Extension/Overdraft Query).

### 6.2 Ground-Truth Label (`is_delinquent`)

See [Ground-Truth Labeling Logic](#14-ground-truth-labeling-logic) for the full derivation.

### 6.3 Internal Call to `split_data.py`

After saving `feature_store.csv`, the script imports and calls `split_data()` to generate train/val/test splits.

---

## 7. Data Splitting

**Script:** `split_data.py` (~50 lines)

Reads `feature_store.csv` and produces three CSVs via two-stage stratified split on `is_delinquent`:

| Split | Ratio | Rows |
|-------|-------|------|
| `train_data.csv` | 70% | 3,500 |
| `val_data.csv` | 15% | 750 |
| `test_data.csv` | 15% | 750 |

Method: `train_test_split(test_size=0.3, stratify)` → `train_test_split(test_size=0.5, stratify)` on remainder. Random state 42.

---

## 8. XGBoost Training

**Script:** `train_xgboost.py` (~100 lines)

### 8.1 Model Configuration

```
Objective:            binary:logistic
n_estimators:         100
learning_rate:        0.1
max_depth:            5
eval_metric:          logloss
early_stopping_rounds: 10
```

Trains on 25 features (see [Feature Catalog](#13-feature-catalog)). Validates on `val_data.csv` with early stopping.

### 8.2 Feature List (25 features used for training)

```
Lifetime:   avg_txn_amt, min_balance, avg_balance, lending_app_txns, gambling_txns,
            atm_txns, bounced_count, failed_txn_count

Temporal:   bounce_count_last_60d, lending_app_last_60d, days_since_last_salary,
            salary_credit_count, failed_txn_last_30d, late_utility_payments,
            atm_trend_ratio, balance_decline_pct, gambling_acceleration

App-Log:    total_logins, balance_check_count, login_freq_change, late_night_logins

Support:    support_ticket_count, negative_sentiment_count, distress_topic_count
```

### 8.3 Evaluation Metrics

Evaluated on `test_data.csv` (750 rows, held-out). Outputs standard classification report, confusion matrix, and accuracy.

### 8.4 SHAP Explainability

Generates a global SHAP summary plot (`shap_summary_delinquency.png`) showing feature importance and directional impact. Saves `shap_explainer.pkl` for dashboard reuse.

### 8.5 Portfolio Scoring

After training, scores the **full** `feature_store.csv` (5,000 customers) and appends two columns:
- `risk_score_prob`: raw P(delinquent) from the model
- `risk_score_scaled`: `int(risk_score_prob * 100)`, capped 0–100

Saved as `model_results_xgb.csv`.

### 8.6 Artifacts Produced

| File | Description |
|------|-------------|
| `xgboost_model.pkl` | Trained XGBClassifier |
| `shap_explainer.pkl` | SHAP Explainer object |
| `shap_summary_delinquency.png` | Global SHAP beeswarm plot |
| `model_results_xgb.csv` | Full portfolio with risk scores (31 columns) |

---

## 9. Isolation Forest Training

**Script:** `train_model.py` (~80 lines)

### 9.1 Design: Purely Unsupervised

The Isolation Forest runs as a **safety net** — it never sees the `is_delinquent` label during training or threshold selection. This ensures it can catch anomalous patterns that the supervised model might miss.

### 9.2 Configuration

```
contamination:  0.1 (initial guess for tree structure)
Scaler:         StandardScaler (fit on train only)
Threshold:      10th percentile of training score distribution
```

### 9.3 Threshold Selection

Instead of tuning against supervised labels, the threshold is set to the 10th percentile of `score_samples()` on the training set. This means roughly 10% of training data will be flagged as anomalous — a reasonable assumption for the bottom tail.

### 9.4 Portfolio Scoring

Scores the full `feature_store.csv` and appends:
- `anomaly_score`: raw IF score (lower = more anomalous)
- `is_anomaly`: `1` (normal) or `-1` (anomaly), based on threshold
- `ground_truth`: maps `is_delinquent` → `-1` if delinquent, `1` if not (for dashboard comparison only)

### 9.5 Artifacts Produced

| File | Description |
|------|-------------|
| `isolation_forest.pkl` | Trained IsolationForest model |
| `scaler.pkl` | StandardScaler (fitted on training features) |
| `threshold.txt` | Numeric threshold value |
| `model_results.csv` | Full portfolio with anomaly scores (32 columns) |

---

## 10. Dashboard

**Script:** `dashboard.py` (172 lines) — Streamlit application

### 10.1 Pages

| Page | Description |
|------|-------------|
| **Portfolio Overview** | KPIs (total customers, high-risk >80, medium 50–80), risk score histogram, risk-vs-balance scatter plot |
| **Model Explainability** | Displays `shap_summary_delinquency.png` with interpretation guide |
| **Customer 360** | Customer-level drill-down: risk score, balance, bounces, gambling txns, rule-based risk factor annotations, savings drawdown trajectory chart |
| **Intervention Queue** | Top-10 high-risk customers (score ≥80) sorted desc, with WhatsApp offer / Call action buttons |
| **Safety Net (Unsupervised)** | IF anomaly count, discrepancy analysis (IF says anomaly but XGBoost says safe), scatter visualization of IF separation |

### 10.2 Data Sources

- `model_results_xgb.csv` — primary risk scores
- `model_results.csv` — IF anomaly flags
- `banking_data.db` → `transactions` table — for customer-level transaction history charts

### 10.3 Running

```bash
streamlit run data_generation/dashboard.py
```

---

## 11. Utility Scripts

### `verify_data.py`
Quick sanity checker: loads `customers.csv` and `transactions.csv`, prints counts of bounced/gambling transactions, samples one customer's balance flow, and counts negative-balance transactions.

### `analyze_practicality.py`
Deeper statistical analysis across 7 signal dimensions:
1. **Salary delays** — standard deviation of salary day by risk level
2. **Lending app usage** — which risk tiers use lending apps
3. **Auto-debit failures** — bounce distribution by risk level
4. **Savings drawdown** — linear regression slope of balance per customer (sampled 50 customers)
5. **Behavioral risks** — gambling and ATM withdrawal counts by risk tier
6. **Utility bill lateness** — average payment delay relative to due date
7. **Discretionary spend ratio** — (FOOD + SHOPPING) / total transactions by risk level

---

## 12. CSV Schemas

### Raw Data (Generated)

#### `customers.csv` — 5,000 rows, 6 columns
| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| `customer_id` | UUID string | v4 UUID | Primary key |
| `name` | string | Indian names | Faker `en_IN` |
| `age` | int | 22–55 | Random |
| `monthly_income` | int | 30,000–150,000 | Monthly salary in ₹ |
| `risk_level` | string | LOW / MEDIUM / HIGH | Generation tier (60/30/10 split) |
| `bill_due_date` | int | 1–27 | Day of month for utility bills |

> **Note:** The `profile` dict column is intentionally excluded from the CSV to prevent data leakage. It exists only in-memory during generation.

#### `transactions.csv` — ~1,385,550 rows, 8 columns
| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| `customer_id` | UUID string | FK → customers | |
| `transaction_date` | datetime | 6-month window | Timestamp |
| `amount` | int | -10,000 to +150,000 (0 for FAILED) | Negative = debit, positive = credit |
| `type` | string | DEBIT / CREDIT | Direction |
| `category` | string | SALARY, FOOD, SHOPPING, UPI_P2P, UPI_LENDING_APP, ATM_WITHDRAWAL, GAMBLING, UTILITY, LOAN_EMI | Transaction category |
| `description` | string | e.g. "UPI/Arya/SHOPPING" | Human-readable label |
| `status` | string | SUCCESS / BOUNCED / FAILED | BOUNCED = EMI rejection; FAILED = overdraft block |
| `running_balance` | int | varies | Balance after this transaction |

#### `app_logs.csv` — ~2,644,696 rows, 4 columns
| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| `customer_id` | UUID string | FK → customers | |
| `timestamp` | datetime | 6-month window | Event time |
| `event_type` | string | LOGIN / VIEW_DASHBOARD / CHECK_BALANCE | App event |
| `device_id` | UUID string | 1–3 stable devices per customer | Device identifier |

#### `support_logs.csv` — ~4,698 rows, 6 columns
| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| `customer_id` | UUID string | FK → customers | |
| `date` | datetime | 6-month window | Ticket date |
| `topic` | string | Info Request, Address Change, Card Replacement, Fees Dispute, Limit Increase, Payment Extension, Overdraft Query | Ticket topic |
| `sentiment` | string | Neutral / Positive / Negative / Anxious / Aggressive | Customer sentiment |
| `description` | string | Template text | Ticket body |
| `channel` | string | Chat / Call / Email | Contact channel |

#### `bill_schedules.csv` — 5,000 rows, 4 columns
| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| `customer_id` | UUID string | FK → customers | |
| `bill_type` | string | ELECTRICITY | Always ELECTRICITY |
| `due_date_day` | int | 1–27 | Same as customer's `bill_due_date` |
| `expected_amount` | int | 500–3,000 | Expected bill amount |

---

### Engineered Data

#### `feature_store.csv` — 5,000 rows, 29 columns
See [Feature Catalog](#13-feature-catalog) for each column's derivation and meaning. Includes `customer_id`, 25 model features, `risk_level`, `monthly_income`, and `is_delinquent`.

#### `train_data.csv` / `val_data.csv` / `test_data.csv`
Same 29-column schema as `feature_store.csv`.

| Split | Rows | Purpose |
|-------|------|---------|
| train | 3,500 | Model fitting |
| val | 750 | Early stopping / hyperparameter tuning |
| test | 750 | Final held-out evaluation |

---

## 13. Feature Catalog

25 features organized into four groups. All are used by both XGBoost and Isolation Forest.

### Group A: Lifetime Transaction Aggregates (8 features)

| # | Feature | SQL/Derivation | Typical Range | Description |
|---|---------|---------------|---------------|-------------|
| 1 | `total_txns` | `COUNT(*)` from transactions | 218–355 | Total transaction count over 6 months |
| 2 | `avg_txn_amt` | `AVG(amount)` | -924 to +1,395 | Average transaction amount (negative-heavy = more debits) |
| 3 | `min_balance` | `MIN(running_balance)` | -74,904 to +39,830 | Lowest balance ever reached |
| 4 | `avg_balance` | `AVG(running_balance)` | -35,036 to +257,935 | Mean balance across all transactions |
| 5 | `lending_app_txns` | `COUNT WHERE category = 'UPI_LENDING_APP'` | 0–49 | Count of lending app transactions |
| 6 | `gambling_txns` | `COUNT WHERE category = 'GAMBLING'` | 0–51 | Count of gambling transactions |
| 7 | `atm_txns` | `COUNT WHERE category = 'ATM_WITHDRAWAL'` | 2–185 | Count of ATM withdrawals |
| 8 | `bounced_count` | `COUNT WHERE status = 'BOUNCED'` | 0–6 | EMI auto-debit failures |
| 9 | `failed_txn_count` | `COUNT WHERE status = 'FAILED'` | 0–263 | Overdraft-blocked transactions |

### Group B: Temporal / Windowed Features (9 features)

| # | Feature | Window | Typical Range | Description |
|---|---------|--------|---------------|-------------|
| 10 | `bounce_count_last_60d` | Last 60 days | 0–2 | Recent EMI bounces |
| 11 | `lending_app_last_60d` | Last 60 days | 0–22 | Recent lending app usage |
| 12 | `days_since_last_salary` | From latest salary | 4–13 | Recency of income (higher = possible delay) |
| 13 | `salary_credit_count` | Lifetime | 6 (usually constant) | Number of salary credits received |
| 14 | `failed_txn_last_30d` | Last 30 days | 0–55 | Recent overdraft blocks (acute stress) |
| 15 | `late_utility_payments` | Lifetime | 0–13 | Utility payments made after day 20 of month |
| 16 | `atm_trend_ratio` | Last 30d / prior 30d | 0.0–7.0 | ATM withdrawal acceleration (>1 = increasing) |
| 17 | `balance_decline_pct` | Last 30d vs prior | -973 to +173 | Percentage balance change (negative = drawdown) |
| 18 | `gambling_acceleration` | Last 30d / prior | 0.0–2.0 | Gambling frequency acceleration |

### Group C: App-Log Features (4 features)

| # | Feature | Derivation | Typical Range | Description |
|---|---------|-----------|---------------|-------------|
| 19 | `total_logins` | `COUNT(*)` from app_logs | 78–3,589 | Total app events over 6 months |
| 20 | `balance_check_count` | `COUNT WHERE event_type = 'CHECK_BALANCE'` | 0–2,095 | Anxiety proxy — frequent checking |
| 21 | `late_night_logins` | `COUNT WHERE hour < 5` | 0–671 | Midnight–5am activity |
| 22 | `login_freq_change` | logins_last_30d / (logins_prior_30d + 1) | 0.0–30.0 | Login frequency acceleration |

### Group D: Support-Log Features (3 features)

| # | Feature | Derivation | Typical Range | Description |
|---|---------|-----------|---------------|-------------|
| 23 | `support_ticket_count` | `COUNT(*)` from support_logs | 0–7 | Total support contacts over 6 months |
| 24 | `negative_sentiment_count` | `COUNT WHERE sentiment IN ('Negative','Anxious','Aggressive')` | 0–7 | Distressed customer interactions |
| 25 | `distress_topic_count` | `COUNT WHERE topic IN ('Fees Dispute','Limit Increase','Payment Extension','Overdraft Query')` | 0–7 | Financial-distress topic tickets |

---

## 14. Ground-Truth Labeling Logic

The `is_delinquent` label is **derived post-generation** from actual behavioral outcomes — it is never assigned before data generation. This prevents circular labeling where the model simply learns the generation rules.

### 14.1 Stress Score Computation

A continuous stress score is computed from 14 weighted signals across four data sources:

```
raw_stress =
    (bounced_count >= 1)           × 2.0    # EMI bounce history
  + (min_balance < 0)              × 1.0    # Ever went negative
  + (avg_balance < 10,000)         × 1.0    # Chronically low savings
  + (gambling_txns > 5)            × 1.0    # Gambling behavior
  + (lending_app_txns > 5)         × 1.0    # Borrowing from lending apps
  + (atm_txns > 80)               × 0.5    # Cash hoarding
  + (balance_check_count > 200)    × 0.5    # Anxious app behavior
  + (bounce_count_last_60d >= 1)   × 1.5    # Recent bounces (temporal)
  + (balance_decline_pct < -0.3)   × 1.0    # Balance drawdown (temporal)
  + (days_since_last_salary > 10)  × 0.5    # Salary delay (temporal)
  + (late_utility_payments >= 2)   × 0.5    # Late bill payments (temporal)
  + (negative_sentiment_count >= 2)× 1.0    # Distressed support calls
  + (distress_topic_count >= 2)    × 0.5    # Financial distress topics
  + (failed_txn_last_30d >= 3)     × 0.5    # Recent overdraft blocks
```

Maximum possible score: **12.0**

### 14.2 Probabilistic Labeling (Anti-Overfitting)

Instead of applying a hard threshold (which would make the label a deterministic function of the features and allow the model to achieve ~100% accuracy by reverse-engineering the rule), the score is converted to a probability via a **sigmoid function**, then labels are drawn stochastically:

```
delinquency_prob = 1 / (1 + exp(-1.2 × (raw_stress - 5.0)))
is_delinquent = Bernoulli(delinquency_prob)
```

- **Centre:** score = 5.0 → 50% probability
- **Steepness:** k = 1.2 (moderate — not too sharp, not too flat)
- **Effect:** A customer with stress score 3 has ~8% chance of being labeled delinquent; score 7 has ~92% chance. This introduces realistic label noise.

### 14.3 Resulting Distribution

| Metric | Value |
|--------|-------|
| Overall delinquency rate | ~37% |
| HIGH-risk tier rate | ~100% |
| MEDIUM-risk tier rate | ~66% |
| LOW-risk tier rate | ~12% |

The non-zero LOW-risk delinquency rate (12%) and non-100% (though near-100%) HIGH-risk rate confirm that no single feature or simple rule can perfectly predict the label.

---

## 15. Model Output Schemas

### `model_results_xgb.csv` — 5,000 rows, 31 columns

All 29 columns from `feature_store.csv` plus:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `risk_score_prob` | float | 0.0–1.0 | P(delinquent) from XGBoost |
| `risk_score_scaled` | int | 0–100 | `int(risk_score_prob × 100)` |

Dashboard thresholds: **≥80 = High Risk**, **50–79 = Medium Risk (Watchlist)**, **<50 = Low Risk**.

### `model_results.csv` — 5,000 rows, 32 columns

All 29 columns from `feature_store.csv` plus:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `anomaly_score` | float | negative values | Raw IF score (lower = more anomalous) |
| `is_anomaly` | int | 1 or -1 | 1 = normal, -1 = anomaly |
| `ground_truth` | int | 1 or -1 | Mapped from `is_delinquent`: 1→-1, 0→1 |

---

## 16. Design Decisions & Rationale

### Why derive the label post-generation instead of using `risk_level`?

The original pipeline assigned `risk_level` (LOW/MEDIUM/HIGH) *before* generating data, then generated transactions *to match* the assigned level. The model's job was trivially easy — it just learned the generation rules. By deriving `is_delinquent` from actual behavioral outcomes *after* all data is generated, the model must learn genuine multi-signal patterns.

### Why probabilistic labeling?

A deterministic threshold (e.g., "score ≥ 5 → delinquent") makes the target a mechanical function of the features. XGBoost will learn to replicate that threshold exactly, yielding 99%+ accuracy that doesn't generalize. The sigmoid + Bernoulli sampling introduces realistic label noise — mimicking the real world where two customers with similar behavior can have different outcomes. This forces the model to learn probabilistic risk estimation rather than memorizing a rule.

### Why cross-contaminate gambling/lending across risk tiers?

In reality, some financially healthy customers gamble occasionally, and some stressed customers never touch lending apps. Without overlap, `gambling_txns > 0` alone achieves perfect classification. The 0.5–3% base rates for LOW/MEDIUM tiers create realistic feature distributions with overlapping tails.

### Why is the Isolation Forest threshold unsupervised?

Tuning the IF threshold against `is_delinquent` labels defeats its purpose as a safety net for "unknown unknowns." By using the 10th percentile of training scores, the IF flags the statistical bottom tail regardless of what the supervised model knows. The dashboard highlights where IF and XGBoost disagree — these discrepancies are the highest-value cases for manual review.

### Why add temporal features?

Lifetime aggregates (e.g., total bounce count) don't capture **behavioral change** — the core signal in pre-delinquency. A customer who bounced 3 EMIs over 6 months is different from one who bounced 3 EMIs in the last 60 days. Temporal features like `bounce_count_last_60d`, `balance_decline_pct`, and `atm_trend_ratio` capture acceleration and trajectory.

### Why include support-log features?

Customers experiencing financial distress often contact support with specific patterns: topics like "Payment Extension" or "Overdraft Query" with negative/anxious sentiment. These are leading indicators that transaction data alone may not capture — a customer might call about payment difficulties *before* actually missing a payment.

---

## 17. Known Limitations & Future Work

### Current Limitations

1. **Bill schedule redundancy:** `bill_schedules.csv` only has ELECTRICITY and duplicates `bill_due_date` from `customers.csv`. Could be expanded to multiple bill types (rent, insurance, subscriptions).

2. **No multi-account simulation:** Each customer has exactly one account. Real banking involves savings + current + credit card interactions.

3. **Static income:** `monthly_income` never changes. Real income can be irregular (freelancers) or change (job loss).

4. **Single EMI per customer:** All customers have exactly one loan EMI at 30% of income. Real portfolios have varying loan counts and EMI ratios.

5. **No geographic features:** Location data could add signals (urban vs rural spending patterns).

### Potential Enhancements

1. **Advanced temporal features:** Rolling 7-day balance slope via `scipy.stats.linregress`, weekend-vs-weekday spending ratios, time-since-last-ATM-withdrawal trends.

2. **Network features:** UPI P2P graph analysis — customers frequently transacting with known delinquent customers.

3. **Multi-product scoring:** Separate delinquency scores per loan product (home loan, personal loan, credit card).

4. **Ensemble safety net:** Add autoencoders or DBSCAN alongside Isolation Forest for richer anomaly detection.

5. **Real-time scoring pipeline:** Adapt the batch pipeline to stream processing for live intervention triggers.

---

## Appendix: File Inventory

| File | Type | Size (approx) | Description |
|------|------|----------------|-------------|
| `generate_synthetic_data.py` | Script | 367 lines | Data generation |
| `setup_database.py` | Script | 50 lines | CSV → SQLite loader |
| `engineer_features.py` | Script | 160 lines | Feature engineering + labeling |
| `split_data.py` | Script | 50 lines | Train/val/test split |
| `train_xgboost.py` | Script | 100 lines | Supervised model training |
| `train_model.py` | Script | 80 lines | Unsupervised model training |
| `dashboard.py` | Script | 172 lines | Streamlit dashboard |
| `verify_data.py` | Script | 40 lines | Quick data checker |
| `analyze_practicality.py` | Script | 133 lines | Statistical analysis |
| `customers.csv` | Data | 5,000 rows | Customer master |
| `transactions.csv` | Data | ~1.39M rows (~155 MB) | Transaction history |
| `app_logs.csv` | Data | ~2.64M rows (~285 MB) | App interaction logs |
| `support_logs.csv` | Data | ~4,698 rows | Support tickets |
| `bill_schedules.csv` | Data | 5,000 rows | Bill due dates |
| `feature_store.csv` | Data | 5,000 rows, 29 cols | Engineered features + label |
| `train_data.csv` | Data | 3,500 rows | Training split |
| `val_data.csv` | Data | 750 rows | Validation split |
| `test_data.csv` | Data | 750 rows | Test split |
| `model_results_xgb.csv` | Data | 5,000 rows, 31 cols | XGBoost scored portfolio |
| `model_results.csv` | Data | 5,000 rows, 32 cols | IF scored portfolio |
| `banking_data.db` | DB | ~700 MB | SQLite database |
| `xgboost_model.pkl` | Model | — | Trained XGBClassifier |
| `isolation_forest.pkl` | Model | — | Trained IsolationForest |
| `scaler.pkl` | Model | — | Fitted StandardScaler |
| `shap_explainer.pkl` | Model | — | SHAP Explainer |
| `shap_summary_delinquency.png` | Image | — | SHAP beeswarm plot |
| `threshold.txt` | Config | 1 line | IF anomaly threshold |
