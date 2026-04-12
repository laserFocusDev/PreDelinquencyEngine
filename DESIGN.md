# Pre-Delinquency Intervention Engine - Hackathon Edition

**Version:** 1.0 (Hackathon)  
**Last Updated:** February 11, 2026  
**Timeline:** 7 Days  
**Status:** Hackathon Project - Open Source Stack

---

## Table of Contents

1. [Hackathon Overview](#hackathon-overview)
2. [Tech Stack (Open Source)](#tech-stack-open-source)
3. [System Architecture](#system-architecture)
4. [Phase-by-Phase Development](#phase-by-phase-development)
5. [Data Model](#data-model)
6. [ML Strategy (Simplified)](#ml-strategy-simplified)
7. [Demo Scenarios](#demo-scenarios)
8. [Setup Instructions](#setup-instructions)

---

## Hackathon Overview

### Purpose
Build a working prototype of an AI-powered Pre-Delinquency Intervention Engine that demonstrates:
- Early detection of customers at financial risk
- Predictive risk scoring using ML
- Automated intervention recommendations
- Visual dashboards for monitoring

### Hackathon Goals
- **Day 1-2:** Data pipeline + Basic ML model → **MVP 1: Risk Scoring**
- **Day 3-4:** Real-time scoring + Basic UI → **MVP 2: Live Dashboard**
- **Day 5-6:** Interventions + Notifications → **MVP 3: Complete System**
- **Day 7:** Polish, documentation, final demo

### Key Features (Hackathon Scope)
- Simulated banking transaction data (10,000 customers)
- XGBoost model for 28-day default prediction
- Real-time risk scoring via API
- Web dashboard showing risk alerts
- Mock intervention system (SMS/Email simulation)
- All running on localhost with Docker Compose

---

## Tech Stack (Open Source)

### Core Technologies - All Local/Open Source

**Development Environment:**
- **Docker & Docker Compose** - Container orchestration
- **Python 3.10+** - Primary language
- **Node.js 18+** - Frontend (optional React dashboard)

**Data Layer:**
- **PostgreSQL 15** - Primary database
- **Redis** - Caching and simple message queue
- **CSV/JSON files** - Simulated transaction data

**ML/AI:**
- **scikit-learn** - XGBoost model training
- **pandas, numpy** - Data processing
- **SHAP** - Model explainability
- **Joblib** - Model serialization

**Backend:**
- **FastAPI** - REST API framework
- **SQLAlchemy** - ORM
- **Alembic** - Database migrations
- **Pydantic** - Data validation

**Frontend:**
- **Streamlit** - Quick dashboard (preferred for hackathon)
- **OR React + Chart.js** - If time permits
- **Plotly** - Interactive visualizations

**Monitoring:**
- **Prometheus** - Metrics (optional)
- **Grafana** - Dashboards (optional)
- **Simple logging** - Python logging to files

**Version Control:**
- **Git/GitHub** - Source control
- **GitHub Actions** - Basic CI (optional)

---

## System Architecture

### Simplified Hackathon Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA GENERATION                          │
│  Python script generates synthetic transaction data         │
│  → 10,000 customers, 6 months of transactions              │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                   POSTGRESQL DATABASE                        │
│  Tables: customers, accounts, transactions,                 │
│          customer_features, risk_scores, interventions       │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│              FEATURE ENGINEERING SERVICE                     │
│  Python script computes features from raw transactions      │
│  → Salary delay, savings decline, spending patterns         │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                    ML MODEL (XGBoost)                        │
│  Train on historical data → Predict 28-day default risk     │
│  Save model with Joblib → Serve via FastAPI                 │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                   FASTAPI BACKEND                            │
│  /predict - Get risk score for customer                     │
│  /batch-score - Score all customers                         │
│  /interventions - Get/create interventions                  │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                 STREAMLIT DASHBOARD                          │
│  - Risk distribution chart                                  │
│  - High-risk customer list                                  │
│  - Feature importance visualization                         │
│  - Mock intervention sender                                 │
└─────────────────────────────────────────────────────────────┘
```

### Docker Compose Services

```yaml
services:
  postgres:
    image: postgres:15
    ports: ["5432:5432"]
    volumes: ["./data/postgres:/var/lib/postgresql/data"]
  
  redis:
    image: redis:7
    ports: ["6379:6379"]
  
  backend:
    build: ./backend
    ports: ["8000:8000"]
    depends_on: [postgres, redis]
  
  dashboard:
    build: ./dashboard
    ports: ["8501:8501"]
    depends_on: [backend]
```

### Project Structure

```
pre-delinquency-engine/
├── docker-compose.yml
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── raw/                    # Synthetic transaction CSVs
│   ├── processed/              # Engineered features
│   └── models/                 # Saved ML models
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── data_generation/
│   │   └── generate_synthetic_data.py
│   │
│   ├── feature_engineering/
│   │   ├── feature_calculator.py
│   │   └── feature_definitions.py
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   ├── predict.py
│   │   └── explainer.py
│   │
│   └── database/
│       ├── schema.sql
│       ├── models.py (SQLAlchemy)
│       └── connection.py
│
├── backend/
│   ├── Dockerfile
│   ├── main.py (FastAPI app)
│   ├── routers/
│   │   ├── predictions.py
│   │   ├── customers.py
│   │   └── interventions.py
│   └── services/
│       ├── risk_scorer.py
│       └── recommendation_engine.py
│
└── dashboard/
    ├── Dockerfile
    ├── app.py (Streamlit)
    └── pages/
        ├── 1_risk_overview.py
        ├── 2_customer_details.py
        └── 3_interventions.py
```

---

## Phase-by-Phase Development

### 7-Day Timeline with Incremental MVPs

---

## **PHASE 1: Data + Basic Model (Days 1-2)**
### **MVP 1: "We can detect risky customers"**

**Goal:** Generate synthetic data, train a basic XGBoost model, and score customers

**Day 1 Morning (4 hours):**
- **TASK 1.1:** Project setup
  - Initialize Git repo
  - Create docker-compose.yml (Postgres + Redis)
  - Set up Python virtual environment
  - Install dependencies (pandas, scikit-learn, xgboost, fastapi, sqlalchemy)

- **TASK 1.2:** Data generation script
  ```python
  # generate_synthetic_data.py
  # Create 10,000 customers
  # 6 months of transactions per customer
  # Include: salary credits, bill payments, spending, savings
  # Simulate ~5% default rate
  ```

**Day 1 Afternoon (4 hours):**
- **TASK 1.3:** Database schema
  - Create PostgreSQL tables (customers, accounts, transactions)
  - Write data loading script
  - Verify data in DB

- **TASK 1.4:** Basic EDA in Jupyter notebook
  - Transaction distributions
  - Customer segments
  - Delinquency patterns

**Day 2 Morning (4 hours):**
- **TASK 1.5:** Feature engineering
  ```python
  # Calculate 15 basic features:
  - salary_delay_days
  - savings_decline_pct_7d
  - failed_autodebits_count
  - discretionary_spending_ratio
  - utility_payment_delay
  # ... etc
  ```

- **TASK 1.6:** Label creation
  - Define "default" as missing payment in next 28 days
  - Create train/test split (80/20)

**Day 2 Afternoon (4 hours):**
- **TASK 1.7:** Train XGBoost model
  ```python
  from xgboost import XGBClassifier
  from sklearn.metrics import roc_auc_score
  
  model = XGBClassifier(
      max_depth=6,
      learning_rate=0.1,
      n_estimators=100
  )
  model.fit(X_train, y_train)
  
  # Target: AUC-ROC > 0.75
  ```

- **TASK 1.8:** Model evaluation & save
  - Calculate metrics (accuracy, precision, recall, AUC)
  - Save model with Joblib
  - Generate SHAP explainability

**MVP 1 Demo:**
- Show synthetic data in database
- Show model accuracy metrics
- Show SHAP feature importance chart
- Run batch scoring on 100 customers

**Deliverables:**
- ✅ PostgreSQL database with 10K customers
- ✅ XGBoost model (saved as .joblib file)
- ✅ Jupyter notebook with EDA & model training
- ✅ CSV file with risk scores for all customers

---

## **PHASE 2: API + Live Dashboard (Days 3-4)**
### **MVP 2: "We have a working system with UI"**

**Goal:** Build FastAPI backend and Streamlit dashboard for real-time risk scoring

**Day 3 Morning (4 hours):**
- **TASK 2.1:** FastAPI setup
  ```python
  # backend/main.py
  from fastapi import FastAPI
  from pydantic import BaseModel
  
  app = FastAPI()
  
  @app.get("/customers/{customer_id}/risk-score")
  def get_risk_score(customer_id: str):
      # Load features from DB
      # Load model
      # Predict
      # Return risk score + tier
  
  @app.post("/batch-score")
  def batch_score():
      # Score all customers
      # Save to risk_scores table
  ```

- **TASK 2.2:** Database models (SQLAlchemy)
  ```python
  class Customer(Base):
      __tablename__ = "customers"
      customer_id = Column(String, primary_key=True)
      name = Column(String)
      # ...
  
  class RiskScore(Base):
      __tablename__ = "risk_scores"
      id = Column(Integer, primary_key=True)
      customer_id = Column(String, ForeignKey("customers.customer_id"))
      risk_score = Column(Float)
      risk_tier = Column(String)  # GREEN, YELLOW, ORANGE, RED
      created_at = Column(DateTime)
  ```

**Day 3 Afternoon (4 hours):**
- **TASK 2.3:** API endpoints
  - `/customers` - List all customers
  - `/customers/{id}` - Get customer details
  - `/customers/{id}/risk-score` - Get latest risk score
  - `/customers/{id}/features` - Get feature breakdown
  - `/batch-score` - Score all customers

- **TASK 2.4:** Add caching with Redis
  - Cache risk scores (TTL: 1 hour)
  - Cache feature vectors (TTL: 30 min)

**Day 4 Morning (4 hours):**
- **TASK 2.5:** Streamlit dashboard - Page 1: Overview
  ```python
  # dashboard/app.py
  import streamlit as st
  import requests
  import plotly.express as px
  
  st.title("Pre-Delinquency Risk Dashboard")
  
  # Risk distribution pie chart
  # High-risk customers table
  # Key metrics (total customers, % at risk)
  ```

- **TASK 2.6:** Streamlit dashboard - Page 2: Customer Details
  ```python
  # Select customer from dropdown
  # Show risk score gauge chart
  # Show top features (SHAP waterfall chart)
  # Show transaction history
  ```

**Day 4 Afternoon (4 hours):**
- **TASK 2.7:** Docker Compose integration
  ```yaml
  services:
    postgres:
      image: postgres:15
    redis:
      image: redis:7
    backend:
      build: ./backend
      ports: ["8000:8000"]
    dashboard:
      build: ./dashboard
      ports: ["8501:8501"]
  ```

- **TASK 2.8:** API documentation (Swagger)
  - Auto-generated with FastAPI
  - Test all endpoints

**MVP 2 Demo:**
- Navigate to http://localhost:8501
- Show risk distribution chart
- Click on high-risk customer
- Show their risk score and explanation
- Show real-time API calls (FastAPI Swagger UI)

**Deliverables:**
- ✅ FastAPI backend with 6+ endpoints
- ✅ Streamlit dashboard (2 pages)
- ✅ Docker Compose for all services
- ✅ Redis caching working
- ✅ API documentation (Swagger)

---

## **PHASE 3: Interventions + Notifications (Days 5-6)**
### **MVP 3: "We can recommend and send interventions"**

**Goal:** Add intervention logic and simulate sending notifications

**Day 5 Morning (4 hours):**
- **TASK 3.1:** Intervention recommendation engine
  ```python
  # backend/services/recommendation_engine.py
  
  def generate_recommendations(risk_tier, customer_features):
      recommendations = []
      
      if risk_tier in ['ORANGE', 'RED']:
          # Payment holiday
          if customer_features['payment_history_score'] > 70:
              recommendations.append({
                  'type': 'PAYMENT_HOLIDAY',
                  'duration_months': 1,
                  'message': 'Take a break - skip 1 EMI payment'
              })
          
          # EMI reduction
          recommendations.append({
              'type': 'EMI_REDUCTION',
              'reduction_pct': 25,
              'message': 'Reduce your EMI by 25% for 6 months'
          })
      
      return recommendations
  ```

- **TASK 3.2:** Interventions database table
  ```sql
  CREATE TABLE interventions (
      id SERIAL PRIMARY KEY,
      customer_id VARCHAR(50),
      risk_score_id INTEGER,
      intervention_type VARCHAR(50),
      status VARCHAR(20),  -- PENDING, SENT, ACCEPTED, REJECTED
      channel VARCHAR(20), -- SMS, EMAIL, APP
      message_text TEXT,
      sent_at TIMESTAMP,
      responded_at TIMESTAMP,
      outcome VARCHAR(50)
  );
  ```

**Day 5 Afternoon (4 hours):**
- **TASK 3.3:** API endpoints for interventions
  - `POST /interventions` - Create intervention
  - `GET /interventions/{id}` - Get intervention details
  - `PUT /interventions/{id}/send` - Simulate sending
  - `PUT /interventions/{id}/accept` - Customer accepts

- **TASK 3.4:** Mock notification service
  ```python
  # backend/services/notification_service.py
  
  def send_sms(customer_id, message):
      # Log to console/file instead of actually sending
      logger.info(f"[SMS] To: {customer_id}, Message: {message}")
      return {"status": "sent", "message_id": uuid.uuid4()}
  
  def send_email(customer_id, subject, body):
      logger.info(f"[EMAIL] To: {customer_id}, Subject: {subject}")
      return {"status": "sent", "message_id": uuid.uuid4()}
  ```

**Day 6 Morning (4 hours):**
- **TASK 3.5:** Streamlit dashboard - Page 3: Interventions
  ```python
  # Show list of high-risk customers
  # Button: "Generate Recommendations"
  # Display recommended interventions
  # Button: "Send SMS" / "Send Email" (mock)
  # Show "sent" status with timestamp
  ```

- **TASK 3.6:** Automated intervention trigger
  ```python
  # Script runs daily (simulate with button in UI)
  # For all customers with risk_score > 70:
  #   - Generate recommendations
  #   - Create intervention record
  #   - Simulate sending notification
  ```

**Day 6 Afternoon (4 hours):**
- **TASK 3.7:** Message personalization
  ```python
  def personalize_message(customer, recommendation):
      template = """
      Hi {name},
      
      We noticed you might need some flexibility. 
      We can offer you: {recommendation}
      
      No penalties. Reply YES to accept.
      
      - Your Bank
      """
      return template.format(
          name=customer.name,
          recommendation=recommendation['message']
      )
  ```

- **TASK 3.8:** Intervention analytics
  - Total interventions sent
  - Interventions by type
  - Mock acceptance rate (random simulation)
  - Cost savings calculator

**MVP 3 Demo:**
- Show customer with risk score 85
- Click "Generate Recommendations"
- Show 2-3 intervention options
- Click "Send SMS"
- Show notification log with timestamp
- Show intervention summary dashboard

**Deliverables:**
- ✅ Recommendation engine
- ✅ Interventions database & API
- ✅ Mock notification service
- ✅ Streamlit intervention page
- ✅ Automated intervention workflow
- ✅ Intervention analytics

---

## **PHASE 4: Polish + Demo Prep (Day 7)**
### **Final MVP: "Production-ready demo"**

**Goal:** Documentation, testing, demo preparation

**Day 7 Morning (4 hours):**
- **TASK 4.1:** README.md
  ```markdown
  # Pre-Delinquency Intervention Engine
  
  ## Quick Start
  ```bash
  docker-compose up -d
  python src/data_generation/generate_synthetic_data.py
  python src/models/train_model.py
  ```
  
  ## Access
  - Dashboard: http://localhost:8501
  - API Docs: http://localhost:8000/docs
  
  ## Architecture
  [Include diagram]
  ```

- **TASK 4.2:** Code cleanup
  - Add docstrings
  - Format with Black
  - Remove debug print statements
  - Add error handling

**Day 7 Afternoon (4 hours):**
- **TASK 4.3:** Demo scenario script
  ```
  1. Show dashboard overview (pie chart, metrics)
  2. Navigate to customer "CUST_1234" 
  3. Explain risk score = 82 (RED tier)
  4. Show SHAP explanation (salary delay = +20 points)
  5. Click "Generate Recommendations"
  6. Show 3 intervention options
  7. Click "Send SMS"
  8. Show intervention sent successfully
  9. Show analytics: "Prevented 150 defaults this week"
  ```

- **TASK 4.4:** Presentation slides (10-12 slides)
  - Problem statement
  - Solution approach
  - Architecture diagram
  - Live demo
  - Impact metrics (simulated)
  - Technical stack
  - Future roadmap

**Day 7 Final Hour:**
- **TASK 4.5:** Practice demo (3x run-throughs)
- **TASK 4.6:** Backup plan if WiFi fails
  - Record screen recording
  - Take screenshots
  - Export demo data

**Final Deliverables:**
- ✅ Complete README with setup instructions
- ✅ Demo script (step-by-step)
- ✅ Presentation slides
- ✅ Video recording (backup)
- ✅ GitHub repository (public)

---

## Data Model

### PostgreSQL Schema

```sql
-- Customers
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    segment VARCHAR(50),  -- RETAIL, PREMIUM, CORPORATE
    created_at TIMESTAMP DEFAULT NOW()
);

-- Accounts
CREATE TABLE accounts (
    account_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    account_type VARCHAR(50),  -- SAVINGS, CHECKING, LOAN, CREDIT_CARD
    balance NUMERIC(12, 2),
    credit_limit NUMERIC(12, 2),
    status VARCHAR(20),
    opened_date DATE
);

-- Transactions
CREATE TABLE transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    account_id VARCHAR(50) REFERENCES accounts(account_id),
    amount NUMERIC(12, 2),
    transaction_type VARCHAR(50),  -- CREDIT, DEBIT
    category VARCHAR(50),  -- SALARY, BILL_PAYMENT, SHOPPING, etc.
    merchant VARCHAR(100),
    timestamp TIMESTAMP,
    description TEXT
);

-- Customer Features (Engineered)
CREATE TABLE customer_features (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    
    -- Financial Stress Indicators
    salary_delay_days INTEGER,
    savings_decline_pct_7d NUMERIC(5, 2),
    savings_balance NUMERIC(12, 2),
    
    -- Payment Behavior
    failed_autodebits_count INTEGER,
    utility_payment_delay_days INTEGER,
    
    -- Spending Patterns
    total_spend_30d NUMERIC(12, 2),
    discretionary_spending_ratio NUMERIC(5, 4),
    
    -- Credit Indicators
    credit_utilization_ratio NUMERIC(5, 4),
    
    computed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(customer_id, computed_at)
);

-- Risk Scores
CREATE TABLE risk_scores (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    risk_score NUMERIC(5, 2),  -- 0-100
    risk_tier VARCHAR(10),     -- GREEN, YELLOW, ORANGE, RED
    default_probability NUMERIC(5, 4),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Interventions
CREATE TABLE interventions (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    risk_score_id INTEGER REFERENCES risk_scores(id),
    intervention_type VARCHAR(50),  -- PAYMENT_HOLIDAY, EMI_REDUCTION, etc.
    status VARCHAR(20),  -- PENDING, SENT, ACCEPTED, REJECTED
    channel VARCHAR(20),  -- SMS, EMAIL, APP
    message_text TEXT,
    sent_at TIMESTAMP,
    responded_at TIMESTAMP,
    outcome VARCHAR(50)
);
```

---

## ML Strategy (Simplified)

### Model: XGBoost Classifier

**Why XGBoost for hackathon:**
- Fast training (minutes, not hours)
- Works well with tabular data
- Built-in feature importance
- Handles missing values
- No need for extensive feature scaling

**Training Process:**

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap

# 1. Load data
df = pd.read_sql("SELECT * FROM customer_features", conn)
X = df.drop(['customer_id', 'default_28d'], axis=1)
y = df['default_28d']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Train
model = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=19  # Since ~5% default rate
)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print(classification_report(y_test, y_pred))

# 5. Explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 6. Save
import joblib
joblib.dump(model, 'models/xgboost_model.joblib')
joblib.dump(explainer, 'models/shap_explainer.joblib')
```

**Key Features (15 total):**

1. `salary_delay_days` - Days past expected salary credit
2. `savings_decline_pct_7d` - % savings drop in last 7 days
3. `savings_balance` - Current savings balance
4. `failed_autodebits_count` - Failed payment attempts (30d)
5. `utility_payment_delay_days` - Days late on utilities
6. `total_spend_30d` - Total spending (30 days)
7. `discretionary_spending_ratio` - Non-essential / total spend
8. `credit_utilization_ratio` - Credit used / limit
9. `avg_transaction_amount` - Average transaction size
10. `transaction_frequency_30d` - Number of transactions
11. `atm_withdrawal_count_7d` - Cash withdrawals
12. `minimum_balance_days` - Days below minimum balance
13. `income_stability_score` - Regularity of income
14. `payment_history_score` - Past payment behavior
15. `account_age_months` - How long customer has been with bank

---

## Demo Scenarios

### Scenario 1: High-Risk Customer Detection

**Customer:** Sarah Martinez (CUST_5432)

**Background:**
- Retail customer, 3 years with bank
- Usually regular salary credit on 1st of month
- Savings balance: $2,500

**Recent Changes:**
- Salary delayed by 12 days
- Savings dropped from $2,500 → $800 (68% decline)
- 2 failed auto-debit attempts
- Increased UPI transactions to lending apps

**System Behavior:**
1. Daily batch scoring detects risk score: **86** (RED tier)
2. Dashboard shows Sarah in "Urgent Attention" list
3. Feature importance shows:
   - Salary delay: +25 points
   - Savings decline: +20 points
   - Failed autodebits: +15 points
4. System recommends:
   - 1-month payment holiday
   - EMI reduction (25% for 6 months)
5. Mock SMS sent: "Hi Sarah, we noticed you might need flexibility..."

**Demo Points:**
- Show early detection (before missing payment)
- Show explainable AI (why risk score is high)
- Show empathetic intervention

---

### Scenario 2: False Positive Avoidance

**Customer:** David Chen (CUST_2341)

**Background:**
- Premium customer
- Large vacation planned (pre-authorized)
- High spending this month on travel

**System Behavior:**
1. Risk score: **45** (YELLOW tier - monitor only)
2. System correctly identifies:
   - High spending is travel-related
   - Savings decline is temporary
   - Payment history excellent
3. No intervention triggered
4. Automatic re-evaluation in 7 days

**Demo Points:**
- Show model understands context
- Low false positive rate
- Not all spending changes = risk

---

## Setup Instructions

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- 8GB RAM, 10GB disk space

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/pre-delinquency-engine.git
cd pre-delinquency-engine

# 2. Start services
docker-compose up -d

# 3. Generate synthetic data
python src/data_generation/generate_synthetic_data.py

# 4. Train model
python src/models/train_model.py

# 5. Run batch scoring
curl -X POST http://localhost:8000/batch-score

# 6. Access dashboard
open http://localhost:8501

# 7. Access API docs
open http://localhost:8000/docs
```

### Environment Variables

Create `.env` file:

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=delinquency_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Model
MODEL_PATH=./data/models/xgboost_model.joblib
MODEL_VERSION=v1.0

# API
API_HOST=0.0.0.0
API_PORT=8000
```

### Directory Structure After Setup

```
pre-delinquency-engine/
├── data/
│   ├── raw/
│   │   ├── customers.csv
│   │   ├── accounts.csv
│   │   └── transactions.csv
│   ├── processed/
│   │   └── customer_features.csv
│   └── models/
│       ├── xgboost_model.joblib
│       └── shap_explainer.joblib
│
├── logs/
│   ├── api.log
│   └── interventions.log
│
└── postgres_data/  (Docker volume)
```

---

## Hackathon Judging Criteria Alignment

### Innovation (25%)
- ✅ Novel application of ML for pre-delinquency detection
- ✅ Proactive vs. reactive approach
- ✅ Explainable AI with SHAP

### Technical Implementation (25%)
- ✅ Full-stack solution (data, ML, API, UI)
- ✅ Docker-ized for easy deployment
- ✅ Clean code with documentation
- ✅ Open-source stack

### Business Impact (25%)
- ✅ Clear value proposition (reduce losses, improve customer relationships)
- ✅ Measurable metrics (delinquency reduction)
- ✅ Scalable solution

### Presentation (25%)
- ✅ Clear demo flow
- ✅ Real-world scenarios
- ✅ Visual dashboards
- ✅ Live system demonstration

---

## Future Enhancements (Post-Hackathon)

1. **Real-time Streaming**
   - Add Kafka for real-time transaction processing
   - Update risk scores on every transaction

2. **Advanced ML Models**
   - LSTM for sequential patterns
   - Graph Neural Networks for relationship networks

3. **A/B Testing Framework**
   - Test different intervention strategies
   - Optimize message content

4. **Mobile App**
   - Customer-facing self-service portal
   - Accept interventions via app

5. **Multi-channel Notifications**
   - Integrate real Twilio for SMS
   - SendGrid for emails
   - WhatsApp Business API

---

**Hackathon Team:**
- Data Scientist (1): Data generation, EDA, model training
- Backend Engineer (1): FastAPI, database, API design
- Full-Stack Engineer (1): Streamlit dashboard, integration
- (Optional) DevOps (1): Docker, deployment, monitoring

**Good luck! 🚀**
