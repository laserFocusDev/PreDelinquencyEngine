# Pre-Delinquency Intervention Engine

> **Explainable Machine Learning for Early Credit Risk Detection**

Predict financial delinquency **before** missed payments occur using behavioral feature engineering, LightGBM, and SHAP explainability on a synthetic multi-source credit dataset.

---

## Overview

Traditional credit risk systems identify borrowers only after delinquency has already begun. This project explores whether behavioral signals can identify financially vulnerable customers **30–60 days before default risk materializes**, enabling proactive intervention.

Rather than relying solely on static credit attributes, the model learns from simulated behavioral data including transaction history, digital banking activity, and customer support interactions.

The project implements an end-to-end machine learning pipeline including:

- Behavioral data simulation
- Feature engineering with leakage prevention
- LightGBM-based credit risk prediction
- Hyperparameter optimization
- SHAP explainability
- Business-oriented evaluation metrics

---

## Problem Statement

Financial institutions often intervene only after a borrower misses a payment, when recovery costs are already increasing.

The objective of this project is to identify customers entering financial stress before delinquency occurs by learning behavioral patterns that precede default.

Instead of predicting whether a customer has already defaulted, the model estimates future delinquency risk from recent customer behavior.

---

# Pipeline

```
Raw Behavioural Logs
│
├── Transaction History
├── Banking App Activity
└── Support Interactions
        │
        ▼
SQLite Feature Store
        │
        ▼
Feature Engineering
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Baseline LightGBM
        │
        ▼
Hyperparameter Optimization
(RandomizedSearchCV)
        │
        ▼
Final LightGBM Model
        │
        ▼
Model Evaluation
        │
        ▼
SHAP Explainability
```

---

# Dataset

This project uses a synthetic multi-source behavioral credit dataset.

### Sources

- Customer transaction history
- Banking application activity
- Customer support logs

The raw logs are aggregated into customer-level features stored using SQLite.

Dataset characteristics:

- ~5,000 customers
- ~160 million simulated transaction records
- Binary classification
- Moderate class imbalance
- Time-aware feature engineering

---

# Feature Engineering

The model uses behavioral indicators rather than traditional demographic variables.

Example engineered features include:

| Feature | Description |
|----------|-------------|
| Balance Decline Rate | Rate of decrease in account balance |
| Transaction Volume Trend | Change in spending behaviour |
| Payment Velocity | Time between repayments |
| Login Frequency Change | Banking application engagement |
| Behavioral Consistency Score | Stability of customer behaviour |
| Credit Utilization Trend | Utilization changes over time |

All features are constructed using historical observations only to prevent target leakage.

---

# Model Development

## Baseline

- LightGBM
- Early stopping
- Class imbalance handling
- Native LightGBM API

## Hyperparameter Optimization

RandomizedSearchCV is used with Stratified K-Fold cross validation.

Optimized parameters include:

- Number of leaves
- Maximum depth
- Learning rate
- Feature fraction
- Bagging fraction
- Regularization
- Minimum child samples

Class imbalance is handled using:

```
scale_pos_weight = Negative Samples / Positive Samples
```

---

# Explainability

Model predictions are interpreted using **SHAP (SHapley Additive Explanations).**

Generated outputs include:

- SHAP Beeswarm Plot
- SHAP Feature Importance Plot
- Ranked Feature Importance Table

These explanations improve transparency and make the model easier to validate in regulated financial environments.

---

# Evaluation

The project evaluates both statistical and business-oriented metrics.

| Metric | Description |
|---------|-------------|
| ROC-AUC | Overall ranking performance |
| PR-AUC | Performance on imbalanced data |
| Recall@Top10% | Percentage of defaults captured within highest-risk customers |
| Classification Report | Precision, Recall and F1 |

The pipeline also reports:

- Train AUC
- Test AUC
- Overfitting Gap
- Mean SHAP importance

---

# Repository Structure

```
PreDelinquencyEngine/

│
├── backend/
│
├── data_generation/
│   ├── setup_database.py
│   ├── split_data.py
│   ├── verify_data.py
│   ├── train_model.py
│   ├── train_xgboost.py
│   ├── train_data.csv
│   ├── val_data.csv
│   ├── test_data.csv
│   └── support_logs.csv
│
├── shap_outputs/
│   ├── shap_summary_bar.png
│   └── shap_summary_beeswarm.png
│
├── lgbm.py
├── DESIGN.md
├── IMPLEMENTATION_TASKS.md
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/laserFocusDev/PreDelinquencyEngine.git

cd PreDelinquencyEngine
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Run the complete pipeline

```bash
python lgbm.py
```

The pipeline automatically performs:

- Data loading
- Class imbalance calculation
- Baseline LightGBM training
- Hyperparameter optimization
- Model evaluation
- Model serialization
- SHAP analysis

Generated outputs include:

```
lgbm_credit_risk_model.pkl

shap_outputs/
    shap_summary_bar.png
    shap_summary_beeswarm.png
```

---

# Technologies

| Category | Technologies |
|-----------|--------------|
| Language | Python |
| Machine Learning | LightGBM, Scikit-learn |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |
| Hyperparameter Tuning | RandomizedSearchCV |
| Storage | SQLite |
| Visualization | Matplotlib |

---

# Limitations

- Uses synthetic behavioral data rather than production financial records.
- Binary labels are probabilistically generated.
- Feature space is intentionally compact.
- No deployment or model monitoring layer.
- Thresholds are not optimized against business costs.

---

# Future Improvements

- Train on Lending Club or Home Credit datasets.
- Implement time-series cross validation.
- Probability calibration using Platt Scaling or Isotonic Regression.
- Add FastAPI inference service.
- Add MLflow experiment tracking.
- Build model drift monitoring.
- Deploy interactive dashboard.

---

# Learning Outcomes

This project demonstrates:

- End-to-end machine learning pipeline design
- Credit risk modelling
- Feature engineering
- Class imbalance handling
- Hyperparameter optimization
- Explainable AI using SHAP
- Business-aware evaluation metrics
- Financial machine learning workflows

---

## License

Released under the MIT License.
