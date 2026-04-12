# Pre-Delinquency Intervention Engine

Predict financial delinquency before it occurs — using behavioral feature engineering, gradient boosting, and SHAP-based explainability on a synthetic multi-source credit dataset.

---

## Problem

By the time a borrower becomes delinquent, the cost of intervention has already increased significantly. Most risk systems flag accounts reactively — after a missed payment — leaving little room for low-cost outreach. This project targets the 30–60 day window before default risk materializes, using behavioral signals (transaction patterns, app activity, support interactions) rather than static credit attributes.

---

## Approach

The pipeline runs end-to-end from raw simulated logs to an interpretable classifier.

**1. Data Simulation & Storage**
- Simulated multi-source behavioral logs: transaction history (~160M+ rows), app interaction events, and support tickets
- Aggregated into a SQLite-backed feature store of 5,000 customer records
- Label generation used probabilistic default modeling — not deterministic rules — to produce realistic class distributions

**2. Feature Engineering**
- Time-windowed features constructed using past-only data with an explicit prediction gap to prevent leakage
- Trend-based signals: balance decline rate, transaction volume shifts, login frequency change
- Rolling aggregates: payment velocity, utilization trends, behavioral consistency scores

**3. Baseline vs. Final Model**
- Baseline: Logistic Regression (scikit-learn) — ROC-AUC ≈ 0.65
- Final: LightGBM — ROC-AUC ≈ 0.72–0.75
- Improvement attributed to LightGBM's capacity to capture non-linear relationships and feature interactions that linear models cannot represent

**4. Explainability**
- SHAP values computed per prediction to surface individual-level risk drivers
- Summary and waterfall plots used to validate that model behavior aligns with domain expectations
- Designed to support the kind of reasoning required in compliance or audit contexts

---

## Results

| Metric | Value |
|---|---|
| ROC-AUC (LightGBM) | 0.72 – 0.75 |
| ROC-AUC (Logistic Regression baseline) | ~0.65 |
| Accuracy | 82 – 86% |
| F1-Score (default class) | 0.45 – 0.55 |
| Log Loss | 0.38 – 0.45 |

> Class imbalance is present. Accuracy is reported for completeness but ROC-AUC and F1 on the minority class are the operative metrics. Evaluation performed on a held-out test set (80/20 split).

**Dataset**: Synthetic Credit Risk Behavioral Dataset — 5,000 customers, 8–10 engineered features. Generated from simulated transaction history, app interaction logs, and support ticket data aggregated via SQLite.

---

## Project Structure

```
Pre-Delinquency-Intervention-Engine/
│
├── data_generation/
│   ├── setup_database.py        # Creates and initializes dataset
│   ├── split_data.py            # Train/validation/test split
│   ├── verify_data.py           # Data validation checks
│   │
│   ├── train_model.py           # Main training pipeline
│   ├── train_xgboost.py         # XGBoost model training
│   │
│   ├── transactions.csv         # Raw dataset
│   ├── train_data.csv           # Training data
│   ├── val_data.csv             # Validation data
│   ├── test_data.csv            # Test data
│   ├── support_logs.csv         # Auxiliary logs
│   │
│   ├── scaler.pkl               # Data scaler
│   ├── xgboost_model.pkl        # Trained model
│   ├── shap_explainer.pkl       # SHAP explainer
│   │
│   ├── shap_summary_delinquency.png   # SHAP visualization
│   └── threshold.txt            # Decision threshold
│
├── shap_outputs/
│   ├── shap_summary_bar.png
│   └── shap_summary_beeswarm.png
│
├── lgbm.py                      # LightGBM model experimentation
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
├── DESIGN.md                    # System design notes
├── IMPLEMENTATION_TASKS.md      # Development tracking
├── LICENSE
│
├── venv/                        # Virtual environment (ignored)
├── .gitignore
└── .gitattributes

```

---

## How to Run

**Requirements**: Python 3.9+

```bash
# Clone the repository
git clone https://github.com/[YOUR_USERNAME]/pre-delinquency-intervention.git
cd pre-delinquency-intervention

# Install dependencies
pip install -r requirements.txt

# Simulate behavioral data and build feature store
python src/simulate.py
python src/features.py

# Train baseline and LightGBM models
python src/model.py

# Generate SHAP explanations
python src/explain.py
```

To run interactively, execute notebooks in order (01 → 04):

```bash
jupyter notebook notebooks/
```

---

## Limitations

- **Synthetic data ceiling**: The dataset is generated, not sourced from a live portfolio. Real behavioral logs carry distributional complexity — temporal drift, population shifts, missingness patterns — that this setup does not replicate.
- **Label realism**: Probabilistic label generation approximates default behavior but may not reflect actual DPD (Days Past Due) thresholds used operationally.
- **Feature set scope**: 8–10 engineered features is compact. Production models typically incorporate bureau-sourced attributes, product-level signals, and macroeconomic covariates.
- **No deployment layer**: The pipeline is batch-oriented with no real-time scoring endpoint, model registry, or drift monitoring.
- **Threshold not calibrated to a business objective**: Operating threshold was not tuned against a cost-of-intervention vs. cost-of-default loss function.

---

## Future Work

- Replace simulated data with a public real-world credit dataset (e.g., Lending Club, Home Credit) for external validity
- Implement walk-forward cross-validation to properly evaluate time-series generalization
- Add probability calibration (isotonic regression or Platt scaling) for reliable risk scores
- Build a monitoring module to track feature and score distribution drift over time
- Explore survival analysis framing to model *time-to-delinquency* rather than binary outcome
- Expose a lightweight scoring API (FastAPI) for integration testing

---

## Stack

`Python` `LightGBM` `scikit-learn` `SHAP` `pandas` `NumPy` `SQLite` `Jupyter`

---

*Built as a student project with focus on real-world applicability in consumer credit risk.*
