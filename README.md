# Pre-Delinquency Intervention Engine

Identify financially at-risk accounts before they miss a payment — using anomaly detection, gradient boosting, and model explainability.

---

## Problem

By the time a borrower becomes delinquent, the cost of intervention has already increased significantly. Most risk systems flag accounts reactively — after a missed payment — leaving little room for low-cost outreach. This project addresses the gap between standard credit monitoring and early-stage financial distress, targeting the 30–60 day window before default risk materializes.

---

## Approach

The pipeline runs in three stages:

**1. Feature Engineering**
- Derived behavioral indicators: payment velocity, utilization trend, rolling balance ratios
- Lag features capturing month-over-month shifts in spending and repayment patterns
- Encoded categorical risk tiers and account-level history signals

**2. Anomaly Detection**
- Isolation Forest used as an unsupervised pre-screen to flag structurally unusual accounts
- Anomaly scores passed downstream as engineered features

**3. Classification**
- XGBoost trained on labeled delinquency outcomes (30+ DPD as positive class)
- Hyperparameters tuned via cross-validated grid search
- Class imbalance handled with `scale_pos_weight`

**4. Explainability**
- SHAP values computed per prediction for individual-level reasoning
- Summary and waterfall plots used to identify top drivers across the population
- Designed to support compliance-style explanations for model decisions

---

## Results

| Metric | Value |
|---|---|
| ROC-AUC | [ADD METRIC HERE] |
| Precision (at threshold) | [ADD METRIC HERE] |
| Recall (at threshold) | [ADD METRIC HERE] |
| F1 Score | [ADD METRIC HERE] |
| False Positive Rate | [ADD METRIC HERE] |

> Threshold was selected to optimize recall while maintaining acceptable precision for business use. Evaluation performed on a held-out test set ([ADD %] split).

**Dataset**: [Synthetic / Real — ADD SOURCE OR DESCRIPTION HERE]. [ADD row count] records, [ADD feature count] features.

---

## Project Structure

```
pre-delinquency-intervention/
├── data/
│   ├── raw/                    # Source data (not tracked)
│   └── processed/              # Cleaned and feature-engineered datasets
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_shap_explainability.ipynb
├── src/
│   ├── features.py             # Feature construction logic
│   ├── model.py                # Training and evaluation
│   ├── anomaly.py              # Isolation Forest wrapper
│   └── explain.py              # SHAP utilities
├── outputs/
│   ├── models/                 # Serialized model artifacts
│   └── plots/                  # SHAP and evaluation figures
├── requirements.txt
└── README.md
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

# Run feature engineering
python src/features.py --input data/raw/accounts.csv --output data/processed/

# Train models
python src/model.py --data data/processed/features.csv --output outputs/models/

# Generate SHAP explanations
python src/explain.py --model outputs/models/xgb_model.pkl --data data/processed/features.csv
```

To run the full pipeline end-to-end via notebooks, execute them in order (01 → 04) using Jupyter:

```bash
jupyter notebook notebooks/
```

---

## Limitations

- **Label quality**: Delinquency labels depend on DPD cutoffs which may not uniformly capture distress across account types.
- **Temporal leakage risk**: Care was taken to prevent future data from entering training features, but this has not been formally audited with a time-series split.
- **No production infrastructure**: The pipeline is offline batch-oriented. There is no real-time scoring endpoint or monitoring layer.
- **Dataset scope**: [If synthetic: The synthetic dataset may not reflect the full distributional complexity of live portfolio data.] [If real: The dataset is limited to a single institution's portfolio and may not generalize.]
- **Threshold sensitivity**: Model performance is sensitive to operating threshold selection. Business context is needed to calibrate this appropriately.

---

## Future Work

- Implement time-series aware cross-validation (walk-forward or expanding window)
- Add calibration layer (Platt scaling or isotonic regression) for reliable probability outputs
- Incorporate alternative data signals: transaction-level features, external bureau attributes
- Build a monitoring module to track score distribution drift over time
- Explore survival analysis framing to predict *time-to-delinquency* rather than binary outcome

---

## Stack

`Python` `XGBoost` `scikit-learn` `SHAP` `pandas` `matplotlib`

---

*Built as a student project with focus on real-world applicability in consumer credit risk.*
