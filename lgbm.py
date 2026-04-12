import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform

warnings.filterwarnings("ignore", category=UserWarning)
shap.initjs()  # required for notebook environments; harmless in scripts

PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# 1. Configuration – single place to change paths / hyper-ranges / thresholds
# -----------------------------------------------------------------------------
CFG = {
    # Data
    "train_path":  "data_generation/train_data.csv",
    "val_path":    "data_generation/val_data.csv",
    "test_path":   "data_generation/test_data.csv",
    "target_col":  "is_delinquent",

    # Model serialisation
    "model_output": "lgbm_credit_risk_model.pkl",

    # SHAP output directory
    "shap_output_dir": "shap_outputs",

    # Early stopping
    "early_stopping_rounds": 100,
    "num_boost_round":       2000,      # upper bound; early stopping cuts this down

    # Business metric
    "top_k_pct": 0.10,                  # Recall@Top10%

    # Hyperparameter search
    "n_iter_search": 40,                # number of random configs to try
    "cv_folds":      5,
    "search_seed":   42,

    # Reproducibility
    "seed": 42,
}


# -----------------------------------------------------------------------------
# 2. Data Loading
# -----------------------------------------------------------------------------
def load_data(cfg: dict) -> tuple[pd.DataFrame, ...]:
    print("-- Loading data -------------------------------------------------")

    train_path = PROJECT_ROOT / cfg["train_path"]
    val_path   = PROJECT_ROOT / cfg["val_path"]
    test_path  = PROJECT_ROOT / cfg["test_path"]

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            print(f"Expected data at: {path}")
            print("Check your project structure")
            raise FileNotFoundError(f"Data file not found: {path}")

    print(f"Train path: {train_path}")
    print(f"Val path  : {val_path}")
    print(f"Test path : {test_path}")

    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)
    test  = pd.read_csv(test_path)

    target = cfg["target_col"]

    # Ensure target is binary int (handle old string mapping if needed)
    if train[target].dtype == object:
        train[target] = (train[target] == "HIGH").astype(int)
        val[target]   = (val[target] == "HIGH").astype(int)
        test[target]  = (test[target] == "HIGH").astype(int)

    # 🔥 Drop useless ID column if present
    drop_cols = [target]
    if "customer_id" in train.columns:
        drop_cols.append("customer_id")

    X_train = train.drop(columns=drop_cols)
    y_train = train[target]

    X_val = val.drop(columns=drop_cols)
    y_val = val[target]

    X_test = test.drop(columns=drop_cols)
    y_test = test[target]

    feature_cols = X_train.columns.tolist()

    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"{name:>5} rows={len(y):>7,} default_rate={y.mean():.2%}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

# -----------------------------------------------------------------------------
# 3. Class-imbalance Calculation
# -----------------------------------------------------------------------------
def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    LightGBM's `scale_pos_weight` mirrors XGBoost's convention:
        scale_pos_weight = n_negatives / n_positives

    In credit risk, positives (defaults) are rare (often 1–5 %).
    A high ratio tells the model to up-weight each default observation,
    which substantially improves recall without any data resampling.
    """
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    ratio = n_neg / n_pos
    print(f"\n  scale_pos_weight = {ratio:.2f}  "
          f"(neg={n_neg:,} / pos={n_pos:,})")
    return ratio


# -----------------------------------------------------------------------------
# 4. Baseline Model Training (with early stopping)
# -----------------------------------------------------------------------------
def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    scale_pos_weight: float,
    cfg: dict,
) -> lgb.Booster:
    """
    Train a well-configured LightGBM model using the native API so we can
    leverage `callbacks` for early stopping and verbose evaluation.

    Key parameter choices for credit risk:
      - objective='binary'          → cross-entropy loss
      - metric='auc'                → threshold-independent; essential for imbalanced
      - scale_pos_weight            → class rebalancing without synthetic samples
      - min_child_samples=50        → prevents leaves that fit only 1-2 defaults
      - reg_alpha / reg_lambda      → L1/L2 to reduce overfitting on sparse financial features
      - subsample + colsample_bytree → stochastic boosting reduces variance
    """
    print("\n-- Training baseline LightGBM -----------------------------------")

    params = {
        "objective":          "binary",
        "metric":             ["auc", "binary_logloss"],
        "boosting_type":      "gbdt",
        "learning_rate":      0.05,
        "num_leaves":         63,           # 2^6-1; keep < 2^(max_depth) rule
        "max_depth":          -1,           # unconstrained; leaves govern complexity
        "min_child_samples":  50,           # critical for rare-event classes
        "feature_fraction":   0.8,          # colsample_bytree equivalent
        "bagging_fraction":   0.8,          # subsample equivalent
        "bagging_freq":       5,
        "reg_alpha":          0.1,          # L1 regularisation
        "reg_lambda":         1.0,          # L2 regularisation
        "scale_pos_weight":   scale_pos_weight,
        "is_unbalance":       False,        # we handle imbalance via scale_pos_weight
        "n_jobs":             -1,
        "seed":               cfg["seed"],
        "verbose":            -1,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(
            stopping_rounds=cfg["early_stopping_rounds"],
            verbose=True,
        ),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=cfg["num_boost_round"],
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    print(f"\n  Best iteration : {model.best_iteration}")
    print(f"  Best val AUC   : {model.best_score['val']['auc']:.5f}")
    return model


# -----------------------------------------------------------------------------
# 5. Hyperparameter Tuning via RandomizedSearchCV
# -----------------------------------------------------------------------------
def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    cfg: dict,
) -> lgb.LGBMClassifier:
    """
    RandomizedSearchCV over key LightGBM hyperparameters.

    -- Why these parameters matter for credit risk ---------------------------

    num_leaves / max_depth:
        Controls model complexity.  Too many leaves → over-fits sparse defaults.
        Typical sweet spot: 31–127 leaves for tabular financial data.

    learning_rate × n_estimators (early stopping):
        Smaller LR + more trees = smoother loss surface, less variance.
        0.01–0.05 paired with early stopping is the standard credit-risk recipe.

    min_child_samples:
        Minimum samples per leaf.  Because defaults are rare, a leaf with 5
        samples is almost certainly noise.  50–200 is typical.

    feature_fraction / bagging_fraction:
        Both reduce correlation between trees → lower variance on imbalanced data.

    reg_alpha (L1):
        Encourages feature sparsity — useful when many engineered features may be
        collinear (e.g., multiple utilisation ratios).

    reg_lambda (L2):
        Shrinks leaf weights; safer default regulariser.

    scale_pos_weight:
        We fix this from the data rather than searching, because the optimal
        value is mathematically determined by the class ratio.
    -------------------------------------------------------------------------
    """
    print("\n-- Hyperparameter Search (RandomizedSearchCV) -------------------")

    # Use the sklearn API wrapper for compatibility with sklearn CV utilities
    base_estimator = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        scale_pos_weight=scale_pos_weight,
        is_unbalance=False,
        n_jobs=-1,
        random_state=cfg["seed"],
        verbose=-1,
        # Fixed early stopping via n_estimators; CV does not support callbacks
        n_estimators=500,
    )

    param_dist = {
        "num_leaves":          randint(31, 150),
        "max_depth":           [-1, 5, 6, 7, 8, 10],
        "learning_rate":       uniform(0.01, 0.09),      # [0.01, 0.10]
        "min_child_samples":   randint(30, 200),
        "feature_fraction":    uniform(0.6, 0.4),        # [0.6, 1.0]
        "bagging_fraction":    uniform(0.6, 0.4),        # [0.6, 1.0]
        "bagging_freq":        [3, 5, 7],
        "reg_alpha":           uniform(0.0, 0.5),
        "reg_lambda":          uniform(0.5, 4.5),        # [0.5, 5.0]
        "min_split_gain":      uniform(0.0, 0.1),
    }

    cv = StratifiedKFold(
        n_splits=cfg["cv_folds"],
        shuffle=True,
        random_state=cfg["seed"],
    )

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_dist,
        n_iter=cfg["n_iter_search"],
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=cfg["search_seed"],
        refit=True,
    )

    search.fit(X_train, y_train)

    print(f"\n  Best CV ROC-AUC : {search.best_score_:.5f}")
    print("  Best params:")
    for k, v in sorted(search.best_params_.items()):
        print(f"    {k:<25} = {v}")

    return search.best_estimator_


# -----------------------------------------------------------------------------
# 6. Evaluation Metrics
# -----------------------------------------------------------------------------
def recall_at_top_k(y_true: np.ndarray,
                    y_scores: np.ndarray,
                    k: float = 0.10) -> float:
    """
    Recall@Top-K%  (business-critical for credit scoring):

    In practice, risk teams review only the top-K% most suspicious accounts.
    This metric answers: "Of all actual defaulters, what fraction would we
    have flagged if we inspected the top-10% highest-risk applicants?"

    A model with high ROC-AUC but low Recall@Top10% may be ranking defaults
    poorly at the very top of the score distribution — exactly where the
    scorecard cuts matter most.
    """
    n_top = max(1, int(len(y_true) * k))
    top_indices = np.argsort(y_scores)[::-1][:n_top]
    top_labels  = y_true[top_indices]
    recall      = top_labels.sum() / max(1, y_true.sum())
    return float(recall)


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    cfg: dict,
    label: str = "Tuned LightGBM",
) -> dict:
    """
    Full evaluation suite returning a results dict.
    Uses predict_proba for sklearn API or predict for native Booster.
    """
    # -- Predictions ----------------------------------------------------------
    if isinstance(model, lgb.Booster):
        train_scores = model.predict(X_train)
        test_scores  = model.predict(X_test)
    else:
        train_scores = model.predict_proba(X_train)[:, 1]
        test_scores  = model.predict_proba(X_test)[:, 1]

    train_auc  = roc_auc_score(y_train, train_scores)
    test_auc   = roc_auc_score(y_test,  test_scores)
    pr_auc     = average_precision_score(y_test, test_scores)
    recall_k   = recall_at_top_k(
        y_test.values, test_scores, k=cfg["top_k_pct"]
    )

    # -- Overfitting delta -----------------------------------------------------
    # A train-test AUC gap > 0.05 is a red flag in credit risk models;
    # regulators and model validators scrutinise this closely.
    auc_gap    = train_auc - test_auc

    results = {
        "model":          label,
        "train_auc":      train_auc,
        "test_auc":       test_auc,
        "auc_gap":        auc_gap,
        "pr_auc":         pr_auc,
        "recall_top10pct": recall_k,
    }
    return results


# -----------------------------------------------------------------------------
# 7. Print Summary Table
# -----------------------------------------------------------------------------
def print_summary(results_list: list[dict]) -> None:
    """Pretty-print a comparison table of all evaluated models."""
    print("\n" + "=" * 80)
    print("  EVALUATION SUMMARY")
    print("=" * 80)
    header = (
        f"{'Model':<28} {'Train AUC':>10} {'Test AUC':>10} "
        f"{'Gap':>8} {'PR-AUC':>9} {'Rec@Top10%':>12}"
    )
    print(header)
    print("-" * 80)
    for r in results_list:
        gap_flag = " ⚠" if r["auc_gap"] > 0.05 else "  "
        print(
            f"{r['model']:<28} "
            f"{r['train_auc']:>10.4f} "
            f"{r['test_auc']:>10.4f} "
            f"{r['auc_gap']:>7.4f}{gap_flag} "
            f"{r['pr_auc']:>9.4f} "
            f"{r['recall_top10pct']:>11.2%}"
        )
    print("=" * 80)
    print("  ⚠  AUC gap > 0.05 may indicate overfitting — review regularisation.")
    print()


# -----------------------------------------------------------------------------
# 8. Model Serialisation
# -----------------------------------------------------------------------------
def save_model(model, path: str) -> None:
    """Persist model to disk via pickle.  LightGBM's own .txt format is also
    available via model.save_model() for the native Booster — pickle is used
    here for consistency across both native and sklearn API models."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"\n  Model saved → {path}  ({size_mb:.2f} MB)")


def load_model(path: str):
    """Utility: load a previously serialised model."""
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# 9. SHAP Interpretability
# -----------------------------------------------------------------------------
def run_shap_analysis(
    model,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    SHAP (SHapley Additive exPlanations) provides model-agnostic,
    theoretically grounded feature attribution.

    In credit risk / model governance:
      - Regulators (e.g. SR 11-7, EBA/ECB model risk guidelines) require
        explainable model decisions.
      - SHAP beeswarm plots reveal *direction* of impact (not just magnitude).
      - Top-feature lists feed into model documentation and challenger reviews.

    We use TreeExplainer which is exact (not approximate) for tree models and
    runs in O(TLD) time where T=trees, L=leaves, D=depth.
    """
    print("\n-- SHAP Analysis ------------------------------------------------")
    output_dir = Path(cfg["shap_output_dir"])
    output_dir.mkdir(exist_ok=True)

    # For sklearn wrapper, extract the underlying Booster
    booster = model.booster_ if hasattr(model, "booster_") else model

    explainer   = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_test)

    # LightGBM binary returns a single array (positive-class SHAP values)
    if isinstance(shap_values, list):
        sv = shap_values[1]   # index 1 → default (positive) class
    else:
        sv = shap_values

    # -- Summary / Beeswarm Plot -----------------------------------------------
    plt.figure(figsize=(12, 8))
    shap.summary_plot(sv, X_test, feature_names=feature_cols, show=False)
    plt.title("SHAP Summary Plot – Default Prediction Model", fontsize=14, pad=14)
    plt.tight_layout()
    beeswarm_path = output_dir / "shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Beeswarm plot saved  → {beeswarm_path}")

    # -- Bar Plot (mean |SHAP|) ------------------------------------------------
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_test, feature_names=feature_cols,
                      plot_type="bar", show=False)
    plt.title("Mean |SHAP| Feature Importance", fontsize=14, pad=14)
    plt.tight_layout()
    bar_path = output_dir / "shap_summary_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Bar plot saved       → {bar_path}")

    # -- Top-N feature table ---------------------------------------------------
    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_df = (
        pd.DataFrame({
            "feature":       feature_cols,
            "mean_abs_shap": mean_abs_shap,
        })
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    importance_df.index += 1   # 1-based rank

    print(f"\n  Top {top_n} Features by Mean |SHAP|:")
    print(f"  {'Rank':>4}  {'Feature':<35} {'Mean |SHAP|':>12}")
    print("  " + "-" * 54)
    for rank, row in importance_df.iterrows():
        print(f"  {rank:>4}  {row['feature']:<35} {row['mean_abs_shap']:>12.5f}")

    return importance_df


# -----------------------------------------------------------------------------
# 10. Main Orchestrator
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  LightGBM Credit-Risk Default Prediction Pipeline")
    print("=" * 80)

    # -- 10.1 Load data --------------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data(CFG)

    # -- 10.2 Class weight -----------------------------------------------------
    print("\n-- Class Imbalance ----------------------------------------------")
    spw = compute_scale_pos_weight(y_train)

    # -- 10.3 Baseline training (native API + early stopping) -----------------
    baseline_model = train_baseline(
        X_train, y_train, X_val, y_val, spw, CFG
    )
    baseline_results = evaluate_model(
        baseline_model, X_train, y_train, X_test, y_test, CFG,
        label="Baseline LightGBM"
    )

    # -- 10.4 Hyperparameter tuning --------------------------------------------
    # Combine train + val for tuning (CV handles validation internally)
    X_tune = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_tune = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    best_model = tune_hyperparameters(X_tune, y_tune, spw, CFG)

    tuned_results = evaluate_model(
        best_model, X_tune, y_tune, X_test, y_test, CFG,
        label="Tuned LightGBM"
    )

    # -- 10.5 Print classification report (tuned model) ------------------------
    print("\n-- Classification Report (Tuned – threshold=0.5) ----------------")
    test_proba  = best_model.predict_proba(X_test)[:, 1]
    test_preds  = (test_proba >= 0.5).astype(int)
    print(classification_report(y_test, test_preds,
                                 target_names=["No Default", "Default"]))

    # -- 10.6 Summary table ----------------------------------------------------
    print_summary([baseline_results, tuned_results])

    # -- 10.7 Save best model --------------------------------------------------
    print("-- Saving Model -------------------------------------------------")
    save_model(best_model, CFG["model_output"])

    # -- 10.8 SHAP interpretability --------------------------------------------
    top_features = run_shap_analysis(
        best_model, X_test, feature_cols, CFG, top_n=10
    )

    print("\n-- Pipeline complete --------------------------------------------")
    print(f"  Model artefact : {CFG['model_output']}")
    print(f"  SHAP outputs   : {CFG['shap_output_dir']}/")
    print("=" * 80)

    return best_model, top_features


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    model, top_features = main()
