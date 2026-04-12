import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "train_path": "train_data.csv",
    "val_path":   "val_data.csv",
    "test_path":  "test_data.csv",
    "target_col": "risk_level",
    "output_dir": "audit_outputs",

    # AUC above this from a single feature = probable leakage
    "single_feature_auc_warning": 0.85,

    # A single stump (depth=1 tree) achieving this = perfect linear separation
    "stump_auc_warning": 0.99,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load
# ─────────────────────────────────────────────────────────────────────────────
def load_all(cfg):
    train = pd.read_csv(cfg["train_path"])
    val   = pd.read_csv(cfg["val_path"])
    test  = pd.read_csv(cfg["test_path"])

    target = cfg["target_col"]

    # Convert risk_level to binary BEFORE concatenation
    if target == "risk_level":
        for df in [train, val, test]:
            df[target] = (df[target] == "HIGH").astype(int)

    full  = pd.concat([train, val, test], ignore_index=True)

    exclude_cols = [target, "customer_id"]
    feature_cols = [c for c in train.columns if c not in exclude_cols]


    print(f"  Combined rows : {len(full):,}")
    print(f"  Features      : {len(feature_cols)}")
    print(f"  Default rate  : {full[target].mean():.2%}\n")

    return train, val, test, full, feature_cols, target

# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-Feature AUC  (single-feature leakage scan)
# ─────────────────────────────────────────────────────────────────────────────
def per_feature_auc(full: pd.DataFrame, feature_cols: list, target: str, cfg: dict) -> pd.DataFrame:
    """
    Compute ROC-AUC of each feature individually against the target.

    Interpretation:
      AUC > 0.85 from a single raw feature is a strong leakage signal.
      Legitimate credit features rarely exceed 0.70-0.75 in isolation
      because default is multi-causal.

      AUC = 1.0 from a single feature means the target is a deterministic
      function of that feature — almost certainly a data generation artifact.

    Note: we use the raw feature value as the score (no model), so this is
    equivalent to asking "how well does sorting by this feature separate
    defaults from non-defaults?"
    """
    print("-- Per-feature AUC (leakage scan) --------------------------------")
    records = []
    y = full[target].values
    for col in feature_cols:
        x = full[col].values.astype(float)
        # Handle constant columns
        if np.std(x) == 0:
            auc = 0.5
        else:
            try:
                auc = roc_auc_score(y, x)
                auc = max(auc, 1 - auc)   # flip if < 0.5 (direction doesn't matter)
            except Exception:
                auc = 0.5
        records.append({"feature": col, "single_feature_auc": auc})

    df = pd.DataFrame(records).sort_values("single_feature_auc", ascending=False)

    warn_thresh = cfg["single_feature_auc_warning"]
    flagged = df[df["single_feature_auc"] >= warn_thresh]

    print(f"\n  {'Feature':<35} {'AUC':>8}  {'Flag':>6}")
    print("  " + "-" * 54)
    for _, row in df.iterrows():
        flag = " <<< LEAKAGE RISK" if row["single_feature_auc"] >= warn_thresh else ""
        print(f"  {row['feature']:<35} {row['single_feature_auc']:>8.4f}{flag}")

    if not flagged.empty:
        print(f"\n  FLAGGED FEATURES (AUC >= {warn_thresh}):")
        for _, row in flagged.iterrows():
            print(f"    {row['feature']}  (AUC={row['single_feature_auc']:.4f})")
    print()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Decision Stump Test  (perfect separation with one split)
# ─────────────────────────────────────────────────────────────────────────────
def stump_separation_test(full: pd.DataFrame, feature_cols: list, target: str, cfg: dict) -> pd.DataFrame:
    """
    Fit a depth-1 decision tree (stump) per feature.

    A stump achieving AUC > 0.99 means a single threshold on one feature
    separates all (or nearly all) defaults from non-defaults.

    This is the exact mechanism behind "early stopping at round 1":
    LightGBM's first tree finds this split and achieves near-perfect AUC
    immediately, so no subsequent tree improves things.

    We also report the split threshold so you can cross-reference against
    your data generation logic.
    """
    print("-- Decision Stump Separation Test --------------------------------")
    y = full[target].values
    records = []

    for col in feature_cols:
        X = full[[col]].values
        stump = DecisionTreeClassifier(max_depth=1, random_state=42)
        stump.fit(X, y)
        proba = stump.predict_proba(X)[:, 1]
        auc   = roc_auc_score(y, proba)
        auc   = max(auc, 1 - auc)

        # Extract the split threshold from the tree internals
        threshold = stump.tree_.threshold[0]
        records.append({
            "feature":         col,
            "stump_auc":       auc,
            "split_threshold": threshold,
        })

    df = pd.DataFrame(records).sort_values("stump_auc", ascending=False)

    warn = cfg["stump_auc_warning"]
    print(f"\n  {'Feature':<35} {'Stump AUC':>10} {'Split At':>12}  {'Flag':>6}")
    print("  " + "-" * 68)
    for _, row in df.iterrows():
        flag = " <<< PERFECT SPLIT" if row["stump_auc"] >= warn else ""
        print(f"  {row['feature']:<35} {row['stump_auc']:>10.4f} "
              f"{row['split_threshold']:>12.4f}{flag}")
    print()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Distribution Separation Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_distributions(
    full: pd.DataFrame,
    feature_cols: list,
    target: str,
    auc_df: pd.DataFrame,
    cfg: dict,
    top_n: int = 6,
):
    """
    Plot KDE distributions of top-N features by AUC, split by default/non-default.

    Non-overlapping distributions confirm perfect separation.
    A feature with zero overlap between the two classes is a leakage indicator.
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(exist_ok=True)

    top_features = auc_df.head(top_n)["feature"].tolist()
    n_cols = 3
    n_rows = (len(top_features) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        defaults     = full.loc[full[target] == 1, feat].dropna()
        non_defaults = full.loc[full[target] == 0, feat].dropna()

        # Check for zero variance before KDE
        if defaults.std() == 0 and non_defaults.std() == 0:
            ax.text(0.5, 0.5, "Zero variance", ha="center", transform=ax.transAxes)
        else:
            try:
                defaults.plot.kde(ax=ax, label="Default", color="crimson", linewidth=2)
            except Exception:
                ax.hist(defaults, alpha=0.5, color="crimson", label="Default", density=True)
            try:
                non_defaults.plot.kde(ax=ax, label="No Default", color="steelblue",
                                      linewidth=2, linestyle="--")
            except Exception:
                ax.hist(non_defaults, alpha=0.5, color="steelblue",
                        label="No Default", density=True)

        auc_val = auc_df.loc[auc_df["feature"] == feat, "single_feature_auc"].values[0]
        ax.set_title(f"{feat}\n(Single-feature AUC={auc_val:.3f})", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel(feat, fontsize=8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Default Label\n"
                 "(Non-overlapping = perfect separation = likely leakage)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = output_dir / "feature_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Distribution plot --> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Target Correlation Check
# ─────────────────────────────────────────────────────────────────────────────
def target_correlation_check(full: pd.DataFrame, feature_cols: list, target: str) -> pd.DataFrame:
    """
    Point-biserial correlation between each numeric feature and the binary target.

    Correlation > 0.90 in absolute value is a near-certain leakage indicator.
    Legitimate features in credit risk rarely exceed 0.30-0.40.
    """
    print("-- Target Correlation Check (point-biserial) ---------------------")
    y = full[target].values
    records = []
    for col in feature_cols:
        x = full[col].values.astype(float)
        if np.std(x) == 0:
            corr, pval = 0.0, 1.0
        else:
            corr, pval = stats.pointbiserialr(y, x)
        records.append({"feature": col, "correlation": corr, "p_value": pval})

    df = (pd.DataFrame(records)
          .assign(abs_corr=lambda d: d["correlation"].abs())
          .sort_values("abs_corr", ascending=False)
          .drop(columns="abs_corr"))

    print(f"\n  {'Feature':<35} {'Corr':>8} {'p-value':>12}  {'Flag':>6}")
    print("  " + "-" * 62)
    for _, row in df.iterrows():
        flag = " <<< VERY HIGH" if abs(row["correlation"]) > 0.90 else \
               " <<< HIGH"     if abs(row["correlation"]) > 0.60 else ""
        print(f"  {row['feature']:<35} {row['correlation']:>8.4f} "
              f"{row['p_value']:>12.2e}{flag}")
    print()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Train / Test Distribution Drift Check
# ─────────────────────────────────────────────────────────────────────────────
def distribution_drift_check(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    feature_cols: list,
    cfg: dict,
) -> pd.DataFrame:
    """
    Kolmogorov-Smirnov test between train and test distributions per feature.

    In synthetic datasets, train and test are often drawn from the exact same
    distribution with zero drift — KS p-values close to 1.0 confirm this.

    In real data, some drift is expected.  p-value < 0.05 flags meaningful
    distribution shift that could degrade production performance.

    This does not directly detect leakage, but confirms whether the splits
    are representative and whether temporal ordering was respected.
    """
    print("-- Train / Test Distribution Drift (KS test) --------------------")
    records = []
    for col in feature_cols:
        ks_stat, p_val = stats.ks_2samp(
            train[col].dropna().values,
            test[col].dropna().values,
        )
        records.append({"feature": col, "ks_stat": ks_stat, "p_value": p_val})

    df = pd.DataFrame(records).sort_values("ks_stat", ascending=False)

    print(f"\n  {'Feature':<35} {'KS Stat':>9} {'p-value':>12}  {'Flag':>6}")
    print("  " + "-" * 65)
    for _, row in df.iterrows():
        flag = " <<< DRIFT" if row["p_value"] < 0.05 else ""
        print(f"  {row['feature']:<35} {row['ks_stat']:>9.4f} "
              f"{row['p_value']:>12.4f}{flag}")
    print()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Print Diagnosis & Remediation
# ─────────────────────────────────────────────────────────────────────────────
def print_diagnosis(auc_df: pd.DataFrame, stump_df: pd.DataFrame,
                    corr_df: pd.DataFrame, cfg: dict):
    """
    Synthesise all audit signals into a ranked diagnosis with
    concrete remediation steps.
    """
    print("=" * 72)
    print("  AUDIT DIAGNOSIS")
    print("=" * 72)

    max_auc   = auc_df["single_feature_auc"].max()
    max_stump = stump_df["stump_auc"].max()
    max_corr  = corr_df["correlation"].abs().max()

    top_leaker     = auc_df.iloc[0]["feature"]
    top_stump_feat = stump_df.iloc[0]["feature"]
    top_stump_thr  = stump_df.iloc[0]["split_threshold"]

    if max_stump >= cfg["stump_auc_warning"]:
        print(f"""
  DIAGNOSIS: PERFECT SEPARATION (deterministic data generation)
  -------------------------------------------------------------
  Feature '{top_stump_feat}' perfectly separates defaults from
  non-defaults at a single threshold of {top_stump_thr:.4f}.

  This is why LightGBM achieves AUC=1.0 at iteration 1:
    - Tree 1 finds this split
    - Residuals after tree 1 are near zero
    - All subsequent iterations see nothing to improve
    - Early stopping triggers at round 1

  PROBABLE CAUSE:
    Your synthetic data generation script likely used a rule such as:
      if feature_X > threshold: label = 1
    This creates a perfectly linearly separable dataset.

  REMEDIATION OPTIONS (choose based on your project goal):
    1. Add controlled noise to the label generation:
         default_prob = sigmoid(linear_combination_of_features + noise)
         label = bernoulli(default_prob)
       This forces the model to learn probabilistic boundaries.

    2. Use multiple features in the label generation rule:
         default_prob = f(feature_A, feature_B, feature_C, ...)
       No single feature should determine the outcome alone.

    3. Add feature noise before generating labels:
         features += np.random.normal(0, sigma, size=features.shape)
       This blurs the decision boundary.

    4. If using real data, investigate whether '{top_stump_feat}'
       is computed using post-default information (leakage).
       Verify the feature timestamp relative to the default event.
""")

    elif max_auc >= cfg["single_feature_auc_warning"]:
        print(f"""
  DIAGNOSIS: SINGLE-FEATURE LEAKAGE (high but not perfect)
  ---------------------------------------------------------
  Feature '{top_leaker}' has single-feature AUC = {max_auc:.4f}.

  PROBABLE CAUSE:
    This feature likely contains information that is only observable
    AFTER the default event (e.g., collection activity, late payment
    flags, account closure indicators).

  REMEDIATION:
    1. Audit the feature construction timestamp for '{top_leaker}'.
       It must be computed using data available BEFORE the prediction point.
    2. Apply a strict point-in-time cutoff: all features must be
       computed using data from T-N days before the default event window.
    3. Remove the feature and retrain — if AUC drops significantly,
       it was the source of leakage.
""")

    elif max_corr > 0.60:
        print(f"""
  DIAGNOSIS: HIGH CORRELATION (possible indirect leakage)
  -------------------------------------------------------
  No single feature perfectly separates defaults, but correlation
  is high (max |r| = {max_corr:.4f}).

  PROBABLE CAUSE:
    Feature engineering may have created aggregates that encode
    near-future behavior (e.g., 30-day rolling windows that
    extend into the default window).

  REMEDIATION:
    1. Review rolling window definitions — ensure windows close
       before the prediction point.
    2. Check for ratio features where denominator encodes default
       state (e.g., features computed differently for defaulted accounts).
""")

    else:
        print("""
  DIAGNOSIS: NO OBVIOUS LEAKAGE DETECTED
  ---------------------------------------
  Single-feature AUCs are within normal range.
  If LightGBM still achieves AUC > 0.95, the features collectively
  form a highly predictive set — validate that this holds on an
  out-of-time test set (different time period, not just a random split).
""")

    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  Credit-Risk Data Audit — Leakage & Separation Diagnostics")
    print("=" * 72)
    print()

    train, val, test, full, feature_cols, target = load_all(CFG)

    auc_df   = per_feature_auc(full, feature_cols, target, CFG)
    stump_df = stump_separation_test(full, feature_cols, target, CFG)
    corr_df  = target_correlation_check(full, feature_cols, target)
    drift_df = distribution_drift_check(train, test, feature_cols, CFG)

    print("-- Distribution Plots -------------------------------------------")
    plot_feature_distributions(full, feature_cols, target, auc_df, CFG, top_n=6)

    print_diagnosis(auc_df, stump_df, corr_df, CFG)

    print("\n-- Duplicate Row Check -------------------------------------------")
    print("Duplicate rows (train):", train.duplicated().sum())
    print("Duplicate rows (test) :", test.duplicated().sum())

    if "customer_id" in train.columns:
        overlap = set(train["customer_id"]) & set(test["customer_id"])
        print("Customer overlap between train/test:", len(overlap))

    return auc_df, stump_df, corr_df, drift_df

if __name__ == "__main__":
    auc_df, stump_df, corr_df, drift_df = main()
