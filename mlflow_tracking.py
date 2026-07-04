# =============================================================================
# mlflow_tracking.py  —  Mutual Fund Predictive Analytics
# =============================================================================
#
# WHAT THIS FILE DOES (read this first!):
#   1. Loads your real Fund_Statistics annualized CSV files (3yr, 5yr, 10yr)
#   2. Builds features: Alpha, Beta, R_Squared, Sharpe_Ratio, Sortino_Ratio,
#      Std_Dev, Treynor_Ratio  (exactly as in your data)
#   3. Trains a Random Forest Classifier to predict top-performing funds
#   4. Tracks EVERYTHING with MLflow — params, metrics, model, charts
#   5. Runs 3 experiments with different settings so your dashboard has
#      multiple runs to compare (this is what looks great in screenshots)
#
# HOW TO RUN:
#   1. Open terminal in VS Code  (Ctrl + `)
#   2. Make sure you are in the project root folder
#   3. Run:  python mlflow_tracking.py
#   4. Then: mlflow ui
#   5. Open browser:  http://127.0.0.1:5000
#
# =============================================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # prevents GUI popup when saving charts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# =============================================================================
# STEP 1 — LOAD YOUR DATA
# =============================================================================
# Your project has 4 annualized stat files. We load all of them and combine.
# The columns we use:
#   Alpha, Beta, R_Squared, Sharpe_Ratio, Sortino_Ratio, Std_Dev, Treynor_Ratio
#   Return  (this is what we predict: top vs poor)
#   Year_Trailing  (3, 5, 10, 15 — lets us filter by investment horizon)

def load_fund_data():
    """
    Loads and combines all annualized fund statistics CSV files.
    Returns a cleaned DataFrame ready for modelling.
    """
    data_folder = os.path.join("Fund_Data", "Fund_Stats_Annualized_data")

    file_map = {
        3:  "Fund_statistics_3years.csv",
        5:  "Fund_statistics_5years.csv",
        10: "Fund_statistics_10years.csv",
        15: "Fund_statistics_15years.csv",
    }

    frames = []
    for years, filename in file_map.items():
        filepath = os.path.join(data_folder, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0)
            frames.append(df)
            print(f"  Loaded {filename}  →  {len(df)} rows")
        else:
            print(f"  WARNING: {filepath} not found — skipping")

    if not frames:
        raise FileNotFoundError(
            "No data files found. Make sure you run this script from the "
            "project root folder (same level as Fund_Data/)."
        )

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows after combining all files: {len(combined)}")
    return combined


def prepare_features(df, horizon_years=None):
    """
    Cleans the DataFrame and creates X (features) and y (label).

    Parameters
    ----------
    df            : raw combined DataFrame
    horizon_years : int or None — filter to a specific Year_Trailing value
                    (3, 5, 10, or 15). None = use all years combined.

    Returns
    -------
    X             : feature DataFrame
    y             : binary label Series  (1 = top performer, 0 = poor)
    feature_cols  : list of column names used as features
    """

    # --- optional filter by investment horizon ---
    if horizon_years is not None:
        df = df[df["Year_Trailing"] == horizon_years].copy()
        print(f"  Filtered to {horizon_years}-year horizon: {len(df)} rows")

    # --- feature columns (exactly what's in your CSV) ---
    feature_cols = [
        "Alpha",
        "Beta",
        "R_Squared",
        "Sharpe_Ratio",
        "Sortino_Ratio",
        "Std_Dev",
        "Treynor_Ratio",
    ]

    # keep only rows where all features AND Return exist
    needed = feature_cols + ["Return"]
    df = df.dropna(subset=needed).copy()
    print(f"  Rows after dropping NaN: {len(df)}")

    # --- create label: top performer = Return above 75th percentile ---
    # This matches your original logic of identifying top funds vs poor funds
    threshold = df["Return"].quantile(0.75)
    df["is_top_performer"] = (df["Return"] >= threshold).astype(int)
    print(f"  Return threshold (75th pct): {threshold:.2f}%")
    print(f"  Top performers: {df['is_top_performer'].sum()}  |  "
          f"Others: {(df['is_top_performer'] == 0).sum()}")

    X = df[feature_cols]
    y = df["is_top_performer"]

    return X, y, feature_cols, df


# =============================================================================
# STEP 2 — PLOT FUNCTIONS  (these produce the artifact charts)
# =============================================================================

def plot_feature_importance(model, feature_cols, save_path):
    """Bar chart of feature importances — saved as PNG artifact."""
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71" if i == sorted_idx[0] else "#3498db" for i in range(len(feature_cols))]
    bars = ax.bar(
        range(len(feature_cols)),
        importances[sorted_idx],
        color=[colors[i] for i in range(len(feature_cols))],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i] for i in sorted_idx], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Importance score", fontsize=10)
    ax.set_title("Random Forest — Feature Importance", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(y_test, y_pred, save_path):
    """Confusion matrix saved as PNG artifact."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Poor / Average", "Top Performer"],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_return_distribution(df_filtered, horizon_years, save_path):
    """Histogram of fund returns — saved as PNG artifact."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_filtered["Return"].dropna(), bins=40, color="#3498db",
            edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(df_filtered["Return"].quantile(0.75), color="#e74c3c",
               linewidth=1.5, linestyle="--", label="75th pct (top threshold)")
    ax.set_xlabel("Annualized Return (%)", fontsize=10)
    ax.set_ylabel("Number of funds", fontsize=10)
    ax.set_title(f"Return distribution — {horizon_years}-year horizon", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def save_predictions_csv(y_test, y_pred, save_path):
    """Saves actual vs predicted as CSV artifact."""
    results_df = pd.DataFrame({
        "actual_label": y_test.values,
        "predicted_label": y_pred,
        "correct": (y_test.values == y_pred).astype(int),
    })
    results_df.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")


# =============================================================================
# STEP 3 — CORE TRAINING + MLFLOW LOGGING FUNCTION
# =============================================================================

def train_and_log(
    run_name,
    horizon_years,
    n_estimators,
    max_depth,
    min_samples_split,
    random_state=42,
):
    """
    Trains one Random Forest experiment and logs everything to MLflow.

    Parameters
    ----------
    run_name         : string label shown in the MLflow dashboard
    horizon_years    : 3, 5, or 10  — which investment horizon to train on
    n_estimators     : number of trees in the forest
    max_depth        : maximum depth of each tree (None = unlimited)
    min_samples_split: minimum samples needed to split a node
    random_state     : for reproducibility
    """

    print(f"\n{'='*60}")
    print(f"  Starting run: {run_name}")
    print(f"  Horizon: {horizon_years} years | "
          f"n_estimators={n_estimators} | max_depth={max_depth}")
    print(f"{'='*60}")

    # --- load and prepare data ---
    raw_df      = load_fund_data()
    X, y, feature_cols, df_filtered = prepare_features(raw_df, horizon_years)

    if len(X) < 20:
        print(f"  SKIP: not enough data ({len(X)} rows) for {horizon_years}-year horizon")
        return

    # --- train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # --- build model ---
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )

    # --- cross-validation (5-fold) ---
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    # --- fit on full training set ---
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- compute metrics ---
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    mse       = mean_squared_error(y_test, y_pred)

    print(f"\n  Results:")
    print(f"    Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    MSE       : {mse:.6f}")
    print(f"    CV mean   : {cv_scores.mean():.4f}  ± {cv_scores.std():.4f}")

    # --- temporary folder for artifact files ---
    artifact_dir = "mlflow_artifacts_temp"
    os.makedirs(artifact_dir, exist_ok=True)

    feat_imp_path   = os.path.join(artifact_dir, "feature_importance.png")
    conf_mat_path   = os.path.join(artifact_dir, "confusion_matrix.png")
    return_dist_path= os.path.join(artifact_dir, "return_distribution.png")
    predictions_path= os.path.join(artifact_dir, "predictions.csv")

    plot_feature_importance(model, feature_cols, feat_imp_path)
    plot_confusion_matrix(y_test, y_pred, conf_mat_path)
    plot_return_distribution(df_filtered, horizon_years, return_dist_path)
    save_predictions_csv(y_test, y_pred, predictions_path)

    # ==========================================================================
    # THIS IS THE MLFLOW MAGIC — everything inside this block gets tracked
    # ==========================================================================
    with mlflow.start_run(run_name=run_name):

        # --- LOG PARAMETERS (inputs) ---
        # These are the "recipe settings" — what you put IN to the model
        mlflow.log_param("horizon_years",      horizon_years)
        mlflow.log_param("n_estimators",       n_estimators)
        mlflow.log_param("max_depth",          max_depth if max_depth else "None")
        mlflow.log_param("min_samples_split",  min_samples_split)
        mlflow.log_param("random_state",       random_state)
        mlflow.log_param("test_size",          0.2)
        mlflow.log_param("label_threshold_pct",75)
        mlflow.log_param("features_used",      ", ".join(feature_cols))
        mlflow.log_param("train_samples",      len(X_train))
        mlflow.log_param("test_samples",       len(X_test))
        mlflow.log_param("top_performer_count",int(y.sum()))

        # --- LOG METRICS (outputs) ---
        # These are the "how good was the cake?" scores
        mlflow.log_metric("accuracy",          round(accuracy, 4))
        mlflow.log_metric("precision",         round(precision, 4))
        mlflow.log_metric("recall",            round(recall, 4))
        mlflow.log_metric("f1_score",          round(f1, 4))
        mlflow.log_metric("mse",               round(mse, 6))
        mlflow.log_metric("cv_mean_accuracy",  round(cv_scores.mean(), 4))
        mlflow.log_metric("cv_std_accuracy",   round(cv_scores.std(), 4))

        # log feature importances as individual metrics — visible in dashboard
        for feat, imp in zip(feature_cols, model.feature_importances_):
            mlflow.log_metric(f"importance_{feat}", round(float(imp), 4))

        # --- LOG ARTIFACTS (files) ---
        # These are the actual files saved alongside the run
        mlflow.log_artifact(feat_imp_path,    artifact_path="charts")
        mlflow.log_artifact(conf_mat_path,    artifact_path="charts")
        mlflow.log_artifact(return_dist_path, artifact_path="charts")
        mlflow.log_artifact(predictions_path, artifact_path="data")

        # --- LOG THE MODEL ITSELF ---
        # This saves the trained sklearn model — you can reload it later
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            registered_model_name=f"MutualFund_RF_{horizon_years}yr",
        )

        # get the run ID so we can print it
        run_id = mlflow.active_run().info.run_id
        print(f"\n  MLflow run ID: {run_id}")
        print(f"  Run logged successfully to experiment: mutual_fund_analysis")

    print(f"\n  Run '{run_name}' complete!")
    return accuracy


# =============================================================================
# STEP 4 — RUN 3 EXPERIMENTS (so the dashboard has multiple runs to compare)
# =============================================================================
# This is the most important part for your GitHub portfolio.
# Running with different settings shows you understand hyperparameter tuning.

if __name__ == "__main__":

    # Set the experiment name — this is the "folder" in the MLflow dashboard
    mlflow.set_experiment("mutual_fund_analysis")

    print("\n" + "="*60)
    print("  MUTUAL FUND PREDICTIVE ANALYTICS — MLflow Tracking")
    print("="*60)
    print("\n  Running 3 experiments with different hyperparameters...")
    print("  After this finishes, run:  mlflow ui")
    print("  Then open:  http://127.0.0.1:5000\n")

    results = {}

    # --- Experiment 1: 3-year horizon, baseline model ---
    # This mirrors your original analysis in FundAnalysis.ipynb
    results["run1"] = train_and_log(
        run_name        = "3yr_baseline_n100",
        horizon_years   = 3,
        n_estimators    = 100,
        max_depth       = None,
        min_samples_split = 2,
    )

    # --- Experiment 2: 5-year horizon, more trees ---
    results["run2"] = train_and_log(
        run_name        = "5yr_deep_n200",
        horizon_years   = 5,
        n_estimators    = 200,
        max_depth       = 10,
        min_samples_split = 5,
    )

    # --- Experiment 3: 10-year horizon, shallow trees ---
    results["run3"] = train_and_log(
        run_name        = "10yr_shallow_n150",
        horizon_years   = 10,
        n_estimators    = 150,
        max_depth       = 5,
        min_samples_split = 10,
    )

    # --- Final summary ---
    print("\n" + "="*60)
    print("  ALL EXPERIMENTS COMPLETE")
    print("="*60)
    for name, acc in results.items():
        if acc is not None:
            print(f"  {name}: accuracy = {acc*100:.1f}%")

    print("\n  Next steps:")
    print("  1. Run:  mlflow ui")
    print("  2. Open: http://127.0.0.1:5000")
    print("  3. Click 'mutual_fund_analysis' experiment")
    print("  4. Compare runs, click on any run to see charts")
    print("  5. Take a screenshot for your mlflow/ folder in GitHub")
    print("="*60 + "\n")