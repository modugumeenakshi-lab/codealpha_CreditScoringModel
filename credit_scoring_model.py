"""
CodeAlpha Machine Learning Internship - Task 1: Credit Scoring Model
======================================================================
Objective : Predict an individual's creditworthiness (good/bad credit risk)
            using past financial data.
Approach  : Classification (Logistic Regression, Decision Tree, Random Forest)
Metrics   : Precision, Recall, F1-Score, ROC-AUC

This script:
 1. Generates/loads a realistic financial dataset (income, debts, payment
    history, credit utilization, etc.)
 2. Performs feature engineering
 3. Trains 3 classification models
 4. Evaluates and compares them using Precision, Recall, F1, ROC-AUC
 5. Saves the best model + a comparison chart

NOTE: This uses a synthetically generated dataset so the script runs
end-to-end with no external downloads. To use a real dataset instead
(e.g. German Credit Data or a Kaggle credit dataset), just replace the
`build_dataset()` function with `pd.read_csv("your_file.csv")` and make
sure the column names match, or adjust FEATURE_COLS / TARGET_COL below.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, classification_report, confusion_matrix
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------
# 1. Build the dataset (feature engineering happens here too)
# ---------------------------------------------------------------------
def build_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """Creates a synthetic but realistic credit-scoring dataset."""
    age = np.random.randint(21, 65, n_samples)
    annual_income = np.random.normal(55000, 20000, n_samples).clip(15000, 200000)
    existing_debt = np.random.normal(15000, 10000, n_samples).clip(0, 100000)
    credit_history_years = np.random.randint(0, 30, n_samples)
    num_late_payments = np.random.poisson(1.2, n_samples)
    num_credit_lines = np.random.randint(1, 15, n_samples)
    credit_utilization = np.random.beta(2, 5, n_samples)  # 0 to 1
    employment_years = np.random.randint(0, 40, n_samples)

    # --- Feature engineering ---
    debt_to_income = existing_debt / (annual_income + 1)

    # --- Construct a realistic target using a weighted "risk score" ---
    risk_score = (
        -0.00002 * annual_income
        + 3.5 * debt_to_income
        + 0.35 * num_late_payments
        + 1.8 * credit_utilization
        - 0.05 * credit_history_years
        - 0.02 * employment_years
        + np.random.normal(0, 0.5, n_samples)  # noise
    )
    # Convert risk score into a binary "default risk" label (1 = bad credit risk)
    threshold = np.percentile(risk_score, 75)  # ~25% are "bad" credit risk
    target = (risk_score > threshold).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income.round(2),
        "existing_debt": existing_debt.round(2),
        "credit_history_years": credit_history_years,
        "num_late_payments": num_late_payments,
        "num_credit_lines": num_credit_lines,
        "credit_utilization": credit_utilization.round(3),
        "employment_years": employment_years,
        "debt_to_income": debt_to_income.round(3),
        "default_risk": target,  # 1 = bad credit risk, 0 = good
    })
    return df


# ---------------------------------------------------------------------
# 2. Train & evaluate models
# ---------------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Good Credit", "Bad Credit"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return metrics, y_proba


def main():
    print("Building dataset...")
    df = build_dataset()
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"Bad credit risk rate: {df['default_risk'].mean():.2%}")

    FEATURE_COLS = [c for c in df.columns if c != "default_risk"]
    TARGET_COL = "default_risk"

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features (helps Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
    }

    results = []
    roc_data = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            metrics, y_proba = evaluate_model(name, model, X_test_scaled, y_test)
        else:
            model.fit(X_train, y_train)
            metrics, y_proba = evaluate_model(name, model, X_test, y_test)

        results.append(metrics)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, metrics["ROC-AUC"])

    # --- Comparison table ---
    results_df = pd.DataFrame(results).set_index("Model").round(3)
    print("\n=========== MODEL COMPARISON ===========")
    print(results_df)
    results_df.to_csv("model_comparison.csv")

    # --- Feature importance (Random Forest) ---
    rf_model = models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n=========== FEATURE IMPORTANCE (Random Forest) ===========")
    print(importances.round(3))

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC curves
    for name, (fpr, tpr, auc) in roc_data.items():
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves")
    axes[0].legend()

    # Feature importance
    importances.plot(kind="barh", ax=axes[1], color="#4C72B0")
    axes[1].set_title("Feature Importance (Random Forest)")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig("credit_scoring_results.png", dpi=150)
    print("\nSaved: model_comparison.csv, credit_scoring_results.png")


if __name__ == "__main__":
    main()
