"""
Train 5 classification models on the UCI Dry Bean Dataset, evaluate them,
and persist the trained artifacts (models + preprocessing objects) so the
Streamlit app can load them without retraining.

Dataset: UCI Dry Bean Dataset
https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
13611 instances, 16 numeric features, 7 bean classes (multi-class problem).

Run:
    python src/train_models.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Dry_Bean_Dataset_full.csv"
MODEL_DIR = ROOT / "model"
TEST_CSV_PATH = ROOT / "test_data.csv"
METRICS_PATH = ROOT / "model" / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.15  # held out, untouched during training -> used as test_data.csv


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
        X, y, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Persist the untouched test split (raw, unscaled, human-readable) as the
    # test_data.csv asset required by the assignment. The Streamlit app will
    # load the saved scaler + models and transform this on the fly.
    test_df = X_test.copy()
    test_df[target_col] = y_test_raw.values
    test_df.to_csv(TEST_CSV_PATH, index=False)
    test_df.to_csv(ROOT / "data" / "test_data.csv", index=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE
        ),
    }

    results = {}
    n_classes = len(label_encoder.classes_)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="weighted"
        )
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        results[name] = {
            "Accuracy": round(float(acc), 4),
            "AUC": round(float(auc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1": round(float(f1), 4),
            "MCC": round(float(mcc), 4),
        }

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
        )

        results[name]["confusion_matrix"] = cm.tolist()
        results[name]["classification_report"] = report

        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / fname)
        print(f"[OK] {name:25s} -> {fname:35s} "
              f"Acc={acc:.4f} AUC={auc:.4f} F1={f1:.4f} MCC={mcc:.4f}")

    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")
    joblib.dump(feature_cols, MODEL_DIR / "feature_columns.joblib")

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved models, scaler, label encoder, feature list, and metrics.json")
    print(f"Test holdout CSV written to: {TEST_CSV_PATH}")

    # Print a clean comparison table for the README
    print("\n=== Comparison Table ===")
    header = f"{'Model':25s} {'Accuracy':>9s} {'AUC':>7s} {'Precision':>10s} {'Recall':>7s} {'F1':>7s} {'MCC':>7s}"
    print(header)
    for name, m in results.items():
        print(f"{name:25s} {m['Accuracy']:9.4f} {m['AUC']:7.4f} {m['Precision']:10.4f} {m['Recall']:7.4f} {m['F1']:7.4f} {m['MCC']:7.4f}")


if __name__ == "__main__":
    main()
