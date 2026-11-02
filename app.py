"""
Streamlit app: Dry Bean Multi-Class Classification Demo

Loads 5 pre-trained classifiers (Logistic Regression, Decision Tree, KNN,
Naive Bayes, SVM) trained on the UCI Dry Bean Dataset and lets a user
upload a test CSV to see predictions and evaluation metrics for each
model WITHOUT retraining anything.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
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

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Dry Bean Classifier Demo", layout="wide")


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}
    with open(MODEL_DIR / "metrics.json") as f:
        train_time_metrics = json.load(f)
    return scaler, label_encoder, feature_cols, models, train_time_metrics


scaler, label_encoder, feature_cols, models, train_time_metrics = load_artifacts()

st.title("Dry Bean Multi-Class Classification")
st.caption(
    "Compare 5 pre-trained classifiers (Logistic Regression, Decision Tree, "
    "KNN, Naive Bayes, SVM) on the UCI Dry Bean Dataset. Upload a test CSV "
    "to score it against the saved models — nothing is retrained here."
)

with st.sidebar:
    st.header("1. Upload test data")
    uploaded_file = st.file_uploader(
        "Upload test CSV (must contain all 16 feature columns + 'Class' target column)",
        type=["csv"],
    )
    st.markdown("---")
    st.header("2. Choose a model")
    model_choice = st.selectbox("Model", list(MODEL_FILES.keys()))
    st.markdown("---")
    st.caption(
        "Don't have a file handy? Use the `test_data.csv` included in this "
        "repository — it is a held-out split that was never used for training."
    )

if uploaded_file is None:
    st.info("Upload a test CSV from the sidebar to see predictions and metrics.")
    st.subheader("Expected columns")
    st.code(", ".join(feature_cols + ["Class"]))
    st.stop()

df = pd.read_csv(uploaded_file)

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required feature column(s): {missing_cols}")
    st.stop()
if "Class" not in df.columns:
    st.error("Uploaded CSV must contain the target column 'Class'.")
    st.stop()

X_raw = df[feature_cols].copy()
y_true_raw = df["Class"].copy()

unseen_labels = set(y_true_raw.unique()) - set(label_encoder.classes_)
if unseen_labels:
    st.error(f"Target column contains labels not seen during training: {unseen_labels}")
    st.stop()

y_true = label_encoder.transform(y_true_raw)
X_scaled = scaler.transform(X_raw)

st.subheader(f"Results — {model_choice}")

model = models[model_choice]
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)

acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)

metric_cols = st.columns(6)
labels = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
values = [acc, auc, prec, rec, f1, mcc]
for col, label, value in zip(metric_cols, labels, values):
    col.metric(label, f"{value:.4f}")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig)

with col_right:
    st.markdown("**Classification Report**")
    report = classification_report(
        y_true, y_pred, target_names=label_encoder.classes_,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)

st.markdown("---")
st.subheader("All 5 models on this uploaded test set")

rows = []
for name, m in models.items():
    yp = m.predict(X_scaled)
    ypb = m.predict_proba(X_scaled)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, yp),
        "AUC": roc_auc_score(y_true, ypb, multi_class="ovr", average="weighted"),
        "Precision": precision_score(y_true, yp, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, yp, average="weighted", zero_division=0),
        "F1": f1_score(y_true, yp, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, yp),
    })

comparison_df = pd.DataFrame(rows).set_index("Model").round(4)
st.dataframe(comparison_df, use_container_width=True)
st.bar_chart(comparison_df[["Accuracy", "F1", "MCC"]])

with st.expander("Metrics recorded at training time (on the original held-out split)"):
    train_metrics_df = pd.DataFrame(
        {name: {k: v for k, v in m.items() if k in
                ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}
         for name, m in train_time_metrics.items()}
    ).T
    st.dataframe(train_metrics_df, use_container_width=True)
