# Dry Bean Multi-Class Classification — Streamlit Demo

## a. Problem Statement

Dry beans are commonly graded and separated by variety for seed production,
commercial sale, and quality control. Manual sorting by shape and visual
inspection is slow and inconsistent. This project frames bean variety
identification as a **multi-class classification problem**: given 16
morphological (shape and size) features extracted from images of beans,
predict which of **7 bean varieties** a given bean belongs to
(`SEKER`, `BARBUNYA`, `BOMBAY`, `CALI`, `HOROZ`, `SIRA`, `DERMASON`).

Five classical ML classifiers are trained on the same dataset, evaluated
with a common set of metrics, and compared through an interactive
Streamlit web application that loads the already-trained models and
scores any uploaded test CSV on demand — no retraining happens in the app.

## b. Dataset Description

- **Source**: UCI Machine Learning Repository — [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Instances**: 13,611 bean images (grains)
- **Features**: 16 numeric morphological features extracted via computer
  vision (exceeds the assignment's minimum of 12), including `Area`,
  `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRation`,
  `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`,
  `roundness`, `Compactness`, `ShapeFactor1`–`ShapeFactor4`.
- **Target**: `Class` — one of 7 bean varieties (multi-class, not binary).
- **Class distribution**: DERMASON (3546), SIRA (2636), SEKER (2027),
  HOROZ (1928), CALI (1630), BARBUNYA (1322), BOMBAY (522) — moderately
  imbalanced, handled using stratified train/test splitting and
  weighted-averaged metrics.
- **Split used**: 85% train (11,569 rows) / 15% held-out test
  (2,042 rows), stratified by class. The held-out 2,042-row split was
  **never used during training** and is shipped as `test_data.csv` /
  `data/test_data.csv` for use in the Streamlit app.

## c. GitHub Repository Link

> https://github.com/subnkve/dry-bean-classifier-app

## d. Models Used

All 5 models were trained on an identical `StandardScaler`-scaled
training split of the Dry Bean Dataset and evaluated on the same
held-out 2,042-row test split.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9216 | 0.9931 | 0.9225 | 0.9216 | 0.9218 | 0.9054 |
| Decision Tree | 0.9011 | 0.9511 | 0.9013 | 0.9011 | 0.9011 | 0.8804 |
| KNN | 0.9182 | 0.9829 | 0.9188 | 0.9182 | 0.9183 | 0.9011 |
| Naive Bayes (Gaussian) | 0.9011 | 0.9902 | 0.9041 | 0.9011 | 0.9012 | 0.8812 |
| Support Vector Machine (RBF) | 0.9236 | 0.9934 | 0.9238 | 0.9236 | 0.9236 | 0.9077 |

*(Precision, Recall, F1 are weighted-averaged across the 7 classes; AUC is
weighted one-vs-rest; see `model/metrics.json` for the full breakdown
including per-class confusion matrices and classification reports, and
`src/train_models.py` for the exact computation.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Despite being a purely linear model, it achieves the 2nd-highest Accuracy and AUC. The 7 bean classes are largely linearly separable once features are standardized, so a linear decision boundary already captures most of the structure. Confusion mainly occurs between morphologically close classes (e.g., SIRA vs DERMASON). |
| Decision Tree | Lowest AUC (0.9511) of all five models and tied for lowest Accuracy. Many of the 16 features are highly correlated (e.g., `Area`, `Perimeter`, `ConvexArea`, `EquivDiameter` all encode size), which encourages a single tree to overfit on a few dominant splits, hurting probability calibration and ranking quality even though raw accuracy stays respectable. |
| KNN | Performs close to the top models (Accuracy 0.9182, AUC 0.9829), confirming that same-class beans form fairly compact, well-separated clusters in standardized feature space. Performance is sensitive to the chosen k (7 used here) and to the redundant/correlated features, which can distort Euclidean distances. |
| Naive Bayes (Gaussian) | Ties Decision Tree on Accuracy but has a much higher AUC (0.9902), meaning it ranks class probabilities well even when its hard predictions are wrong. This is because the conditional-independence assumption is clearly violated by the strongly correlated shape/size features, which biases the predicted class boundaries without destroying the relative probability ranking. |
| Support Vector Machine (RBF) | **Best overall** — highest Accuracy, AUC, F1 and MCC. The RBF kernel captures non-linear boundaries between morphologically similar classes (e.g., SIRA/DERMASON, BARBUNYA/CALI) better than a linear boundary, at the cost of being the most computationally expensive model to train and to score. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF kernel)** — it leads on every single metric (Accuracy 0.9236, AUC 0.9934, F1 0.9236, MCC 0.9077), with Logistic Regression a close, much cheaper runner-up. |

## Project Structure

```
bean-classifier-app/
├── app.py                      # Streamlit app (entry point)
├── requirements.txt
├── README.md
├── test_data.csv               # held-out test split (features + target)
├── data/
│   ├── Dry_Bean_Dataset_full.csv   # full raw dataset
│   └── test_data.csv               # copy of the held-out test split
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   ├── scaler.joblib               # StandardScaler fit on training data
│   ├── label_encoder.joblib        # encodes the 7 class labels
│   ├── feature_columns.joblib      # ordered list of expected feature columns
│   └── metrics.json                # full metrics + confusion matrices + reports
└── src/
    └── train_models.py         # trains all 5 models and saves the artifacts above
```

## How to Run Locally

```bash
pip install -r requirements.txt
python src/train_models.py      # optional: retrains models and regenerates artifacts
streamlit run app.py
```

## Streamlit App Features

- **Dataset upload (CSV)**: upload any CSV containing the 16 feature
  columns plus the `Class` target column (use `test_data.csv` as a ready
  example).
- **Model selection dropdown**: choose between Logistic Regression,
  Decision Tree, KNN, Naive Bayes, and SVM.
- **Evaluation metrics display**: Accuracy, AUC, Precision, Recall, F1,
  MCC for the selected model, plus a side-by-side comparison table/chart
  across all 5 models on the uploaded data.
- **Confusion matrix & classification report**: rendered for the
  selected model.

The app **loads the pre-trained `.joblib` artifacts and never retrains**
on the uploaded data — it only transforms (via the saved `StandardScaler`)
and predicts.

## Live Streamlit App Link

> https://<your-app-name>.streamlit.app

*(Fill in after deploying — see Step 6 of the assignment / deployment
notes below.)*
