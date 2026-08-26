"""
VitaTwin - Diabetes Risk Model Training
Dataset: diabetes_012_health_indicators_BRFSS2015.csv
Target: Diabetes_012  (0 = no diabetes, 1 = pre-diabetic, 2 = diabetic)

All features are things a user can plausibly self-report (no lab values),
which fits VitaTwin's "manual input only" data collection model.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, accuracy_score
)

RANDOM_STATE = 42
DATA_PATH = "/home/claude/vitatwin/data/diabetes_012_health_indicators_BRFSS2015.csv"
MODEL_DIR = "/home/claude/vitatwin/models"

TARGET = "Diabetes_012"

# ---------------------------------------------------------------------------
# 1. Load + clean
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Raw shape: {df.shape}")

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {before - len(df)} exact-duplicate rows -> {df.shape}")

# All columns are already numeric/binary/ordinal-coded in this dataset.
feature_cols = [c for c in df.columns if c != TARGET]
X = df[feature_cols].copy()
y_3class = df[TARGET].astype(int).copy()

# Binary framing: 0 = no diabetes, 1 = at risk (prediabetic OR diabetic).
# Chosen as the PRIMARY model because the 3-class version cannot separate
# "prediabetes" (2% of data) from the other classes with these features
# (F1 ~0.05 on that class) - see notes below. Binary risk is also more
# actionable for a preventive-care product.
y = (y_3class >= 1).astype(int)

print("\n3-class distribution (kept for reference/secondary view):")
print(y_3class.value_counts(normalize=True).sort_index())
print("\nBinary ('at risk') distribution (PRIMARY target):")
print(y.value_counts(normalize=True).sort_index())

# ---------------------------------------------------------------------------
# 2. Split (stratified so the rare classes are represented in each split)
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)
print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------------------
# 3. Baseline: Logistic Regression (interpretable, fast)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

logreg = LogisticRegression(
    max_iter=1000, class_weight="balanced",
    random_state=RANDOM_STATE
)
logreg.fit(X_train_s, y_train)
pred_val_lr = logreg.predict(X_val_s)

print("\n=== Logistic Regression (validation) ===")
print(classification_report(y_val, pred_val_lr, digits=3))
print("Macro F1:", f1_score(y_val, pred_val_lr, average="macro"))

# ---------------------------------------------------------------------------
# 4. Main model: HistGradientBoostingClassifier
#    (sklearn's built-in equivalent to LightGBM/XGBoost, handles imbalance
#     via sample_weight, no external dependency needed)
# ---------------------------------------------------------------------------
class_counts = y_train.value_counts()
class_weight_map = {c: len(y_train) / (len(class_counts) * cnt) for c, cnt in class_counts.items()}
sample_weight = y_train.map(class_weight_map).values

hgb = HistGradientBoostingClassifier(
    max_iter=300,
    max_depth=8,
    learning_rate=0.08,
    l2_regularization=1.0,
    random_state=RANDOM_STATE
)
hgb.fit(X_train, y_train, sample_weight=sample_weight)
pred_val_hgb = hgb.predict(X_val)
proba_val_hgb = hgb.predict_proba(X_val)

print("\n=== HistGradientBoosting (validation) ===")
print(classification_report(y_val, pred_val_hgb, digits=3))
print("Macro F1:", f1_score(y_val, pred_val_hgb, average="macro"))
print("ROC-AUC:", roc_auc_score(y_val, proba_val_hgb[:, 1]))

# ---------------------------------------------------------------------------
# 5. Final evaluation on held-out test set (best model = HGB)
# ---------------------------------------------------------------------------
pred_test = hgb.predict(X_test)
proba_test = hgb.predict_proba(X_test)

print("\n=== HistGradientBoosting (TEST SET - final numbers) ===")
print(classification_report(y_test, pred_test, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, pred_test))
print("Accuracy:", accuracy_score(y_test, pred_test))
print("Macro F1:", f1_score(y_test, pred_test, average="macro"))
print("ROC-AUC:", roc_auc_score(y_test, proba_test[:, 1]))

# ---------------------------------------------------------------------------
# 6. Feature importance (permutation-free, HGB gives no native importances,
#    so we use sklearn's permutation_importance on a validation subsample)
# ---------------------------------------------------------------------------
from sklearn.inspection import permutation_importance
sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_val), size=min(20000, len(X_val)), replace=False)
perm = permutation_importance(
    hgb, X_val.iloc[sample_idx], y_val.iloc[sample_idx],
    n_repeats=5, random_state=RANDOM_STATE, scoring="f1_macro"
)
importances = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
print("\nTop feature importances:")
print(importances.head(10))

# ---------------------------------------------------------------------------
# 7. Save model + metadata
# ---------------------------------------------------------------------------
joblib.dump(hgb, f"{MODEL_DIR}/diabetes_model.joblib")

metadata = {
    "model_name": "diabetes_risk_v1",
    "target": TARGET,
    "framing": "binary: 0 = no diabetes, 1 = at risk (prediabetic or diabetic)",
    "classes": {0: "no_diabetes", 1: "at_risk"},
    "feature_order": feature_cols,
    "test_accuracy": float(accuracy_score(y_test, pred_test)),
    "test_macro_f1": float(f1_score(y_test, pred_test, average="macro")),
    "test_roc_auc": float(roc_auc_score(y_test, proba_test[:, 1])),
    "top_features": importances.head(10).to_dict(),
    "feature_descriptions": {
        "HighBP": "1 = told by a doctor they have high blood pressure",
        "HighChol": "1 = told by a doctor they have high cholesterol",
        "CholCheck": "1 = had cholesterol checked in last 5 years",
        "BMI": "Body Mass Index (numeric)",
        "Smoker": "1 = smoked at least 100 cigarettes in lifetime",
        "Stroke": "1 = ever told they had a stroke",
        "HeartDiseaseorAttack": "1 = coronary heart disease or MI history",
        "PhysActivity": "1 = physical activity in past 30 days (excl. job)",
        "Fruits": "1 = consumes fruit 1+ times/day",
        "Veggies": "1 = consumes vegetables 1+ times/day",
        "HvyAlcoholConsump": "1 = heavy alcohol consumption",
        "AnyHealthcare": "1 = has any health coverage",
        "NoDocbcCost": "1 = couldn't see doctor due to cost in past year",
        "GenHlth": "self-rated general health, 1=excellent .. 5=poor",
        "MentHlth": "days of poor mental health in past 30 days (0-30)",
        "PhysHlth": "days of poor physical health in past 30 days (0-30)",
        "DiffWalk": "1 = serious difficulty walking/climbing stairs",
        "Sex": "0 = female, 1 = male",
        "Age": "13-level age bracket code (1=18-24 ... 13=80+)",
        "Education": "1-6 education level code",
        "Income": "1-8 income level code",
    }
}
with open(f"{MODEL_DIR}/diabetes_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

joblib.dump(scaler, f"{MODEL_DIR}/diabetes_scaler.joblib")  # kept in case we want logreg fallback later

print("\nSaved model + metadata to", MODEL_DIR)
