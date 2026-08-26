"""
VitaTwin - Diabetes Risk Model v2
Dataset: diabetes_data.csv (balanced 50/50 BRFSS lifestyle indicators)
Target: Diabetes (0/1, binary)

This replaces v1 (trained on the imbalanced 3-class BRFSS file). Same
feature philosophy: everything here is self-reportable, no lab values.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
DATA_PATH = "/home/claude/vitatwin/data/diabetes_data.csv"
MODEL_DIR = "/home/claude/vitatwin/models"
TARGET = "Diabetes"

df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
print(f"Shape after dedup: {df.shape}")

feature_cols = [c for c in df.columns if c != TARGET]
X = df[feature_cols].copy()
y = df[TARGET].astype(int).copy()
print("Class balance:", y.value_counts(normalize=True).to_dict())

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

hgb = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08, l2_regularization=1.0, random_state=RANDOM_STATE)
hgb.fit(X_train, y_train)

pred_val = hgb.predict(X_val)
print("\n=== Validation ===")
print(classification_report(y_val, pred_val, digits=3))

pred_test = hgb.predict(X_test)
proba_test = hgb.predict_proba(X_test)
print("\n=== Test (final) ===")
print(classification_report(y_test, pred_test, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, pred_test))
acc = accuracy_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test, average="macro")
auc = roc_auc_score(y_test, proba_test[:, 1])
print("Accuracy:", acc, " Macro F1:", f1, " ROC-AUC:", auc)

sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_val), size=min(15000, len(X_val)), replace=False)
perm = permutation_importance(hgb, X_val.iloc[sample_idx], y_val.iloc[sample_idx], n_repeats=5, random_state=RANDOM_STATE, scoring="f1_macro")
importances = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
print("\nTop features:\n", importances.head(10))

joblib.dump(hgb, f"{MODEL_DIR}/diabetes_model.joblib")
metadata = {
    "model_name": "diabetes_risk_v2",
    "dataset": "diabetes_data.csv (balanced BRFSS lifestyle indicators)",
    "target": TARGET,
    "framing": "binary: 0 = no diabetes, 1 = diabetes/at risk",
    "classes": {0: "no_diabetes", 1: "at_risk"},
    "feature_order": feature_cols,
    "test_accuracy": float(acc),
    "test_macro_f1": float(f1),
    "test_roc_auc": float(auc),
    "top_features": importances.head(10).to_dict(),
    "notes": "Balanced dataset (real BRFSS survey responses, verified no synthetic-clone leakage between train/test).",
}
with open(f"{MODEL_DIR}/diabetes_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("\nSaved diabetes model v2.")
