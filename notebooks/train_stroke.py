"""
VitaTwin - Stroke Risk Model v2
Dataset: stroke_data.csv (balanced 50/50, synthetically expanded to 40,910 rows)

Same leakage concern as heart disease data: synthetic oversampling creates
near-duplicate rows. We use the same group-aware, leakage-safe evaluation
approach here.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
DATA_PATH = "/home/claude/vitatwin/data/stroke_data.csv"
MODEL_DIR = "/home/claude/vitatwin/models"
TARGET = "stroke"

df = pd.read_csv(DATA_PATH).dropna().reset_index(drop=True)
print(f"Shape: {df.shape}")

feature_cols = [c for c in df.columns if c != TARGET]
X = df[feature_cols].copy()
y = df[TARGET].astype(int).copy()
print("Class balance:", y.value_counts(normalize=True).to_dict())

Xs = StandardScaler().fit_transform(X)
db = DBSCAN(eps=0.6, min_samples=2).fit(Xs)
groups = db.labels_.copy()
noise_idx = np.where(groups == -1)[0]
next_id = groups.max() + 1
for i in noise_idx:
    groups[i] = next_id
    next_id += 1
print(f"{len(df)} rows collapsed into {len(set(groups))} near-duplicate clusters")

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

eval_model = HistGradientBoostingClassifier(max_iter=250, max_depth=6, learning_rate=0.08, random_state=RANDOM_STATE)
eval_model.fit(X_train, y_train)
pred_test = eval_model.predict(X_test)
proba_test = eval_model.predict_proba(X_test)

print("\n=== LEAKAGE-SAFE TEST RESULTS (honest numbers) ===")
print(classification_report(y_test, pred_test, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, pred_test))
acc = accuracy_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test, average="macro")
auc = roc_auc_score(y_test, proba_test[:, 1])
print("Accuracy:", acc, " Macro F1:", f1, " ROC-AUC:", auc)

sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), size=min(6000, len(X_test)), replace=False)
perm = permutation_importance(eval_model, X_test.iloc[sample_idx], y_test.iloc[sample_idx], n_repeats=5, random_state=RANDOM_STATE, scoring="f1_macro")
importances = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
print("\nTop features:\n", importances.head(10))

final_model = HistGradientBoostingClassifier(max_iter=250, max_depth=6, learning_rate=0.08, random_state=RANDOM_STATE)
final_model.fit(X, y)
joblib.dump(final_model, f"{MODEL_DIR}/stroke_model.joblib")

metadata = {
    "model_name": "stroke_risk_v2",
    "dataset": "stroke_data.csv (balanced, synthetically expanded)",
    "target": TARGET,
    "classes": {0: "lower_risk", 1: "at_risk"},
    "feature_order": feature_cols,
    "honest_test_accuracy": float(acc),
    "honest_test_macro_f1": float(f1),
    "honest_test_roc_auc": float(auc),
    "top_features": importances.head(10).to_dict(),
    "notes": "Evaluated with a group-aware split to avoid synthetic near-duplicate leakage. Final model refit on all data.",
    "feature_descriptions": {
        "sex": "1 = male, 0 = female",
        "age": "age in years",
        "hypertension": "1 = has hypertension",
        "heart_disease": "1 = has heart disease",
        "ever_married": "1 = yes",
        "work_type": "encoded work category",
        "Residence_type": "1 = urban, 0 = rural",
        "avg_glucose_level": "average blood glucose level",
        "bmi": "body mass index",
        "smoking_status": "encoded smoking category",
    }
}
with open(f"{MODEL_DIR}/stroke_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("\nSaved stroke model.")
