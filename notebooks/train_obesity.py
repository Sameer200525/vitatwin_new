"""
VitaTwin - Obesity Risk Model
Dataset: ObesityDataSet_raw_and_data_sinthetic.csv

IMPORTANT DESIGN DECISION: Height and Weight are EXCLUDED from the features.
NObeyesdad (the obesity category) is directly defined by BMI = Weight/Height^2,
so including them would make this a trivial BMI calculator dressed up as ML
(near 100% "accuracy" but zero predictive value). Instead we predict
obesity RISK purely from lifestyle/behavioral habits - which is the
genuinely useful, preventive-care framing for VitaTwin.

Target: binary - 1 = currently overweight or in an obesity category,
                  0 = normal weight or underweight
(collapsed from the original 7-class NObeyesdad label)
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
DATA_PATH = "/home/claude/vitatwin/data/ObesityDataSet_raw_and_data_sinthetic.csv"
MODEL_DIR = "/home/claude/vitatwin/models"

OVERWEIGHT_CLASSES = [
    "Overweight_Level_I", "Overweight_Level_II",
    "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
]

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")

y = df["NObeyesdad"].isin(OVERWEIGHT_CLASSES).astype(int)
print("Class balance:", y.value_counts(normalize=True).to_dict())

X = df.drop(columns=["NObeyesdad", "Height", "Weight"]).copy()

# Encode categoricals, saving each encoder so the API layer can reuse the exact mapping
cat_cols = X.select_dtypes(include="object").columns.tolist()
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c])
    encoders[c] = list(le.classes_)  # index position = encoded value

feature_cols = X.columns.tolist()

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

clf = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

pred_val = clf.predict(X_val)
print("\n=== Validation ===")
print(classification_report(y_val, pred_val, digits=3))

pred_test = clf.predict(X_test)
proba_test = clf.predict_proba(X_test)
print("\n=== Test (final) ===")
print(classification_report(y_test, pred_test, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, pred_test))
acc = accuracy_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test, average="macro")
auc = roc_auc_score(y_test, proba_test[:, 1])
print("Accuracy:", acc, " Macro F1:", f1, " ROC-AUC:", auc)

perm = permutation_importance(clf, X_val, y_val, n_repeats=10, random_state=RANDOM_STATE, scoring="f1_macro")
importances = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
print("\nTop features:\n", importances.head(10))

# Refit on all data for the shipped model
final_model = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08, random_state=RANDOM_STATE)
final_model.fit(X, y)
joblib.dump(final_model, f"{MODEL_DIR}/obesity_model.joblib")

metadata = {
    "model_name": "obesity_risk_v1",
    "dataset": "ObesityDataSet_raw_and_data_sinthetic.csv",
    "target": "overweight_or_obese (binary, derived from NObeyesdad)",
    "classes": {0: "normal_or_underweight", 1: "overweight_or_obese"},
    "feature_order": feature_cols,
    "categorical_encoders": encoders,  # {column: [class_at_index_0, class_at_index_1, ...]}
    "test_accuracy": float(acc),
    "test_macro_f1": float(f1),
    "test_roc_auc": float(auc),
    "top_features": importances.head(10).to_dict(),
    "notes": (
        "Height/Weight excluded on purpose - the label is BMI-derived so "
        "including them would be a trivial calculator, not a real prediction. "
        "This model estimates obesity risk from lifestyle habits alone."
    ),
    "feature_descriptions": {
        "Gender": "Male/Female",
        "Age": "age in years",
        "family_history_with_overweight": "yes/no - family history of being overweight",
        "FAVC": "frequent consumption of high-caloric food (yes/no)",
        "FCVC": "frequency of vegetable consumption (1-3 scale)",
        "NCP": "number of main meals per day",
        "CAEC": "eating between meals: no/Sometimes/Frequently/Always",
        "SMOKE": "smokes (yes/no)",
        "CH2O": "daily water intake (1-3 scale, roughly liters)",
        "SCC": "monitors calorie consumption (yes/no)",
        "FAF": "physical activity frequency (0-3 scale)",
        "TUE": "time using technology devices (0-2 scale)",
        "CALC": "alcohol consumption: no/Sometimes/Frequently/Always",
        "MTRANS": "main transportation mode",
    }
}
with open(f"{MODEL_DIR}/obesity_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("\nSaved obesity model.")
