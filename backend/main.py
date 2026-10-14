"""
VitaTwin Backend API
Run with: uvicorn main:app --reload --port 8000

Serves the five trained risk models. Hypertension is not an ML model; the
frontend reports a transparent measurement-based rule check instead.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
from pathlib import Path

from code_mappings import (
    age_to_bracket, GENERAL_HEALTH_OPTIONS, bmi as compute_bmi, WORK_TYPE_OPTIONS,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

app = FastAPI(title="VitaTwin API", version="0.1.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "VITATWIN_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model + metadata once at startup ---
diabetes_model = joblib.load(MODEL_DIR / "diabetes_model.joblib")
with open(MODEL_DIR / "diabetes_model_metadata.json") as f:
    diabetes_meta = json.load(f)

heart_model = joblib.load(MODEL_DIR / "heart_disease_model.joblib")
with open(MODEL_DIR / "heart_disease_model_metadata.json") as f:
    heart_meta = json.load(f)

stroke_model = joblib.load(MODEL_DIR / "stroke_model.joblib")
with open(MODEL_DIR / "stroke_model_metadata.json") as f:
    stroke_meta = json.load(f)

obesity_model = joblib.load(MODEL_DIR / "obesity_model.joblib")
with open(MODEL_DIR / "obesity_model_metadata.json") as f:
    obesity_meta = json.load(f)

kidney_model = joblib.load(MODEL_DIR / "kidney_model.joblib")
with open(MODEL_DIR / "kidney_model_metadata.json") as f:
    kidney_meta = json.load(f)


class DiabetesRiskInput(BaseModel):
    # Demographics
    age_years: int = Field(..., ge=18, le=100, description="Actual age in years")
    sex: str = Field(..., pattern="^(male|female)$")

    # Body
    weight_kg: float = Field(..., gt=0)
    height_cm: float = Field(..., gt=0)

    # Medical history (self-reported, yes/no)
    high_blood_pressure: bool
    high_cholesterol: bool
    cholesterol_checked_last_5_years: bool
    history_of_stroke: bool
    heart_disease_or_attack_history: bool
    difficulty_walking_or_climbing_stairs: bool

    # Lifestyle
    smoker: bool = Field(..., description="Smoked at least 100 cigarettes in lifetime")
    heavy_alcohol_consumption: bool
    physically_active_last_30_days: bool
    eats_fruit_daily: bool
    eats_vegetables_daily: bool

    # Self-rated wellbeing
    general_health: str = Field(..., description=f"One of: {list(GENERAL_HEALTH_OPTIONS.keys())}")
    poor_mental_health_days_last_30: int = Field(..., ge=0, le=30)
    poor_physical_health_days_last_30: int = Field(..., ge=0, le=30)


class DiabetesRiskOutput(BaseModel):
    risk_label: str
    risk_probability: float
    bmi: float
    global_feature_importance: list[str] = Field(
        ..., description="Dataset-level feature importance names, not verified causes for this individual prediction."
    )
    disclaimer: str = (
        "This is a statistical risk estimate based on population survey data, "
        "not a medical diagnosis. Consult a healthcare professional for advice."
    )


@app.get("/")
def root():
    return {"status": "VitaTwin API is running", "model": diabetes_meta["model_name"]}


@app.post("/predict/diabetes", response_model=DiabetesRiskOutput)
def predict_diabetes(payload: DiabetesRiskInput):
    if payload.general_health not in GENERAL_HEALTH_OPTIONS:
        raise HTTPException(400, f"Invalid general_health value. Options: {list(GENERAL_HEALTH_OPTIONS.keys())}")

    bmi_val = compute_bmi(payload.weight_kg, payload.height_cm)

    # Build feature vector in the exact order the model was trained on
    feature_map = {
        "Age": age_to_bracket(payload.age_years),
        "Sex": 1 if payload.sex == "male" else 0,
        "HighChol": int(payload.high_cholesterol),
        "CholCheck": int(payload.cholesterol_checked_last_5_years),
        "BMI": bmi_val,
        "Smoker": int(payload.smoker),
        "HeartDiseaseorAttack": int(payload.heart_disease_or_attack_history),
        "PhysActivity": int(payload.physically_active_last_30_days),
        "Fruits": int(payload.eats_fruit_daily),
        "Veggies": int(payload.eats_vegetables_daily),
        "HvyAlcoholConsump": int(payload.heavy_alcohol_consumption),
        "GenHlth": GENERAL_HEALTH_OPTIONS[payload.general_health],
        "MentHlth": payload.poor_mental_health_days_last_30,
        "PhysHlth": payload.poor_physical_health_days_last_30,
        "DiffWalk": int(payload.difficulty_walking_or_climbing_stairs),
        "Stroke": int(payload.history_of_stroke),
        "HighBP": int(payload.high_blood_pressure),
    }

    X = pd.DataFrame([feature_map])[diabetes_meta["feature_order"]]

    proba = diabetes_model.predict_proba(X)[0][1]  # probability of class 1 ("at risk")
    label = "at_risk" if proba >= 0.5 else "lower_risk"

    global_importance = list(diabetes_meta["top_features"].keys())[:5]

    return DiabetesRiskOutput(
        risk_label=label,
        risk_probability=round(float(proba), 3),
        bmi=bmi_val,
        global_feature_importance=global_importance,
    )


# ---------------------------------------------------------------------------
# Heart Disease
# ---------------------------------------------------------------------------

class HeartDiseaseInput(BaseModel):
    age_years: int = Field(..., ge=18, le=100)
    sex: str = Field(..., pattern="^(male|female)$")
    resting_blood_pressure: float = Field(..., gt=0, description="Measured mm Hg")
    cholesterol: float = Field(..., gt=0, description="Measured serum cholesterol, mg/dl")
    fasting_blood_sugar_over_120: bool
    max_heart_rate_achieved: float = Field(..., gt=0, description="Measured during a stress test or clinical exercise assessment")
    exercise_induced_angina: bool
    st_depression_exercise: float = Field(..., description="ST depression induced by exercise relative to rest; 0 if unknown")
    major_vessels_colored: int = Field(..., ge=0, le=4, description="from an angiogram; enter only if tested")
    # NOTE: these 4 fields come directly from a cardiac test report (ECG/angiogram).
    # This dataset's category codes do not reliably follow standard textbook
    # ordering (verified empirically against known positive/negative cases),
    # so we pass through the exact numeric code from the report rather than
    # guess an English label that could be wrong for a health context.
    chest_pain_type_code: int = Field(..., ge=0, le=3, description="Chest pain type code exactly as shown on the person's test report/chart (0-3)")
    resting_ecg_code: int = Field(..., ge=0, le=2, description="Resting ECG result code exactly as shown on the report (0-2)")
    st_slope_code: int = Field(..., ge=0, le=2, description="ST segment slope code exactly as shown on the report (0-2)")
    thalassemia_code: int = Field(..., ge=0, le=3, description="Thalassemia test result code exactly as shown on the report (0-3)")


class RiskOutput(BaseModel):
    risk_label: str
    risk_probability: float
    global_feature_importance: list[str] = Field(
        ..., description="Dataset-level feature importance names, not verified causes for this individual prediction."
    )
    imputed_features: list[str] = Field(
        default_factory=list, description="Input features replaced with training medians by the kidney model."
    )
    assumed_defaults: list[str] = Field(
        default_factory=list, description="Categorical input fields assigned a default because no value was supplied."
    )
    disclaimer: str = (
        "This is a statistical risk estimate based on population data, "
        "not a medical diagnosis. Consult a healthcare professional for advice."
    )


@app.post("/predict/heart_disease", response_model=RiskOutput)
def predict_heart_disease(payload: HeartDiseaseInput):
    feature_map = {
        "age": payload.age_years,
        "sex": 1 if payload.sex == "male" else 0,
        "cp": payload.chest_pain_type_code,
        "trestbps": payload.resting_blood_pressure,
        "chol": payload.cholesterol,
        "fbs": int(payload.fasting_blood_sugar_over_120),
        "restecg": payload.resting_ecg_code,
        "thalach": payload.max_heart_rate_achieved,
        "exang": int(payload.exercise_induced_angina),
        "oldpeak": payload.st_depression_exercise,
        "slope": payload.st_slope_code,
        "ca": payload.major_vessels_colored,
        "thal": payload.thalassemia_code,
    }
    X = pd.DataFrame([feature_map])[heart_meta["feature_order"]]
    proba = heart_model.predict_proba(X)[0][1]
    label = "at_risk" if proba >= 0.5 else "lower_risk"
    global_importance = list(heart_meta["top_features"].keys())[:5]

    return RiskOutput(
        risk_label=label,
        risk_probability=round(float(proba), 3),
        global_feature_importance=global_importance,
    )


# ---------------------------------------------------------------------------
# Stroke
# ---------------------------------------------------------------------------

class StrokeInput(BaseModel):
    sex: str = Field(..., pattern="^(male|female)$")
    age_years: int = Field(..., ge=1, le=110)
    has_hypertension: bool
    has_heart_disease: bool
    ever_married: bool
    work_type: str = Field(..., description=f"One of: {list(WORK_TYPE_OPTIONS.keys())}")
    residence_type: str = Field(..., pattern="^(urban|rural)$")
    avg_glucose_level: float = Field(..., gt=0, description="Measured mg/dL from a glucose meter or recent lab")
    weight_kg: float = Field(..., gt=0)
    height_cm: float = Field(..., gt=0)
    is_smoker: bool


@app.post("/predict/stroke", response_model=RiskOutput)
def predict_stroke(payload: StrokeInput):
    if payload.work_type not in WORK_TYPE_OPTIONS:
        raise HTTPException(400, f"Invalid work_type. Options: {list(WORK_TYPE_OPTIONS.keys())}")

    bmi_val = compute_bmi(payload.weight_kg, payload.height_cm)

    feature_map = {
        "sex": 1 if payload.sex == "male" else 0,
        "age": payload.age_years,
        "hypertension": int(payload.has_hypertension),
        "heart_disease": int(payload.has_heart_disease),
        "ever_married": int(payload.ever_married),
        "work_type": WORK_TYPE_OPTIONS[payload.work_type],
        "Residence_type": 1 if payload.residence_type == "urban" else 0,
        "avg_glucose_level": payload.avg_glucose_level,
        "bmi": bmi_val,
        "smoking_status": int(payload.is_smoker),
    }
    X = pd.DataFrame([feature_map])[stroke_meta["feature_order"]]
    proba = stroke_model.predict_proba(X)[0][1]
    label = "at_risk" if proba >= 0.5 else "lower_risk"
    global_importance = list(stroke_meta["top_features"].keys())[:5]

    return RiskOutput(
        risk_label=label,
        risk_probability=round(float(proba), 3),
        global_feature_importance=global_importance,
    )


# ---------------------------------------------------------------------------
# Obesity
# ---------------------------------------------------------------------------

class ObesityInput(BaseModel):
    gender: str = Field(..., pattern="^(male|female)$")
    age_years: int = Field(..., ge=10, le=100)
    family_history_overweight: bool
    frequent_high_calorie_food: bool = Field(..., description="Frequently eats high-calorie food")
    vegetable_frequency: int = Field(..., ge=1, le=3, description="1=rarely, 2=sometimes, 3=every meal")
    main_meals_per_day: int = Field(..., ge=1, le=4)
    eating_between_meals: str = Field(..., pattern="^(no|Sometimes|Frequently|Always)$")
    smokes: bool
    daily_water_intake: int = Field(..., ge=1, le=3, description="1=<1L, 2=1-2L, 3=>2L")
    monitors_calories: bool
    physical_activity_frequency: int = Field(..., ge=0, le=3, description="0=none ... 3=4+ days/week")
    daily_screen_time: int = Field(..., ge=0, le=2, description="0=0-2hrs, 1=3-5hrs, 2=5+hrs")
    alcohol_consumption: str = Field(..., pattern="^(no|Sometimes|Frequently|Always)$")
    main_transport: str = Field(..., description="Public_Transportation, Walking, Automobile, Motorbike, or Bike")


@app.post("/predict/obesity", response_model=RiskOutput)
def predict_obesity(payload: ObesityInput):
    enc = obesity_meta["categorical_encoders"]

    def encode(col, value):
        if value not in enc[col]:
            raise HTTPException(400, f"Invalid value for {col}: {value}. Options: {enc[col]}")
        return enc[col].index(value)

    feature_map = {
        "Gender": encode("Gender", "Male" if payload.gender == "male" else "Female"),
        "Age": payload.age_years,
        "family_history_with_overweight": encode("family_history_with_overweight", "yes" if payload.family_history_overweight else "no"),
        "FAVC": encode("FAVC", "yes" if payload.frequent_high_calorie_food else "no"),
        "FCVC": payload.vegetable_frequency,
        "NCP": payload.main_meals_per_day,
        "CAEC": encode("CAEC", payload.eating_between_meals),
        "SMOKE": encode("SMOKE", "yes" if payload.smokes else "no"),
        "CH2O": payload.daily_water_intake,
        "SCC": encode("SCC", "yes" if payload.monitors_calories else "no"),
        "FAF": payload.physical_activity_frequency,
        "TUE": payload.daily_screen_time,
        "CALC": encode("CALC", payload.alcohol_consumption),
        "MTRANS": encode("MTRANS", payload.main_transport),
    }
    X = pd.DataFrame([feature_map])[obesity_meta["feature_order"]]
    proba = obesity_model.predict_proba(X)[0][1]
    label = "at_risk" if proba >= 0.5 else "lower_risk"
    global_importance = list(obesity_meta["top_features"].keys())[:5]

    return RiskOutput(
        risk_label=label,
        risk_probability=round(float(proba), 3),
        global_feature_importance=global_importance,
    )


# ---------------------------------------------------------------------------
# Kidney Disease
# ---------------------------------------------------------------------------

class KidneyInput(BaseModel):
    # Basic / always self-reportable
    age_years: int = Field(..., ge=1, le=110)
    blood_pressure: float = Field(..., gt=0, description="Measured mm Hg")
    hypertension_history: bool
    diabetes_history: bool
    coronary_artery_disease_history: bool
    appetite: str = Field(..., pattern="^(good|poor)$")
    pedal_edema: bool = Field(..., description="Leg/ankle swelling")
    anemia: bool

    # Optional / advanced - from a recent blood or urine test
    urine_specific_gravity: float | None = Field(None, gt=0, description="Measured lab value; required for this model's minimum input set")
    urine_albumin: float | None = Field(None, ge=0, le=5, description="Lab value 0-5, leave blank if unknown")
    urine_sugar: float | None = Field(None, ge=0, le=5, description="Lab value 0-5, leave blank if unknown")
    urine_rbc: str | None = Field(None, pattern="^(normal|abnormal)$", description="Leave blank if unknown")
    urine_pus_cells: str | None = Field(None, pattern="^(normal|abnormal)$", description="Leave blank if unknown")
    urine_pus_cell_clumps: bool | None = None
    urine_bacteria: bool | None = None
    blood_glucose_random: float | None = Field(None, gt=0, description="Measured mg/dl, leave null if unknown")
    blood_urea: float | None = Field(None, gt=0, description="Measured mg/dl, leave null if unknown")
    serum_creatinine: float | None = Field(None, gt=0, description="Measured mg/dl; required for this model's minimum input set")
    sodium: float | None = Field(None, description="mEq/L, leave blank if unknown")
    potassium: float | None = Field(None, description="mEq/L, leave blank if unknown")
    hemoglobin: float | None = Field(None, gt=0, description="Measured g/dl; required for this model's minimum input set")
    packed_cell_volume: float | None = Field(None, description="%, leave blank if unknown")
    white_cell_count: float | None = Field(None, description="cells/cumm, leave blank if unknown")
    red_cell_count: float | None = Field(None, description="millions/cmm, leave blank if unknown")


@app.post("/predict/kidney_disease", response_model=RiskOutput)
def predict_kidney(payload: KidneyInput):
    required_labs = {
        "serum_creatinine": payload.serum_creatinine,
        "hemoglobin": payload.hemoglobin,
        "urine_specific_gravity": payload.urine_specific_gravity,
    }
    missing_labs = [name for name, value in required_labs.items() if value is None]
    if missing_labs:
        raise HTTPException(
            status_code=422,
            detail=f"Kidney model requires these measured lab values: {', '.join(missing_labs)}",
        )

    enc = kidney_meta["categorical_encoding_maps"]
    numeric_cols = kidney_meta["numeric_cols"]

    numeric_values = {
        "age": payload.age_years, "bp": payload.blood_pressure,
        "sg": payload.urine_specific_gravity, "al": payload.urine_albumin, "su": payload.urine_sugar,
        "bgr": payload.blood_glucose_random, "bu": payload.blood_urea, "sc": payload.serum_creatinine,
        "sod": payload.sodium, "pot": payload.potassium, "hemo": payload.hemoglobin,
        "pcv": payload.packed_cell_volume, "wc": payload.white_cell_count, "rc": payload.red_cell_count,
    }

    feature_map = {col: numeric_values.get(col) for col in numeric_cols}

    assumed_defaults = []

    def encode_cat(col, raw_value, default_key):
        if raw_value is None:
            assumed_defaults.append(f"{col}: assumed {default_key}")
            raw_value = default_key
        mapping = enc[col]
        if raw_value not in mapping:
            raise HTTPException(400, f"Invalid value for {col}: {raw_value}")
        return mapping[raw_value]

    feature_map["rbc"] = encode_cat("rbc", payload.urine_rbc, "normal")
    feature_map["pc"] = encode_cat("pc", payload.urine_pus_cells, "normal")
    pcc_val = None if payload.urine_pus_cell_clumps is None else ("present" if payload.urine_pus_cell_clumps else "notpresent")
    ba_val = None if payload.urine_bacteria is None else ("present" if payload.urine_bacteria else "notpresent")
    feature_map["pcc"] = encode_cat("pcc", pcc_val, "notpresent")
    feature_map["ba"] = encode_cat("ba", ba_val, "notpresent")
    feature_map["htn"] = encode_cat("htn", "yes" if payload.hypertension_history else "no", "no")
    feature_map["dm"] = encode_cat("dm", "yes" if payload.diabetes_history else "no", "no")
    feature_map["cad"] = encode_cat("cad", "yes" if payload.coronary_artery_disease_history else "no", "no")
    feature_map["appet"] = encode_cat("appet", payload.appetite, "good")
    feature_map["pe"] = encode_cat("pe", "yes" if payload.pedal_edema else "no", "no")
    feature_map["ane"] = encode_cat("ane", "yes" if payload.anemia else "no", "no")

    X = pd.DataFrame([feature_map])[kidney_meta["feature_order"]]
    # The training script median-imputed optional CKD fields. Keep the caller's
    # missing values as null in the request and report every median substitution.
    medians = kidney_meta["numeric_medians"]
    imputed_features = []
    for col in kidney_meta["numeric_cols"]:
        if pd.isna(X.at[0, col]):
            X.at[0, col] = medians[col]
            imputed_features.append(kidney_meta.get("feature_descriptions", {}).get(col, col))

    proba = kidney_model.predict_proba(X)[0][1]
    label = "at_risk" if proba >= 0.5 else "lower_risk"
    global_importance = list(kidney_meta["top_features"].keys())[:5]

    return RiskOutput(
        risk_label=label,
        risk_probability=round(float(proba), 3),
        global_feature_importance=global_importance,
        imputed_features=imputed_features,
        assumed_defaults=assumed_defaults,
    )
