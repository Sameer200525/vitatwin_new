"""
Converts real-world user input (actual age, income, education) into the
coded brackets the BRFSS-trained model expects. Keeps the ML layer decoupled
from the UI layer so users never have to know these codes exist.
"""

def age_to_bracket(age_years: int) -> int:
    """BRFSS 13-level age bracket, 1 = 18-24 ... 13 = 80+"""
    brackets = [
        (24, 1), (29, 2), (34, 3), (39, 4), (44, 5), (49, 6),
        (54, 7), (59, 8), (64, 9), (69, 10), (74, 11), (79, 12),
    ]
    for max_age, code in brackets:
        if age_years <= max_age:
            return code
    return 13  # 80+


EDUCATION_OPTIONS = {
    "never_attended_or_kindergarten_only": 1,
    "elementary": 2,
    "some_high_school": 3,
    "high_school_graduate": 4,
    "some_college_or_technical_school": 5,
    "college_graduate": 6,
}

INCOME_OPTIONS = {
    "under_10k": 1,
    "10k_to_15k": 2,
    "15k_to_20k": 3,
    "20k_to_25k": 4,
    "25k_to_35k": 5,
    "35k_to_50k": 6,
    "50k_to_75k": 7,
    "75k_or_more": 8,
}

GENERAL_HEALTH_OPTIONS = {
    "excellent": 1,
    "very_good": 2,
    "good": 3,
    "fair": 4,
    "poor": 5,
}


CHEST_PAIN_OPTIONS = {
    "typical_angina": 0,
    "atypical_angina": 1,
    "non_anginal_pain": 2,
    "asymptomatic": 3,
}

RESTING_ECG_OPTIONS = {
    "normal": 0,
    "st_t_wave_abnormality": 1,
    "left_ventricular_hypertrophy": 2,
}

ST_SLOPE_OPTIONS = {
    "upsloping": 0,
    "flat": 1,
    "downsloping": 2,
}

THALASSEMIA_OPTIONS = {
    "unknown": 0,
    "normal": 1,
    "fixed_defect": 2,
    "reversible_defect": 3,
}

WORK_TYPE_OPTIONS = {
    "government_job": 0,
    "never_worked": 1,
    "private": 2,
    "self_employed": 3,
    "child_not_working": 4,
}


def bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)

