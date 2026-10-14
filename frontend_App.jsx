import React, { useState, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const DISEASES = [
  { id: "diabetes", label: "Diabetes", endpoint: "/predict/diabetes" },
  { id: "heart_disease", label: "Heart Disease", endpoint: "/predict/heart_disease" },
  { id: "stroke", label: "Stroke", endpoint: "/predict/stroke" },
  { id: "obesity", label: "Obesity", endpoint: "/predict/obesity" },
  { id: "kidney_disease", label: "Kidney Disease", endpoint: "/predict/kidney_disease" },
];

const todayStr = () => new Date().toISOString().slice(0, 10);
const daysBetween = (a, b) => Math.round((new Date(b) - new Date(a)) / 86400000);
const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
const bmi = (kg, cm) => (kg && cm ? kg / Math.pow(cm / 100, 2) : null);
const hasNumber = (value) => value !== "" && value != null && Number.isFinite(Number(value));
const numberOrNull = (value) => hasNumber(value) ? Number(value) : null;

// ---------------------------------------------------------------------------
// Persistent storage helpers
// ---------------------------------------------------------------------------
async function loadJSON(key, fallback) {
  try {
    if (window.storage) {
      const res = await window.storage.get(key, false);
      return res ? JSON.parse(res.value) : fallback;
    }
    const value = window.localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}
async function saveJSON(key, value) {
  try {
    if (window.storage) {
      await window.storage.set(key, JSON.stringify(value), false);
    } else {
      window.localStorage.setItem(key, JSON.stringify(value));
    }
    return true;
  } catch {
    return false;
  }
}

const DEFAULT_PROFILE = {
  set_up: false,
  profile_data_confirmed: false,
  age_years: "", sex: "", height_cm: "",
  diagnosed_hypertension: false, diagnosed_high_cholesterol: false, diagnosed_diabetes: false,
  diagnosed_heart_disease: false, diagnosed_stroke_history: false, diagnosed_cad: false,
  diagnosed_anemia: false, cholesterol_checked_last_5_years: true,
  family_history_overweight: false, ever_married: true, work_type: "private",
  residence_type: "urban", main_transport: "Public_Transportation",
  monitors_calories: false, main_meals_per_day: "", smoker: false,
  cholesterol_value: "", known_max_heart_rate: "", fasting_blood_sugar_over_120: "",
  exercise_induced_angina: "", chest_pain_type_code: "", resting_ecg_code: "", st_slope_code: "", thalassemia_code: "",
  major_vessels_colored: "", st_depression_exercise: "", clinical_data_confirmed: false,
  serum_creatinine: "", blood_urea: "", hemoglobin: "", urine_specific_gravity: "", urine_albumin: "",
};

const EMPTY_LOG = {
  date: todayStr(),
  checkin_reviewed: false,
  weight_kg: "", systolic_bp: "", diastolic_bp: "", glucose: "",
  exercised: false, ate_fruit: false, ate_vegetables: false, ate_high_calorie: false,
  water_glasses: "", sleep_hours: "", screen_time_hours: "", mood: "",
  smoked: false, drank_alcohol: false, appetite: "good",
  general_health: "",
  leg_swelling: false, chest_discomfort: false, fatigue_or_dizziness: false, difficulty_walking: false,
  mental_health_not_good: false, physical_health_not_good: false,
};

// ---------------------------------------------------------------------------
// Aggregation: turn 30 days of logs + profile into the 5 backend payloads
// ---------------------------------------------------------------------------
function aggregate(logs) {
  const byDate = new Map();
  logs.filter((l) => l && typeof l.date === "string").forEach((log) => byDate.set(log.date, log));
  const recent = [...byDate.values()]
    .filter((l) => {
      const age = daysBetween(l.date, todayStr());
      return Number.isFinite(age) && age >= 0 && age < 30;
    })
    .sort((a, b) => a.date.localeCompare(b.date));
  const n = Math.max(recent.length, 1);
  const reviewed = recent.filter((log) => log.checkin_reviewed === true);
  const reviewedDenominator = Math.max(reviewed.length, 1);
  const ratio = (pred) => reviewed.filter(pred).length / reviewedDenominator;
  const withField = (f) => [...recent].reverse().find((l) => hasNumber(l[f]));
  const averageField = (f) => {
    const values = recent.map((l) => numberOrNull(l[f])).filter((v) => v != null);
    return values.length ? avg(values) : null;
  };
  const latestBP = [...recent].reverse().find((l) => hasNumber(l.systolic_bp) && hasNumber(l.diastolic_bp)) || null;

  return {
    n: recent.length,
    reviewedCount: reviewed.length,
    smokedDays: reviewed.filter((log) => log.smoked).length,
    activeDays: reviewed.filter((log) => log.exercised).length,
    activeRatio: ratio((l) => l.exercised),
    fruitRatio: ratio((l) => l.ate_fruit),
    vegRatio: ratio((l) => l.ate_vegetables),
    highCalRatio: ratio((l) => l.ate_high_calorie),
    smokedRatio: ratio((l) => l.smoked),
    alcoholRatio: ratio((l) => l.drank_alcohol),
    diffWalkRatio: ratio((l) => l.difficulty_walking),
    legSwellingRatio: ratio((l) => l.leg_swelling),
    fatigueRatio: ratio((l) => l.fatigue_or_dizziness),
    chestPainRatio: ratio((l) => l.chest_discomfort),
    poorAppetiteRatio: ratio((l) => l.appetite === "poor"),
    poorMentalHealthDays: Math.min(reviewed.filter((l) => l.mental_health_not_good).length, 30),
    poorPhysicalHealthDays: Math.min(reviewed.filter((l) => l.physical_health_not_good).length, 30),
    avgWater: averageField("water_glasses"),
    avgScreen: averageField("screen_time_hours"),
    avgMood: averageField("mood"),
    avgGlucose: averageField("glucose"),
    latestGeneralHealth: [...reviewed].reverse().find((log) => ["excellent", "very_good", "good", "fair", "poor"].includes(log.general_health))?.general_health ?? null,
    latestWeight: numberOrNull(withField("weight_kg")?.weight_kg),
    latestBP: latestBP ? { systolic_bp: Number(latestBP.systolic_bp), diastolic_bp: Number(latestBP.diastolic_bp) } : null,
    latestGlucose: numberOrNull(withField("glucose")?.glucose),
  };
}

function buildPayloads(profile, agg) {
  const weight = numberOrNull(agg.latestWeight);
  const height = numberOrNull(profile.height_cm);
  const systolic = agg.latestBP?.systolic_bp ?? null;
  const diastolic = agg.latestBP?.diastolic_bp ?? null;
  const averageGlucose = numberOrNull(agg.avgGlucose);
  const latestGlucose = numberOrNull(agg.latestGlucose);
  const age = numberOrNull(profile.age_years);
  const profileReady = profile.set_up && profile.profile_data_confirmed && age != null && Number.isInteger(age) && age >= 10 && age <= 110 && ["male", "female"].includes(profile.sex);
  const highBP = Boolean(profile.diagnosed_hypertension || (systolic != null && diastolic != null && (systolic >= 130 || diastolic >= 80)));
  const smoker = profile.smoker || agg.smokedRatio >= 0.3;
  const active30 = agg.activeRatio >= 0.3;
  const fruitDaily = agg.fruitRatio >= 0.6;
  const vegDaily = agg.vegRatio >= 0.6;
  const diffWalk = agg.diffWalkRatio >= 0.3;
  const heavyAlcohol = agg.alcoholRatio >= 0.6;

  const ready = agg.n > 0 && profileReady;
  const missing = (fields) => fields.filter(([, value]) => value == null || value === "").map(([label]) => label);
  const canRun = (fields) => ready && missing(fields).length === 0;
  const statuses = {};
  const setStatus = (id, fields, description) => {
    const absent = ready ? missing(fields) : ["a saved profile and a check-in from the last 30 days"];
    statuses[id] = absent.length ? { status: "insufficient_data", message: `Missing: ${[...new Set(absent)].join(", ")}. ${description}` } : { status: "ready", message: "" };
  };

  const commonDemographics = [["age", age], ["sex", profileReady ? profile.sex : null]];
  setStatus("diabetes", [...commonDemographics, ["adult age (18+)", age != null && age >= 18 ? age : null], ["confirmed profile", profile.profile_data_confirmed ? true : null], ["reviewed daily check-in", agg.reviewedCount > 0 ? true : null], ["general-health self-rating", agg.latestGeneralHealth], ["weight from a recent check-in", weight]], "Diabetes assessment needs confirmed adult demographics, a reviewed daily check-in, a general-health response, and a recent weight.");
  setStatus("heart_disease", [
    ...commonDemographics, ["adult age (18-100)", age != null && age >= 18 && age <= 100 ? age : null], ["measured systolic blood pressure", systolic], ["cholesterol from a lab", numberOrNull(profile.cholesterol_value)],
    ["confirmation that cardiac fields are from a clinical report", profile.clinical_data_confirmed ? true : null],
    ["fasting blood sugar status from a lab", profile.fasting_blood_sugar_over_120], ["maximum heart rate from a test", numberOrNull(profile.known_max_heart_rate)],
    ["exercise-induced angina from a clinical assessment", profile.exercise_induced_angina], ["ST depression from a stress test", numberOrNull(profile.st_depression_exercise)],
    ["major vessels from angiography", numberOrNull(profile.major_vessels_colored)], ["chest pain type from a clinical report", numberOrNull(profile.chest_pain_type_code)],
    ["resting ECG result", numberOrNull(profile.resting_ecg_code)], ["ST slope from a stress test", numberOrNull(profile.st_slope_code)],
    ["thalassemia result from a clinical report", numberOrNull(profile.thalassemia_code)],
  ], "Heart disease model requires clinical test data; routine logs and symptoms cannot replace ECG, stress-test, or angiogram results.");
  setStatus("stroke", [...commonDemographics, ["measured glucose in the 30-day window", averageGlucose], ["weight from a recent check-in", weight], ["height", height]], "Stroke assessment needs measured glucose averaged across recent check-ins and a recent weight.");
  const mealsPerDay = numberOrNull(profile.main_meals_per_day);
  const validMealsPerDay = Number.isInteger(mealsPerDay) && mealsPerDay >= 1 && mealsPerDay <= 4 ? mealsPerDay : null;
  setStatus("obesity", [...commonDemographics, ["age in supported range (10-100)", age != null && age <= 100 ? age : null], ["confirmed profile", profile.profile_data_confirmed ? true : null], ["reviewed daily yes/no check-in", agg.reviewedCount > 0 ? true : null], ["main meals per day (1-4)", validMealsPerDay], ["recent water intake", agg.avgWater], ["recent screen time", agg.avgScreen]], "Obesity assessment needs a confirmed profile, a reviewed daily check-in, main meals, water intake, and screen time.");
  setStatus("kidney_disease", [
    ...commonDemographics, ["reviewed daily yes/no check-in", agg.reviewedCount > 0 ? true : null], ["measured blood pressure", systolic], ["serum creatinine lab result", numberOrNull(profile.serum_creatinine)],
    ["hemoglobin lab result", numberOrNull(profile.hemoglobin)], ["urine specific gravity lab result", numberOrNull(profile.urine_specific_gravity)],
  ], "Kidney assessment needs measured blood pressure plus recent kidney-related lab results.");

  const hypertension = agg.latestBP ? {
    status: "available",
    systolic: agg.latestBP.systolic_bp,
    diastolic: agg.latestBP.diastolic_bp,
    diagnosed: Boolean(profile.diagnosed_hypertension),
    elevated: agg.latestBP.systolic_bp >= 130 || agg.latestBP.diastolic_bp >= 80,
    reviewedDays: agg.reviewedCount,
    smokedDays: agg.smokedDays,
    activeDays: agg.activeDays,
  } : {
    status: "insufficient_data",
    diagnosed: Boolean(profile.diagnosed_hypertension),
    reviewedDays: agg.reviewedCount,
    smokedDays: agg.smokedDays,
    activeDays: agg.activeDays,
  };

  return {
    recent_log_count: agg.n,
    statuses,
    hypertension,
    diabetes: canRun([...commonDemographics, ["adult age (18+)", age != null && age >= 18 ? age : null], ["confirmed profile", profile.profile_data_confirmed ? true : null], ["reviewed check-in", agg.reviewedCount > 0 ? true : null], ["general health", agg.latestGeneralHealth], ["weight", weight], ["height", height]]) ? {
      age_years: age, sex: profile.sex, weight_kg: weight, height_cm: height,
      high_blood_pressure: highBP, high_cholesterol: profile.diagnosed_high_cholesterol,
      cholesterol_checked_last_5_years: profile.cholesterol_checked_last_5_years,
      history_of_stroke: profile.diagnosed_stroke_history,
      heart_disease_or_attack_history: profile.diagnosed_heart_disease,
      difficulty_walking_or_climbing_stairs: diffWalk,
      smoker, heavy_alcohol_consumption: heavyAlcohol,
      physically_active_last_30_days: active30,
      eats_fruit_daily: fruitDaily, eats_vegetables_daily: vegDaily,
      general_health: agg.latestGeneralHealth,
      poor_mental_health_days_last_30: agg.poorMentalHealthDays,
      poor_physical_health_days_last_30: agg.poorPhysicalHealthDays,
    } : null,
    heart_disease: canRun([
      ...commonDemographics, ["adult age (18-100)", age != null && age >= 18 && age <= 100 ? age : null], ["measured systolic blood pressure", systolic], ["cholesterol", numberOrNull(profile.cholesterol_value)],
      ["clinical report confirmation", profile.clinical_data_confirmed ? true : null],
      ["fasting blood sugar status", profile.fasting_blood_sugar_over_120], ["maximum heart rate", numberOrNull(profile.known_max_heart_rate)],
      ["exercise-induced angina", profile.exercise_induced_angina], ["ST depression", numberOrNull(profile.st_depression_exercise)],
      ["major vessels", numberOrNull(profile.major_vessels_colored)], ["chest pain type", numberOrNull(profile.chest_pain_type_code)],
      ["resting ECG", numberOrNull(profile.resting_ecg_code)], ["ST slope", numberOrNull(profile.st_slope_code)],
      ["thalassemia", numberOrNull(profile.thalassemia_code)],
    ]) ? {
      age_years: age, sex: profile.sex,
      resting_blood_pressure: systolic, cholesterol: Number(profile.cholesterol_value),
      fasting_blood_sugar_over_120: profile.fasting_blood_sugar_over_120 === "yes",
      max_heart_rate_achieved: Number(profile.known_max_heart_rate),
      exercise_induced_angina: profile.exercise_induced_angina === "yes",
      st_depression_exercise: Number(profile.st_depression_exercise),
      major_vessels_colored: Number(profile.major_vessels_colored),
      chest_pain_type_code: Number(profile.chest_pain_type_code),
      resting_ecg_code: Number(profile.resting_ecg_code),
      st_slope_code: Number(profile.st_slope_code),
      thalassemia_code: Number(profile.thalassemia_code),
    } : null,
    stroke: canRun([...commonDemographics, ["measured glucose in the 30-day window", averageGlucose], ["weight", weight], ["height", height]]) ? {
      sex: profile.sex, age_years: age, has_hypertension: highBP,
      has_heart_disease: profile.diagnosed_heart_disease, ever_married: profile.ever_married,
      work_type: profile.work_type, residence_type: profile.residence_type,
      avg_glucose_level: averageGlucose, weight_kg: weight, height_cm: height, is_smoker: smoker,
    } : null,
    obesity: canRun([...commonDemographics, ["age in supported range (10-100)", age != null && age <= 100 ? age : null], ["confirmed profile", profile.profile_data_confirmed ? true : null], ["reviewed check-in", agg.reviewedCount > 0 ? true : null], ["main meals per day", validMealsPerDay], ["water intake", agg.avgWater], ["screen time", agg.avgScreen]]) ? {
      gender: profile.sex, age_years: age, family_history_overweight: profile.family_history_overweight,
      frequent_high_calorie_food: agg.highCalRatio >= 0.4,
      vegetable_frequency: agg.vegRatio < 0.34 ? 1 : agg.vegRatio < 0.67 ? 2 : 3,
      main_meals_per_day: validMealsPerDay,
      eating_between_meals: agg.highCalRatio >= 0.6 ? "Frequently" : agg.highCalRatio >= 0.2 ? "Sometimes" : "no",
      smokes: smoker,
      daily_water_intake: (agg.avgWater ?? 0) < 4 ? 1 : (agg.avgWater ?? 0) < 8 ? 2 : 3,
      monitors_calories: profile.monitors_calories,
      physical_activity_frequency: agg.activeRatio < 0.15 ? 0 : agg.activeRatio < 0.4 ? 1 : agg.activeRatio < 0.7 ? 2 : 3,
      daily_screen_time: (agg.avgScreen ?? 0) < 3 ? 0 : (agg.avgScreen ?? 0) < 6 ? 1 : 2,
      alcohol_consumption: agg.alcoholRatio >= 0.6 ? "Frequently" : agg.alcoholRatio >= 0.2 ? "Sometimes" : "no",
      main_transport: profile.main_transport,
    } : null,
    kidney_disease: canRun([
      ...commonDemographics, ["reviewed check-in", agg.reviewedCount > 0 ? true : null], ["measured blood pressure", systolic], ["serum creatinine", numberOrNull(profile.serum_creatinine)],
      ["hemoglobin", numberOrNull(profile.hemoglobin)], ["urine specific gravity", numberOrNull(profile.urine_specific_gravity)],
    ]) ? {
      age_years: age, blood_pressure: systolic,
      hypertension_history: highBP, diabetes_history: profile.diagnosed_diabetes,
      coronary_artery_disease_history: profile.diagnosed_cad,
      appetite: agg.poorAppetiteRatio >= 0.4 ? "poor" : "good",
      pedal_edema: agg.legSwellingRatio >= 0.2,
      anemia: profile.diagnosed_anemia,
      serum_creatinine: numberOrNull(profile.serum_creatinine),
      blood_urea: numberOrNull(profile.blood_urea),
      hemoglobin: numberOrNull(profile.hemoglobin),
      urine_specific_gravity: numberOrNull(profile.urine_specific_gravity),
      urine_albumin: numberOrNull(profile.urine_albumin),
      blood_glucose_random: latestGlucose,
    } : null,
  };
}

// ---------------------------------------------------------------------------
// Advice engine — rule-based, tied to each disease's actual top factors
// ---------------------------------------------------------------------------
const ADVICE_RULES = {
  diabetes: [
    { when: (p) => !p.physically_active_last_30_days, tip: "You've logged little activity this month — aim for at least 20-30 minutes of brisk walking on most days; this is one of the strongest levers against diabetes risk." },
    { when: (p) => bmi(p.weight_kg, p.height_cm) >= 27, tip: "Your current BMI is in a higher-risk range — even a 5-7% weight reduction meaningfully lowers diabetes risk." },
    { when: (p) => p.high_blood_pressure, tip: "You reported diagnosed hypertension or a recent reading at the app's elevated threshold — follow up with a clinician and continue monitoring." },
    { when: (p) => !p.eats_vegetables_daily || !p.eats_fruit_daily, tip: "Try to include vegetables or fruit at most meals — fiber-rich foods slow glucose spikes." },
    { when: (p) => p.general_health === "fair" || p.general_health === "poor", tip: "Your overall wellbeing check-ins have been low lately — consider a general checkup to rule out underlying issues." },
  ],
  heart_disease: [
    { when: (p) => p.major_vessels_colored === 0 && false, tip: "" }, // placeholder, angiogram data not actionable via lifestyle
    { when: (p) => p.exercise_induced_angina, tip: "Your clinical record indicates exercise-induced angina — discuss it with a doctor before increasing activity intensity." },
    { when: (p) => p.cholesterol >= 240, tip: "Your cholesterol figure is elevated — reducing saturated fat and increasing fiber can help; consider a lipid panel if you haven't had one recently." },
    { when: (p) => p.resting_blood_pressure >= 130, tip: "Your resting blood pressure is elevated — this is a major heart disease driver. Regular cardio activity and sodium reduction both help." },
    { when: (p) => p.fasting_blood_sugar_over_120, tip: "Elevated blood glucose adds cardiovascular strain on top of any heart risk — worth tracking alongside your diabetes numbers." },
  ],
  stroke: [
    { when: (p) => p.has_hypertension, tip: "Hypertension is the single largest modifiable stroke risk factor — consistent blood pressure control matters more here than almost anything else." },
    { when: (p) => p.avg_glucose_level >= 140, tip: "Your glucose readings run high — poorly controlled blood sugar significantly raises stroke risk over time." },
    { when: (p) => p.is_smoker, tip: "Smoking roughly doubles stroke risk — quitting is the highest-impact single change available to you here." },
    { when: (p) => bmi(p.weight_kg, p.height_cm) >= 30, tip: "Higher BMI compounds stroke risk mainly through blood pressure and glucose — the same activity/diet changes help both." },
  ],
  obesity: [
    { when: (p) => p.physical_activity_frequency <= 1, tip: "You're logging activity on fewer than ~2 days a week — even short daily walks meaningfully shift this risk category over a few months." },
    { when: (p) => p.frequent_high_calorie_food, tip: "High-calorie food is showing up frequently in your log — try swapping in one lower-calorie meal or snack a day rather than an all-at-once overhaul." },
    { when: (p) => p.vegetable_frequency <= 1, tip: "Vegetable intake is low in your recent logs — aim to add a serving to lunch or dinner most days." },
    { when: (p) => p.daily_screen_time >= 2, tip: "High screen time often displaces activity — even standing/moving breaks during screen time help." },
    { when: (p) => p.family_history_overweight, tip: "Family history raises baseline risk — it makes the lifestyle factors above proportionally more protective, not less worth doing." },
  ],
  kidney_disease: [
    { when: (p) => p.hypertension_history, tip: "Hypertension is a leading cause of kidney damage over time — the same blood-pressure control that helps your heart also protects kidney function." },
    { when: (p) => p.diabetes_history, tip: "Diabetes is the other leading cause of CKD — consistent glucose control substantially slows kidney-related decline." },
    { when: (p) => p.pedal_edema, tip: "Leg/ankle swelling can be an early kidney (or heart) signal — worth mentioning to a doctor, especially if it's persistent." },
    { when: (p) => p.anemia, tip: "You reported diagnosed anemia — review it alongside kidney lab results; anemia has many possible causes." },
    { when: (p) => !p.serum_creatinine, tip: "This estimate is based on history alone (no recent labs) — a basic metabolic panel (creatinine, eGFR) would sharpen this considerably." },
  ],
};

function getAdvice(diseaseId, payload) {
  const rules = ADVICE_RULES[diseaseId] || [];
  const tips = rules.filter((r) => r.tip && r.when(payload)).map((r) => r.tip);
  return tips.length ? tips.slice(0, 4) : ["No specific red flags in your recent logs for this condition — keep up your current habits and continue logging."];
}

// ---------------------------------------------------------------------------
// Small UI primitives
// ---------------------------------------------------------------------------
function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}
function NumberInput({ value, onChange, ...props }) {
  return <input type="number" className="ctrl" value={value} onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))} {...props} />;
}
function Select({ value, onChange, options }) {
  return (
    <select className="ctrl" value={value} onChange={(e) => onChange(e.target.value)}>
      {value === "" && <option value="">Select…</option>}
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}
function Toggle({ value, onChange, labelYes = "Yes", labelNo = "No" }) {
  return (
    <div className="toggle" onClick={() => onChange(!value)} role="switch" aria-checked={value}>
      <div className={"toggle-track" + (value ? " on" : "")}><div className="toggle-thumb" /></div>
      <span className="toggle-text">{value ? labelYes : labelNo}</span>
    </div>
  );
}
function RiskGauge({ probability, size = 100 }) {
  const pct = Math.round(probability * 100);
  const color = pct >= 60 ? "var(--risk)" : pct >= 30 ? "var(--warn)" : "var(--safe)";
  const r = size / 2 - 9;
  const circumference = 2 * Math.PI * r;
  const offset = circumference - (pct / 100) * circumference;
  const c = size / 2;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={c} cy={c} r={r} fill="none" stroke="var(--border)" strokeWidth="8" />
      <circle cx={c} cy={c} r={r} fill="none" stroke={color} strokeWidth="8" strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round" transform={`rotate(-90 ${c} ${c})`} style={{ transition: "stroke-dashoffset 0.8s ease" }} />
      <text x={c} y={c + 5} textAnchor="middle" className="gauge-number" fill={color} fontSize={size * 0.2}>{pct}%</text>
    </svg>
  );
}
function PulseHeader() {
  return (
    <div className="pulse-header">
      <svg viewBox="0 0 800 44" preserveAspectRatio="none" className="pulse-svg">
        <path d="M0,22 L180,22 L210,6 L235,38 L260,2 L285,22 L800,22" fill="none" stroke="var(--accent)" strokeWidth="2" className="pulse-path" />
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Profile screen
// ---------------------------------------------------------------------------
function ProfileScreen({ profile, setProfile, onSave, saving }) {
  const set = (k) => (v) => setProfile({ ...profile, [k]: v });
  return (
    <div>
      <div className="section-intro">
        These details change rarely — set them up once. Edit any time from this tab.
      </div>
      <div className="form-grid">
        <div className="form-section-title">About you</div>
        <Field label="Age (years)"><NumberInput value={profile.age_years} onChange={set("age_years")} min={10} max={110} step={1} /></Field>
        <Field label="Sex"><Select value={profile.sex} onChange={set("sex")} options={[{ value: "male", label: "Male" }, { value: "female", label: "Female" }]} /></Field>
        <Field label="Height (cm)"><NumberInput value={profile.height_cm} onChange={set("height_cm")} min={100} max={230} /></Field>
        <Field label="I reviewed and confirmed all profile answers"><Toggle value={profile.profile_data_confirmed} onChange={set("profile_data_confirmed")} /></Field>
        <Field label="Smoker"><Toggle value={profile.smoker} onChange={set("smoker")} /></Field>
        <Field label="Ever married"><Toggle value={profile.ever_married} onChange={set("ever_married")} /></Field>
        <Field label="Work type"><Select value={profile.work_type} onChange={set("work_type")} options={["government_job", "never_worked", "private", "self_employed", "child_not_working"].map((v) => ({ value: v, label: v.replace(/_/g, " ") }))} /></Field>
        <Field label="Residence"><Select value={profile.residence_type} onChange={set("residence_type")} options={[{ value: "urban", label: "Urban" }, { value: "rural", label: "Rural" }]} /></Field>
        <Field label="Main transport"><Select value={profile.main_transport} onChange={set("main_transport")} options={["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"].map((v) => ({ value: v, label: v.replace("_", " ") }))} /></Field>
        <Field label="Monitors calorie intake"><Toggle value={profile.monitors_calories} onChange={set("monitors_calories")} /></Field>
        <Field label="Main meals per day"><NumberInput value={profile.main_meals_per_day} onChange={set("main_meals_per_day")} min={1} max={4} step={1} /></Field>
        <Field label="Family history of overweight"><Toggle value={profile.family_history_overweight} onChange={set("family_history_overweight")} /></Field>

        <div className="form-section-title">Diagnosed conditions</div>
        <Field label="Hypertension"><Toggle value={profile.diagnosed_hypertension} onChange={set("diagnosed_hypertension")} /></Field>
        <Field label="High cholesterol"><Toggle value={profile.diagnosed_high_cholesterol} onChange={set("diagnosed_high_cholesterol")} /></Field>
        <Field label="Diabetes"><Toggle value={profile.diagnosed_diabetes} onChange={set("diagnosed_diabetes")} /></Field>
        <Field label="Heart disease"><Toggle value={profile.diagnosed_heart_disease} onChange={set("diagnosed_heart_disease")} /></Field>
        <Field label="Stroke history"><Toggle value={profile.diagnosed_stroke_history} onChange={set("diagnosed_stroke_history")} /></Field>
        <Field label="Coronary artery disease"><Toggle value={profile.diagnosed_cad} onChange={set("diagnosed_cad")} /></Field>
        <Field label="Anemia"><Toggle value={profile.diagnosed_anemia} onChange={set("diagnosed_anemia")} /></Field>
        <Field label="Cholesterol checked (5yrs)"><Toggle value={profile.cholesterol_checked_last_5_years} onChange={set("cholesterol_checked_last_5_years")} /></Field>

        <div className="form-section-title">Optional — from a recent lab report (leave blank if unknown)</div>
        <Field label="Cholesterol (mg/dl)"><NumberInput value={profile.cholesterol_value} onChange={set("cholesterol_value")} min={100} max={600} /></Field>
        <Field label="Maximum heart rate (test report)"><NumberInput value={profile.known_max_heart_rate} onChange={set("known_max_heart_rate")} min={60} max={220} /></Field>
        <Field label="Fasting blood sugar >120 (lab)" hint="Unknown is not treated as No"><Select value={profile.fasting_blood_sugar_over_120} onChange={set("fasting_blood_sugar_over_120")} options={[{ value: "yes", label: "Yes" }, { value: "no", label: "No" }]} /></Field>
        <Field label="Serum creatinine (mg/dl)"><NumberInput value={profile.serum_creatinine} onChange={set("serum_creatinine")} min={0} max={20} step={0.1} /></Field>
        <Field label="Blood urea (mg/dl)"><NumberInput value={profile.blood_urea} onChange={set("blood_urea")} min={0} max={400} /></Field>
        <Field label="Hemoglobin (g/dl)"><NumberInput value={profile.hemoglobin} onChange={set("hemoglobin")} min={0} max={20} step={0.1} /></Field>
        <Field label="Urine specific gravity"><NumberInput value={profile.urine_specific_gravity} onChange={set("urine_specific_gravity")} min={1} max={1.1} step={0.001} /></Field>
        <Field label="Urine albumin (0-5)"><NumberInput value={profile.urine_albumin} onChange={set("urine_albumin")} min={0} max={5} step={1} /></Field>

        <div className="form-section-title">Heart model clinical data — enter only values from a clinician's test report</div>
          <Field label="I confirmed these values are from a clinical report"><Toggle value={profile.clinical_data_confirmed} onChange={set("clinical_data_confirmed")} /></Field>
        <Field label="Exercise-induced angina" hint="Not inferred from daily chest-discomfort logs"><Select value={profile.exercise_induced_angina} onChange={set("exercise_induced_angina")} options={[{ value: "yes", label: "Yes" }, { value: "no", label: "No" }]} /></Field>
        <Field label="Chest pain type code (0-3)"><NumberInput value={profile.chest_pain_type_code} onChange={set("chest_pain_type_code")} min={0} max={3} step={1} /></Field>
        <Field label="Resting ECG code (0-2)"><NumberInput value={profile.resting_ecg_code} onChange={set("resting_ecg_code")} min={0} max={2} step={1} /></Field>
        <Field label="ST slope code (0-2)"><NumberInput value={profile.st_slope_code} onChange={set("st_slope_code")} min={0} max={2} step={1} /></Field>
        <Field label="Thalassemia code (0-3)"><NumberInput value={profile.thalassemia_code} onChange={set("thalassemia_code")} min={0} max={3} step={1} /></Field>
        <Field label="Major vessels colored (0-4)"><NumberInput value={profile.major_vessels_colored} onChange={set("major_vessels_colored")} min={0} max={4} step={1} /></Field>
      </div>
      <button className="vt-run-btn" onClick={onSave} disabled={saving}>{saving ? "Saving…" : "Save Profile"}</button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Daily check-in screen
// ---------------------------------------------------------------------------
function CheckinScreen({ log, setLog, onSave, saving, alreadyLoggedToday }) {
  const set = (k) => (v) => setLog({ ...log, [k]: v });
  return (
    <div>
      <div className="section-intro">
        {alreadyLoggedToday
          ? "You've already logged today — saving again will update today's entry."
          : "Takes under a minute. The more you log, the more accurate your risk profile becomes."}
      </div>
      <div className="form-grid">
        <div className="form-section-title">Body & vitals</div>
        <Field label="Weight (kg)"><NumberInput value={log.weight_kg} onChange={set("weight_kg")} min={20} max={300} /></Field>
        <Field label="Systolic BP" hint="if measured today"><NumberInput value={log.systolic_bp} onChange={set("systolic_bp")} min={70} max={220} /></Field>
        <Field label="Diastolic BP" hint="if measured today"><NumberInput value={log.diastolic_bp} onChange={set("diastolic_bp")} min={40} max={140} /></Field>
        <Field label="Blood glucose (mg/dl)" hint="if measured today"><NumberInput value={log.glucose} onChange={set("glucose")} min={40} max={500} /></Field>

        <div className="form-section-title">Activity & diet</div>
        <Field label="Exercised today"><Toggle value={log.exercised} onChange={set("exercised")} /></Field>
        <Field label="Ate fruit today"><Toggle value={log.ate_fruit} onChange={set("ate_fruit")} /></Field>
        <Field label="Ate vegetables today"><Toggle value={log.ate_vegetables} onChange={set("ate_vegetables")} /></Field>
        <Field label="Ate high-calorie/junk food"><Toggle value={log.ate_high_calorie} onChange={set("ate_high_calorie")} /></Field>
        <Field label="Water (glasses)"><NumberInput value={log.water_glasses} onChange={set("water_glasses")} min={0} max={20} /></Field>

        <div className="form-section-title">Lifestyle</div>
        <Field label="Sleep (hours)"><NumberInput value={log.sleep_hours} onChange={set("sleep_hours")} min={0} max={14} step={0.5} /></Field>
        <Field label="Screen time (hours)"><NumberInput value={log.screen_time_hours} onChange={set("screen_time_hours")} min={0} max={20} /></Field>
        <Field label="Smoked today"><Toggle value={log.smoked} onChange={set("smoked")} /></Field>
        <Field label="Drank alcohol today"><Toggle value={log.drank_alcohol} onChange={set("drank_alcohol")} /></Field>

        <div className="form-section-title">How you're feeling</div>
        <Field label="Mood" hint="1=rough, 5=great"><NumberInput value={log.mood} onChange={set("mood")} min={1} max={5} /></Field>
        <Field label="Overall health"><Select value={log.general_health} onChange={set("general_health")} options={[{ value: "excellent", label: "Excellent" }, { value: "very_good", label: "Very good" }, { value: "good", label: "Good" }, { value: "fair", label: "Fair" }, { value: "poor", label: "Poor" }]} /></Field>
        <Field label="Mental health not good today"><Toggle value={log.mental_health_not_good} onChange={set("mental_health_not_good")} /></Field>
        <Field label="Physical health not good today"><Toggle value={log.physical_health_not_good} onChange={set("physical_health_not_good")} /></Field>
        <Field label="Appetite"><Select value={log.appetite} onChange={set("appetite")} options={[{ value: "good", label: "Good" }, { value: "poor", label: "Poor" }]} /></Field>
        <Field label="Leg/ankle swelling"><Toggle value={log.leg_swelling} onChange={set("leg_swelling")} /></Field>
        <Field label="Chest discomfort"><Toggle value={log.chest_discomfort} onChange={set("chest_discomfort")} /></Field>
        <Field label="Fatigue / dizziness"><Toggle value={log.fatigue_or_dizziness} onChange={set("fatigue_or_dizziness")} /></Field>
        <Field label="Difficulty walking/stairs"><Toggle value={log.difficulty_walking} onChange={set("difficulty_walking")} /></Field>
        <Field label="I reviewed today's yes/no answers" hint="Leave No only if No is accurate for today"><Toggle value={log.checkin_reviewed} onChange={set("checkin_reviewed")} /></Field>
      </div>
      <button className="vt-run-btn" onClick={onSave} disabled={saving}>{saving ? "Saving…" : "Save Today's Entry"}</button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dashboard screen
// ---------------------------------------------------------------------------
function DashboardScreen({ logs, results, statuses, onRunAssessment, running, payloads, lastRun }) {
  if (logs.length === 0) {
    return <div className="section-intro">No daily entries yet — log at least one day's check-in to generate your first risk assessment.</div>;
  }
  const atRisk = DISEASES.filter((d) => results[d.id]?.risk_label === "at_risk");

  return (
    <div>
      <div className="ov-summary">
        <div className="ov-summary-title">Digital Twin Status</div>
        <div className="ov-empty">
          {payloads?.recent_log_count ?? logs.filter((l) => {
            const age = daysBetween(l.date, todayStr());
            return Number.isFinite(age) && age >= 0 && age < 30;
          }).length} recent day{(payloads?.recent_log_count ?? logs.filter((l) => {
            const age = daysBetween(l.date, todayStr());
            return Number.isFinite(age) && age >= 0 && age < 30;
          }).length) !== 1 ? "s" : ""} logged (30-day window)
          {lastRun && ` · last assessed ${new Date(lastRun).toLocaleString()}`}
        </div>
        <div className="estimate-note">Medical measurements are never filled with typical-value estimates. Assessments run only when their required inputs are available.</div>
        <button className="vt-run-btn" style={{ marginTop: 14 }} onClick={onRunAssessment} disabled={running}>
          {running ? "Assessing available models…" : "Run Risk Assessment"}
        </button>
        <div className="ov-card" style={{ marginTop: 14 }}>
          <div style={{ flex: 1 }}>
            <div className="ov-card-title">Hypertension measurement check</div>
            <div className="ov-card-sub">Rule-based assessment — not an ML prediction or diagnosis</div>
            {payloads?.hypertension?.status === "available" ? (
              <div className="advice-item" style={{ marginTop: 8 }}>
                Latest paired reading: {payloads.hypertension.systolic}/{payloads.hypertension.diastolic} mmHg. {payloads.hypertension.elevated
                  ? "This reading meets the app's elevated-measurement flag (130 systolic or 80 diastolic and above); repeat measurements and clinician advice are needed."
                  : "This reading is below the app's 130/80 flag; a single reading cannot rule out hypertension."}
                {payloads.hypertension.diagnosed && " You also reported a prior hypertension diagnosis."}
              </div>
            ) : (
              <div className="advice-item" style={{ marginTop: 8 }}>
                {payloads?.hypertension?.diagnosed
                  ? "You reported a prior diagnosis, but there is no recent paired blood-pressure measurement to assess."
                  : "Insufficient data: enter a measured systolic and diastolic reading in a recent check-in."}
              </div>
            )}
            <div className="advice-item" style={{ marginTop: 8 }}>
              {payloads?.hypertension?.reviewedDays > 0
                ? `Reported lifestyle log (not an estimate): smoking on ${payloads.hypertension.smokedDays} of ${payloads.hypertension.reviewedDays} reviewed days; activity on ${payloads.hypertension.activeDays} of those days.`
                : "No reviewed daily yes/no check-ins yet, so no lifestyle observations are available."}
            </div>
          </div>
        </div>
      </div>

      {(Object.keys(results).length > 0 || Object.keys(statuses || {}).length > 0) && (
        <>
          {atRisk.length > 0 ? (
            <div className="alert-banner">
              ⚠ Elevated risk detected: {atRisk.map((d) => d.label).join(", ")}. See details and advice below.
            </div>
          ) : (DISEASES.every((d) => results[d.id]) ? (
            <div className="alert-banner safe">✓ No elevated risk detected across the five completed model assessments. This is not a medical clearance.</div>
          ) : (
            <div className="alert-banner">Some model assessments could not run. Review each status below; unavailable results are not considered low risk.</div>
          ))}

          <div className="overview-grid">
            {DISEASES.map((d) => {
              const r = results[d.id];
              const status = statuses?.[d.id];
              return (
                <div className="ov-card" key={d.id}>
                  {r && <RiskGauge probability={r.risk_probability} size={72} />}
                  <div style={{ flex: 1 }}>
                    <div className="ov-card-title">{d.label}</div>
                    {r ? (
                      <>
                        <div className="ov-card-sub" style={{ color: r.risk_label === "at_risk" ? "var(--risk)" : "var(--safe)" }}>
                          {r.risk_label === "at_risk" ? "Elevated model estimate" : "Lower model estimate"}
                        </div>
                        {(r.imputed_features?.length > 0 || r.assumed_defaults?.length > 0) && (
                          <div className="advice-item" style={{ marginTop: 6 }}>
                            The kidney model used median numeric values for: {(r.imputed_features || []).join(", ") || "none"}. Missing categorical fields assumed: {(r.assumed_defaults || []).join(", ") || "none"}.
                          </div>
                        )}
                        {r.risk_label === "at_risk" && (
                          <div className="advice-list">
                            <div className="advice-item">Suggestions below are based on reported inputs, not individually verified causes of the model result.</div>
                            {getAdvice(d.id, payloads[d.id]).map((tip, i) => <div className="advice-item" key={i}>• {tip}</div>)}
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="advice-item" style={{ marginTop: 6 }}>{status?.message || "No result available."}</div>
                    )}
                    {status?.status === "error" && <div className="vt-error">Request failed: {status.message}</div>}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="disclaimer" style={{ marginTop: 18 }}>
            These are statistical estimates from population data, not medical diagnoses. Consult a healthcare professional for advice, especially for any flagged condition.
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// History screen
// ---------------------------------------------------------------------------
function HistoryScreen({ logs }) {
  if (logs.length === 0) return <div className="section-intro">No entries logged yet.</div>;
  const sorted = [...logs].sort((a, b) => (a.date < b.date ? 1 : -1));
  return (
    <div className="history-table">
      <div className="history-row history-head">
        <span>Date</span><span>Weight</span><span>BP</span><span>Glucose</span><span>Mood</span><span>Active</span>
      </div>
      {sorted.map((l) => (
        <div className="history-row" key={l.date}>
          <span>{l.date}</span>
          <span>{l.weight_kg || "—"}</span>
          <span>{hasNumber(l.systolic_bp) && hasNumber(l.diastolic_bp) ? `${l.systolic_bp}/${l.diastolic_bp}` : "—"}</span>
          <span>{l.glucose || "—"}</span>
          <span>{l.mood}/5</span>
          <span>{l.exercised ? "✓" : "—"}</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main app
// ---------------------------------------------------------------------------
export default function VitaTwinApp() {
  const [tab, setTab] = useState("dashboard");
  const [profile, setProfile] = useState(DEFAULT_PROFILE);
  const [logs, setLogs] = useState([]);
  const [todayLog, setTodayLog] = useState(EMPTY_LOG);
  const [results, setResults] = useState({});
  const [statuses, setStatuses] = useState({});
  const [payloads, setPayloads] = useState({});
  const [lastRun, setLastRun] = useState(null);
  const [loaded, setLoaded] = useState(false);
  const [savingProfile, setSavingProfile] = useState(false);
  const [savingLog, setSavingLog] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    (async () => {
      const storedProfile = await loadJSON("profile", DEFAULT_PROFILE);
      const p = { ...DEFAULT_PROFILE, ...(storedProfile && typeof storedProfile === "object" ? storedProfile : {}) };
      const storedLogs = await loadJSON("daily_logs", []);
      const l = Array.isArray(storedLogs) ? storedLogs.filter((entry) => entry && typeof entry.date === "string") : [];
      const cached = await loadJSON("last_assessment", null);
      setProfile(p);
      setLogs(l);
      const existingToday = l.find((e) => e.date === todayStr());
      if (existingToday) setTodayLog(existingToday);
      if (cached?.assessment_version === 2) {
        setResults(cached.results || {});
        setPayloads(cached.payloads || {});
        setStatuses(cached.statuses || cached.payloads?.statuses || {});
        setLastRun(cached.timestamp || null);
      }
      if (!p.set_up) setTab("profile");
      setLoaded(true);
    })();
  }, []);

  const saveProfile = async () => {
    setSavingProfile(true);
    const updated = { ...profile, set_up: true };
    const saved = await saveJSON("profile", updated);
    if (saved) {
      setProfile(updated);
      setResults({});
      setStatuses({});
      setPayloads({});
      setLastRun(null);
      await saveJSON("last_assessment", null);
      setError(null);
      setTab("checkin");
    } else {
      setError("Your profile could not be saved in this browser. Check storage settings and try again.");
    }
    setSavingProfile(false);
  };

  const saveLog = async () => {
    setSavingLog(true);
    setError(null);
    const others = logs.filter((l) => l.date !== todayLog.date);
    const newLogs = [...others, todayLog];
    const saved = await saveJSON("daily_logs", newLogs);
    if (saved) {
      setLogs(newLogs);
      setResults({});
      setStatuses({});
      setPayloads({});
      setLastRun(null);
      await saveJSON("last_assessment", null);
      setTab("dashboard");
    } else {
      setError("Today's check-in could not be saved in this browser. Your entry has not been recorded.");
    }
    setSavingLog(false);
  };

  const runAssessment = async () => {
    setRunning(true);
    setError(null);
    try {
      const agg = aggregate(logs);
      const built = buildPayloads(profile, agg);
      setPayloads(built);
      const newResults = {};
      const newStatuses = { ...built.statuses };
      const requests = DISEASES.filter((d) => built[d.id]).map(async (d) => {
        try {
          const res = await fetch(`${API_BASE}${d.endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(built[d.id]),
          });
          if (!res.ok) {
            const detail = await res.text();
            throw new Error(`${res.status}: ${detail}`);
          }
          return { id: d.id, result: await res.json() };
        } catch (e) {
          return { id: d.id, error: e.message || "Network request failed" };
        }
      });
      for (const outcome of await Promise.all(requests)) {
        if (outcome.result) {
          newResults[outcome.id] = outcome.result;
          newStatuses[outcome.id] = { status: "success", message: "Assessment completed." };
        } else {
          newStatuses[outcome.id] = { status: "error", message: outcome.error };
        }
      }
      setResults(newResults);
      setStatuses(newStatuses);
      const ts = new Date().toISOString();
      setLastRun(ts);
      const saved = await saveJSON("last_assessment", { assessment_version: 2, results: newResults, statuses: newStatuses, payloads: built, timestamp: ts });
      if (!saved) setError("Assessments completed, but the results could not be saved in this browser.");
    } catch (e) {
      setError(`Unable to prepare the assessments: ${e.message}`);
    } finally {
      setRunning(false);
    }
  };

  if (!loaded) return <div className="vt-root"><div className="vt-body">Loading your digital twin…</div></div>;

  const alreadyLoggedToday = logs.some((l) => l.date === todayStr());

  return (
    <div className="vt-root">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');
        .vt-root {
          --bg:#0B1414; --surface:#101C1C; --surface2:#16302F; --text:#E9EAE2; --text-dim:#8FA8A2;
          --accent:#FF6B4A; --safe:#4ADE80; --warn:#FBBF24; --risk:#FB5B48; --border:#21302F;
          font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); min-height:100%;
          border-radius:12px; overflow:hidden;
        }
        .vt-root * { box-sizing:border-box; }
        .pulse-header { width:100%; height:40px; overflow:hidden; background:var(--surface); border-bottom:1px solid var(--border); }
        .pulse-svg { width:100%; height:100%; }
        .pulse-path { stroke-dasharray:900; stroke-dashoffset:900; animation:pulse-draw 3.2s ease-in-out infinite; }
        @keyframes pulse-draw { 0%{stroke-dashoffset:900;opacity:.3} 50%{stroke-dashoffset:0;opacity:1} 100%{stroke-dashoffset:-900;opacity:.3} }
        .vt-header { padding:18px 26px 14px; }
        .vt-title { font-family:'Space Grotesk',sans-serif; font-size:24px; font-weight:700; letter-spacing:-.02em; margin:0; }
        .vt-title span { color:var(--accent); }
        .vt-subtitle { color:var(--text-dim); font-size:12.5px; margin-top:3px; }
        .vt-tabs { display:flex; gap:4px; padding:0 26px; border-bottom:1px solid var(--border); overflow-x:auto; }
        .vt-tab { font-family:'Space Grotesk',sans-serif; font-size:13px; font-weight:600; padding:10px 14px; cursor:pointer; color:var(--text-dim); border-bottom:2px solid transparent; white-space:nowrap; }
        .vt-tab.active { color:var(--accent); border-bottom-color:var(--accent); }
        .vt-tab:hover { color:var(--text); }
        .vt-body { padding:22px 26px 30px; }
        .section-intro { color:var(--text-dim); font-size:13px; margin-bottom:18px; line-height:1.5; }
        .form-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px 18px; margin-bottom:20px; }
        .form-section-title { grid-column:1/-1; font-family:'Space Grotesk',sans-serif; font-size:11.5px; text-transform:uppercase; letter-spacing:.08em; color:var(--accent); margin:8px 0 -2px; padding-top:10px; border-top:1px solid var(--border); }
        .form-section-title:first-child { border-top:none; padding-top:0; margin-top:0; }
        .field { display:flex; flex-direction:column; gap:5px; }
        .field-label { font-size:12px; color:var(--text-dim); font-weight:500; }
        .field-hint { font-size:10.5px; color:#5C726C; }
        .ctrl { background:var(--surface2); border:1px solid var(--border); color:var(--text); border-radius:7px; padding:8px 10px; font-size:13px; font-family:'Inter',sans-serif; outline:none; }
        .ctrl:focus { border-color:var(--accent); }
        .toggle { display:flex; align-items:center; gap:8px; cursor:pointer; padding:4px 0; }
        .toggle-track { width:34px; height:19px; border-radius:20px; background:var(--surface2); border:1px solid var(--border); position:relative; transition:background .2s; }
        .toggle-track.on { background:var(--accent); border-color:var(--accent); }
        .toggle-thumb { width:13px; height:13px; border-radius:50%; background:var(--text); position:absolute; top:2px; left:2px; transition:transform .2s; }
        .toggle-track.on .toggle-thumb { transform:translateX(15px); }
        .toggle-text { font-size:12.5px; color:var(--text-dim); }
        .vt-run-btn { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:13.5px; background:var(--accent); color:#100806; border:none; border-radius:8px; padding:11px 20px; cursor:pointer; }
        .vt-run-btn:hover { filter:brightness(1.08); }
        .vt-run-btn:disabled { opacity:.5; cursor:not-allowed; }
        .vt-error { background:rgba(251,91,72,.1); border:1px solid var(--risk); color:#FFB3A5; padding:11px 13px; border-radius:8px; font-size:12px; margin-top:12px; line-height:1.5; }
        .ov-summary { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:18px; margin-bottom:18px; }
        .ov-summary-title { font-family:'Space Grotesk',sans-serif; font-size:14px; font-weight:600; margin-bottom:4px; }
        .ov-empty { color:var(--text-dim); font-size:12px; }
        .estimate-note { font-size:11px; color:var(--warn); margin-top:8px; line-height:1.5; }
        .alert-banner { background:rgba(251,91,72,.12); border:1px solid var(--risk); color:#FFB3A5; padding:12px 16px; border-radius:8px; font-size:13px; margin-bottom:18px; font-weight:500; }
        .alert-banner.safe { background:rgba(74,222,128,.1); border-color:var(--safe); color:#B4F5CB; }
        .overview-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }
        .ov-card { background:var(--surface2); border:1px solid var(--border); border-radius:10px; padding:16px; display:flex; gap:14px; align-items:flex-start; }
        .ov-card-title { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:14.5px; }
        .ov-card-sub { font-size:11.5px; margin-top:2px; font-weight:600; }
        .advice-list { margin-top:10px; }
        .advice-item { font-size:11.5px; color:var(--text-dim); line-height:1.6; }
        .gauge-number { font-family:'JetBrains Mono',monospace; font-weight:700; }
        .disclaimer { font-size:11px; color:#5C726C; line-height:1.5; }
        .history-table { display:flex; flex-direction:column; gap:2px; font-size:12.5px; }
        .history-row { display:grid; grid-template-columns:1.4fr 1fr 1fr 1fr .8fr .8fr; padding:9px 12px; background:var(--surface2); border-radius:6px; }
        .history-head { color:var(--text-dim); font-family:'Space Grotesk',sans-serif; font-size:11px; text-transform:uppercase; letter-spacing:.05em; background:transparent; }
      `}</style>

      <PulseHeader />
      <div className="vt-header">
        <h1 className="vt-title">Vita<span>Twin</span></h1>
        <div className="vt-subtitle">Log daily → review measurement-based insights and available risk assessments</div>
      </div>

      <div className="vt-tabs">
        <div className={"vt-tab" + (tab === "dashboard" ? " active" : "")} onClick={() => setTab("dashboard")}>Dashboard</div>
        <div className={"vt-tab" + (tab === "checkin" ? " active" : "")} onClick={() => setTab("checkin")}>
          Daily Check-in {!alreadyLoggedToday && "●"}
        </div>
        <div className={"vt-tab" + (tab === "profile" ? " active" : "")} onClick={() => setTab("profile")}>Profile</div>
        <div className={"vt-tab" + (tab === "history" ? " active" : "")} onClick={() => setTab("history")}>History</div>
      </div>

      <div className="vt-body">
        {error && <div className="vt-error" role="alert">{error}</div>}
        {tab === "dashboard" && (
          <DashboardScreen logs={logs} results={results} statuses={statuses} payloads={payloads} lastRun={lastRun}
            onRunAssessment={runAssessment} running={running} />
        )}
        {tab === "checkin" && (
          <CheckinScreen log={todayLog} setLog={setTodayLog} onSave={saveLog} saving={savingLog} alreadyLoggedToday={alreadyLoggedToday} />
        )}
        {tab === "profile" && (
          <ProfileScreen profile={profile} setProfile={setProfile} onSave={saveProfile} saving={savingProfile} />
        )}
        {tab === "history" && <HistoryScreen logs={logs} />}
      </div>
    </div>
  );
}
