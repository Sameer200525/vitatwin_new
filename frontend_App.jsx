import React, { useState, useEffect, useCallback } from "react";

const API_BASE = "http://localhost:8000";

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
  age_years: 35, sex: "male", height_cm: 175,
  diagnosed_hypertension: false, diagnosed_high_cholesterol: false, diagnosed_diabetes: false,
  diagnosed_heart_disease: false, diagnosed_stroke_history: false, diagnosed_cad: false,
  diagnosed_anemia: false, cholesterol_checked_last_5_years: true,
  family_history_overweight: false, ever_married: true, work_type: "private",
  residence_type: "urban", main_transport: "Public_Transportation",
  monitors_calories: false, main_meals_per_day: 3, smoker: false,
  cholesterol_value: "", known_max_heart_rate: "",
  chest_pain_type_code: 0, resting_ecg_code: 0, st_slope_code: 1, thalassemia_code: 1,
  major_vessels_colored: 0, st_depression_exercise: 0,
  serum_creatinine: "", blood_urea: "", hemoglobin: "", urine_specific_gravity: "", urine_albumin: "",
};

const EMPTY_LOG = {
  date: todayStr(),
  weight_kg: "", systolic_bp: "", diastolic_bp: "", glucose: "",
  exercised: false, ate_fruit: false, ate_vegetables: false, ate_high_calorie: false,
  water_glasses: 6, sleep_hours: 7, screen_time_hours: 3, mood: 3,
  smoked: false, drank_alcohol: false, appetite: "good",
  leg_swelling: false, chest_discomfort: false, fatigue_or_dizziness: false, difficulty_walking: false,
};

// ---------------------------------------------------------------------------
// Aggregation: turn 30 days of logs + profile into the 5 backend payloads
// ---------------------------------------------------------------------------
function aggregate(logs) {
  const recent = logs.filter((l) => daysBetween(l.date, todayStr()) <= 30);
  const n = Math.max(recent.length, 1);
  const ratio = (pred) => recent.filter(pred).length / n;
  const withField = (f) => [...recent].reverse().find((l) => l[f] !== "" && l[f] != null);

  return {
    n: recent.length,
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
    poorMoodDays: Math.min(recent.filter((l) => l.mood <= 2).length, 30),
    symptomDays: Math.min(recent.filter((l) => l.leg_swelling || l.chest_discomfort || l.fatigue_or_dizziness || l.difficulty_walking).length, 30),
    avgWater: avg(recent.map((l) => Number(l.water_glasses) || 0)),
    avgScreen: avg(recent.map((l) => Number(l.screen_time_hours) || 0)),
    avgMood: avg(recent.map((l) => Number(l.mood) || 3)),
    latestWeight: withField("weight_kg")?.weight_kg ?? null,
    latestBP: withField("systolic_bp"),
    latestGlucose: withField("glucose"),
  };
}

function generalHealthFromMood(m) {
  if (m >= 4.5) return "excellent";
  if (m >= 3.5) return "very_good";
  if (m >= 2.5) return "good";
  if (m >= 1.5) return "fair";
  return "poor";
}

function buildPayloads(profile, agg) {
  const weight = Number(agg.latestWeight) || 70;
  const height = Number(profile.height_cm) || 170;
  const systolic = Number(agg.latestBP?.systolic_bp) || 118;
  const diastolic = Number(agg.latestBP?.diastolic_bp) || 78;
  const glucose = Number(agg.latestGlucose?.glucose) || 100;
  const age = Number(profile.age_years) || 30;

  const highBP = profile.diagnosed_hypertension || systolic >= 130 || diastolic >= 80;
  const smoker = profile.smoker || agg.smokedRatio >= 0.3;
  const active30 = agg.activeRatio >= 0.3;
  const fruitDaily = agg.fruitRatio >= 0.6;
  const vegDaily = agg.vegRatio >= 0.6;
  const diffWalk = agg.diffWalkRatio >= 0.3;
  const heavyAlcohol = agg.alcoholRatio >= 0.6;

  const usedEstimates = [];
  if (!agg.latestBP) usedEstimates.push("blood pressure");
  if (!agg.latestGlucose) usedEstimates.push("blood glucose");
  if (!profile.cholesterol_value) usedEstimates.push("cholesterol");
  if (!profile.known_max_heart_rate) usedEstimates.push("max heart rate");

  return {
    usedEstimates,
    diabetes: {
      age_years: age, sex: profile.sex, weight_kg: weight, height_cm: height,
      high_blood_pressure: highBP, high_cholesterol: profile.diagnosed_high_cholesterol,
      cholesterol_checked_last_5_years: profile.cholesterol_checked_last_5_years,
      history_of_stroke: profile.diagnosed_stroke_history,
      heart_disease_or_attack_history: profile.diagnosed_heart_disease,
      difficulty_walking_or_climbing_stairs: diffWalk,
      smoker, heavy_alcohol_consumption: heavyAlcohol,
      physically_active_last_30_days: active30,
      eats_fruit_daily: fruitDaily, eats_vegetables_daily: vegDaily,
      general_health: generalHealthFromMood(agg.avgMood || 3),
      poor_mental_health_days_last_30: agg.poorMoodDays,
      poor_physical_health_days_last_30: agg.symptomDays,
    },
    heart_disease: {
      age_years: age, sex: profile.sex,
      resting_blood_pressure: systolic, cholesterol: Number(profile.cholesterol_value) || 200,
      fasting_blood_sugar_over_120: glucose > 120,
      max_heart_rate_achieved: Number(profile.known_max_heart_rate) || Math.round(220 - age),
      exercise_induced_angina: agg.chestPainRatio >= 0.1,
      st_depression_exercise: Number(profile.st_depression_exercise) || 0,
      major_vessels_colored: Number(profile.major_vessels_colored) || 0,
      chest_pain_type_code: Number(profile.chest_pain_type_code) || 0,
      resting_ecg_code: Number(profile.resting_ecg_code) || 0,
      st_slope_code: profile.st_slope_code ?? 1,
      thalassemia_code: profile.thalassemia_code ?? 1,
    },
    stroke: {
      sex: profile.sex, age_years: age, has_hypertension: highBP,
      has_heart_disease: profile.diagnosed_heart_disease, ever_married: profile.ever_married,
      work_type: profile.work_type, residence_type: profile.residence_type,
      avg_glucose_level: glucose, weight_kg: weight, height_cm: height, is_smoker: smoker,
    },
    obesity: {
      gender: profile.sex, age_years: age, family_history_overweight: profile.family_history_overweight,
      frequent_high_calorie_food: agg.highCalRatio >= 0.4,
      vegetable_frequency: agg.vegRatio < 0.34 ? 1 : agg.vegRatio < 0.67 ? 2 : 3,
      main_meals_per_day: Number(profile.main_meals_per_day) || 3,
      eating_between_meals: agg.highCalRatio >= 0.6 ? "Frequently" : agg.highCalRatio >= 0.2 ? "Sometimes" : "no",
      smokes: smoker,
      daily_water_intake: agg.avgWater < 4 ? 1 : agg.avgWater < 8 ? 2 : 3,
      monitors_calories: profile.monitors_calories,
      physical_activity_frequency: agg.activeRatio < 0.15 ? 0 : agg.activeRatio < 0.4 ? 1 : agg.activeRatio < 0.7 ? 2 : 3,
      daily_screen_time: agg.avgScreen < 3 ? 0 : agg.avgScreen < 6 ? 1 : 2,
      alcohol_consumption: agg.alcoholRatio >= 0.6 ? "Frequently" : agg.alcoholRatio >= 0.2 ? "Sometimes" : "no",
      main_transport: profile.main_transport,
    },
    kidney_disease: {
      age_years: age, blood_pressure: systolic,
      hypertension_history: highBP, diabetes_history: profile.diagnosed_diabetes,
      coronary_artery_disease_history: profile.diagnosed_cad,
      appetite: agg.poorAppetiteRatio >= 0.4 ? "poor" : "good",
      pedal_edema: agg.legSwellingRatio >= 0.2,
      anemia: profile.diagnosed_anemia || agg.fatigueRatio >= 0.5,
      serum_creatinine: profile.serum_creatinine || null,
      blood_urea: profile.blood_urea || null,
      hemoglobin: profile.hemoglobin || null,
      urine_specific_gravity: profile.urine_specific_gravity || null,
      urine_albumin: profile.urine_albumin || null,
      blood_glucose_random: agg.latestGlucose ? glucose : null,
    },
  };
}

// ---------------------------------------------------------------------------
// Advice engine — rule-based, tied to each disease's actual top factors
// ---------------------------------------------------------------------------
const ADVICE_RULES = {
  diabetes: [
    { when: (p) => !p.physically_active_last_30_days, tip: "You've logged little activity this month — aim for at least 20-30 minutes of brisk walking on most days; this is one of the strongest levers against diabetes risk." },
    { when: (p) => bmi(p.weight_kg, p.height_cm) >= 27, tip: "Your current BMI is in a higher-risk range — even a 5-7% weight reduction meaningfully lowers diabetes risk." },
    { when: (p) => p.high_blood_pressure, tip: "Your blood pressure readings run high — this compounds diabetes risk. Reducing sodium and increasing activity both help both conditions together." },
    { when: (p) => !p.eats_vegetables_daily || !p.eats_fruit_daily, tip: "Try to include vegetables or fruit at most meals — fiber-rich foods slow glucose spikes." },
    { when: (p) => p.general_health === "fair" || p.general_health === "poor", tip: "Your overall wellbeing check-ins have been low lately — consider a general checkup to rule out underlying issues." },
  ],
  heart_disease: [
    { when: (p) => p.major_vessels_colored === 0 && false, tip: "" }, // placeholder, angiogram data not actionable via lifestyle
    { when: (p) => p.exercise_induced_angina, tip: "You've logged chest discomfort around exercise — this is worth discussing with a doctor before increasing activity intensity." },
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
    { when: (p) => p.anemia, tip: "Fatigue/anemia symptoms are worth a basic blood panel — anemia is common with reduced kidney function but has many other causes too." },
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
        <Field label="Age (years)"><NumberInput value={profile.age_years} onChange={set("age_years")} min={10} max={110} /></Field>
        <Field label="Sex"><Select value={profile.sex} onChange={set("sex")} options={[{ value: "male", label: "Male" }, { value: "female", label: "Female" }]} /></Field>
        <Field label="Height (cm)"><NumberInput value={profile.height_cm} onChange={set("height_cm")} min={100} max={230} /></Field>
        <Field label="Smoker"><Toggle value={profile.smoker} onChange={set("smoker")} /></Field>
        <Field label="Ever married"><Toggle value={profile.ever_married} onChange={set("ever_married")} /></Field>
        <Field label="Work type"><Select value={profile.work_type} onChange={set("work_type")} options={["government_job", "never_worked", "private", "self_employed", "child_not_working"].map((v) => ({ value: v, label: v.replace(/_/g, " ") }))} /></Field>
        <Field label="Residence"><Select value={profile.residence_type} onChange={set("residence_type")} options={[{ value: "urban", label: "Urban" }, { value: "rural", label: "Rural" }]} /></Field>
        <Field label="Main transport"><Select value={profile.main_transport} onChange={set("main_transport")} options={["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"].map((v) => ({ value: v, label: v.replace("_", " ") }))} /></Field>
        <Field label="Monitors calorie intake"><Toggle value={profile.monitors_calories} onChange={set("monitors_calories")} /></Field>
        <Field label="Main meals per day"><NumberInput value={profile.main_meals_per_day} onChange={set("main_meals_per_day")} min={1} max={6} /></Field>
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

        <div className="form-section-title">Optional — from a recent lab report or test (leave blank if unknown)</div>
        <Field label="Cholesterol (mg/dl)"><NumberInput value={profile.cholesterol_value} onChange={set("cholesterol_value")} min={100} max={600} /></Field>
        <Field label="Known max heart rate"><NumberInput value={profile.known_max_heart_rate} onChange={set("known_max_heart_rate")} min={60} max={220} /></Field>
        <Field label="Serum creatinine (mg/dl)"><NumberInput value={profile.serum_creatinine} onChange={set("serum_creatinine")} min={0} max={20} step={0.1} /></Field>
        <Field label="Blood urea (mg/dl)"><NumberInput value={profile.blood_urea} onChange={set("blood_urea")} min={0} max={400} /></Field>
        <Field label="Hemoglobin (g/dl)"><NumberInput value={profile.hemoglobin} onChange={set("hemoglobin")} min={0} max={20} step={0.1} /></Field>
        <Field label="Chest pain type code (0-3)"><NumberInput value={profile.chest_pain_type_code} onChange={set("chest_pain_type_code")} min={0} max={3} /></Field>
        <Field label="Resting ECG code (0-2)"><NumberInput value={profile.resting_ecg_code} onChange={set("resting_ecg_code")} min={0} max={2} /></Field>
        <Field label="ST slope code (0-2)"><NumberInput value={profile.st_slope_code} onChange={set("st_slope_code")} min={0} max={2} /></Field>
        <Field label="Thalassemia code (0-3)"><NumberInput value={profile.thalassemia_code} onChange={set("thalassemia_code")} min={0} max={3} /></Field>
        <Field label="Major vessels colored (0-4)"><NumberInput value={profile.major_vessels_colored} onChange={set("major_vessels_colored")} min={0} max={4} /></Field>
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
        <Field label="Appetite"><Select value={log.appetite} onChange={set("appetite")} options={[{ value: "good", label: "Good" }, { value: "poor", label: "Poor" }]} /></Field>
        <Field label="Leg/ankle swelling"><Toggle value={log.leg_swelling} onChange={set("leg_swelling")} /></Field>
        <Field label="Chest discomfort"><Toggle value={log.chest_discomfort} onChange={set("chest_discomfort")} /></Field>
        <Field label="Fatigue / dizziness"><Toggle value={log.fatigue_or_dizziness} onChange={set("fatigue_or_dizziness")} /></Field>
        <Field label="Difficulty walking/stairs"><Toggle value={log.difficulty_walking} onChange={set("difficulty_walking")} /></Field>
      </div>
      <button className="vt-run-btn" onClick={onSave} disabled={saving}>{saving ? "Saving…" : "Save Today's Entry"}</button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dashboard screen
// ---------------------------------------------------------------------------
function DashboardScreen({ profile, logs, results, onRunAssessment, running, error, payloads, lastRun }) {
  if (logs.length === 0) {
    return <div className="section-intro">No daily entries yet — log at least one day's check-in to generate your first risk assessment.</div>;
  }
  const atRisk = DISEASES.filter((d) => results[d.id]?.risk_label === "at_risk");

  return (
    <div>
      <div className="ov-summary">
        <div className="ov-summary-title">Digital Twin Status</div>
        <div className="ov-empty">
          {logs.length} day{logs.length !== 1 ? "s" : ""} logged
          {lastRun && ` · last assessed ${new Date(lastRun).toLocaleString()}`}
        </div>
        {payloads?.usedEstimates?.length > 0 && (
          <div className="estimate-note">
            Using population-typical estimates for: {payloads.usedEstimates.join(", ")} — log these or add them to your profile for sharper results.
          </div>
        )}
        <button className="vt-run-btn" style={{ marginTop: 14 }} onClick={onRunAssessment} disabled={running}>
          {running ? "Assessing all 5 models…" : "Run Risk Assessment"}
        </button>
        {error && <div className="vt-error">{error}</div>}
      </div>

      {Object.keys(results).length > 0 && (
        <>
          {atRisk.length > 0 ? (
            <div className="alert-banner">
              ⚠ Elevated risk detected: {atRisk.map((d) => d.label).join(", ")}. See details and advice below.
            </div>
          ) : (
            <div className="alert-banner safe">✓ No elevated risk detected across your 5 assessments today.</div>
          )}

          <div className="overview-grid">
            {DISEASES.map((d) => {
              const r = results[d.id];
              if (!r) return null;
              return (
                <div className="ov-card" key={d.id}>
                  <RiskGauge probability={r.risk_probability} size={72} />
                  <div style={{ flex: 1 }}>
                    <div className="ov-card-title">{d.label}</div>
                    <div className="ov-card-sub" style={{ color: r.risk_label === "at_risk" ? "var(--risk)" : "var(--safe)" }}>
                      {r.risk_label === "at_risk" ? "Elevated risk" : "Lower risk"}
                    </div>
                    {r.risk_label === "at_risk" && (
                      <div className="advice-list">
                        {getAdvice(d.id, payloads[d.id]).map((tip, i) => <div className="advice-item" key={i}>• {tip}</div>)}
                      </div>
                    )}
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
          <span>{l.systolic_bp ? `${l.systolic_bp}/${l.diastolic_bp}` : "—"}</span>
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
  const [payloads, setPayloads] = useState({});
  const [lastRun, setLastRun] = useState(null);
  const [loaded, setLoaded] = useState(false);
  const [savingProfile, setSavingProfile] = useState(false);
  const [savingLog, setSavingLog] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    (async () => {
      const p = await loadJSON("profile", DEFAULT_PROFILE);
      const l = await loadJSON("daily_logs", []);
      const cached = await loadJSON("last_assessment", null);
      setProfile(p);
      setLogs(l);
      const existingToday = l.find((e) => e.date === todayStr());
      if (existingToday) setTodayLog(existingToday);
      if (cached) {
        setResults(cached.results || {});
        setPayloads(cached.payloads || {});
        setLastRun(cached.timestamp || null);
      }
      if (!p.set_up) setTab("profile");
      setLoaded(true);
    })();
  }, []);

  const saveProfile = async () => {
    setSavingProfile(true);
    const updated = { ...profile, set_up: true };
    await saveJSON("profile", updated);
    setProfile(updated);
    setSavingProfile(false);
    setTab("checkin");
  };

  const saveLog = async () => {
    setSavingLog(true);
    const others = logs.filter((l) => l.date !== todayLog.date);
    const newLogs = [...others, todayLog];
    await saveJSON("daily_logs", newLogs);
    setLogs(newLogs);
    setSavingLog(false);
    setTab("dashboard");
  };

  const runAssessment = async () => {
    setRunning(true);
    setError(null);
    try {
      const agg = aggregate(logs);
      const built = buildPayloads(profile, agg);
      setPayloads(built);
      const newResults = {};
      for (const d of DISEASES) {
        const { usedEstimates, ...allPayloads } = built;
        const res = await fetch(`${API_BASE}${d.endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(allPayloads[d.id]),
        });
        if (!res.ok) throw new Error(`${d.label}: ${res.status} ${await res.text()}`);
        newResults[d.id] = await res.json();
      }
      setResults(newResults);
      const ts = new Date().toISOString();
      setLastRun(ts);
      await saveJSON("last_assessment", { results: newResults, payloads: built, timestamp: ts });
    } catch (e) {
      setError(
        `Couldn't reach the backend at ${API_BASE}. Make sure it's running locally: ` +
        `cd vitatwin/backend && uvicorn main:app --reload --port 8000. (${e.message})`
      );
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
        <div className="vt-subtitle">Log daily → your digital twin watches for risk → get tailored advice</div>
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
        {tab === "dashboard" && (
          <DashboardScreen profile={profile} logs={logs} results={results} payloads={payloads} lastRun={lastRun}
            onRunAssessment={runAssessment} running={running} error={error} />
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
