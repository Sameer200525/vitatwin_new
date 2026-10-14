# VitaTwin

VitaTwin is a React dashboard, a FastAPI backend, and five trained disease
models. Hypertension has **no trained model** in this repository; the frontend
provides a separate, transparent blood-pressure measurement rule check.

## Run locally

Prerequisites: Node.js and Python 3.10 or newer.

### Backend

From the repository root:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Interactive API documentation is at http://localhost:8000/docs.
Set `VITATWIN_CORS_ORIGINS` to a comma-separated list of exact frontend origins
when using a hosted frontend. Local Vite development/preview origins are allowed
by default.

### Frontend

From the repository root in a second terminal:

```powershell
npm install
npm run dev
```

The API defaults to `http://localhost:8000`; override it with
`VITE_API_BASE_URL` before building/deploying the frontend.

## How assessments work

- Profile and daily check-in data are saved in browser storage. Each date has
  one entry; editing today's entry replaces that date's existing entry.
- Model use of profile history requires explicit profile review. Yes/no daily
  answers only contribute to aggregation when that day's check-in is marked
  reviewed.
- Model inputs use check-ins within the last 30 calendar days. Old/future dates
  are excluded, repeated dates are deduplicated, measured glucose is averaged
  for the stroke model, and a recent weight/BP reading is used only when entered.
- Each trained model is requested independently. Missing required inputs are
  shown as insufficient data; a failed request does not discard other results.
- Heart-disease assessment requires actual clinical-report values including
  ECG, stress-test and angiogram fields. Ordinary daily symptom logs are not
  used as substitutes.
- Kidney assessment requires measured blood pressure, serum creatinine,
  hemoglobin, and urine specific gravity. Other missing kidney labs are filled
  with the training-set median by the model endpoint and listed as
  `imputed_features`; missing categorical urinalysis fields use stated defaults
  listed as `assumed_defaults` in the result.
- Hypertension is a rule-based reading flag at systolic 130 mmHg or diastolic
  80 mmHg and above. It is not an ML prediction or a diagnosis. One reading
  cannot confirm or rule out hypertension.
- Advice is generated from reported inputs using transparent frontend rules.
  Backend `global_feature_importance` values describe dataset-level model
  importance; they are not patient-specific explanations or verified causes.

All model outputs are statistical estimates, not medical diagnoses. They are
not a substitute for care from a qualified clinician.

## Model metadata and limitations

The checked-in JSON metadata records the following evaluation metrics. They
were not independently recomputed as part of the code fixes described here.

| Model | Metric recorded in metadata | Notes |
|---|---|---|
| Diabetes | Accuracy 0.745; ROC-AUC 0.820 | Binary risk estimate from survey/lifestyle features |
| Heart disease | Accuracy 0.914; ROC-AUC 0.954 | Requires clinical exam values; data is described as synthetically expanded |
| Stroke | Accuracy 0.855; ROC-AUC 0.914 | Requires measured glucose, body measurements and profile history |
| Obesity | Accuracy 0.912; ROC-AUC 0.969 | Lifestyle model; height/weight are excluded from its trained features |
| Kidney disease | 5-fold CV accuracy 0.993; ROC-AUC 0.9996 | Small, imputed dataset; requires several measured lab values |

Scores from public/synthetic datasets may not generalize to the user or another
population. The models are not clinically validated, calibrated diagnoses.
