# VitaTwin — Full Preventive Health Digital Twin (v1)

Five trained risk models + a FastAPI backend + a connected React dashboard.

## What's in here
```
vitatwin/
├── models/                              # trained models + metadata (feature order, metrics, encoders)
│   ├── diabetes_model.joblib / _metadata.json
│   ├── heart_disease_model.joblib / _metadata.json
│   ├── stroke_model.joblib / _metadata.json
│   ├── obesity_model.joblib / _metadata.json
│   └── kidney_model.joblib / _metadata.json
├── backend/
│   ├── main.py              # FastAPI app — one /predict/<disease> endpoint per model
│   ├── code_mappings.py     # converts real-world input -> model codes
│   └── requirements.txt
├── notebooks/                           # full, rerunnable training scripts (one per model)
└── frontend_App.jsx                     # React dashboard, calls the backend directly
```

## Run it

**1. Backend**
```bash
cd vitatwin/backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Swagger UI at **http://localhost:8000/docs** to test any endpoint directly.

**2. Frontend**
The dashboard (`frontend_App.jsx`) is a **daily-logging digital twin**, not a
one-off form:

- **Profile tab** — one-time setup (age, sex, height, diagnosed conditions,
  optional lab values). Persists via the artifact's built-in storage.
- **Daily Check-in tab** — log today's weight, activity, diet, mood, symptoms
  in under a minute. Every entry is saved and stored per-day.
- **Dashboard tab** — one button ("Run Risk Assessment") aggregates your last
  30 days of logs + profile into all 5 models' input formats, calls the
  backend, and shows: a risk alert banner if anything is elevated, a gauge per
  disease, and **tailored advice** generated from which of that disease's real
  risk factors are actually present in your logged data (not generic tips).
- **History tab** — a simple log of past entries.

All data persists across sessions (backed by the artifact storage API) — log a
few days in a row and the risk assessment gets more accurate as it has more to
average over. With the backend running locally (step 1) and this chat open in
the same browser, it talks to `http://localhost:8000` directly.

To run it as a standalone app instead: copy `frontend_App.jsx` into a
Create React App / Vite project as `src/App.jsx`. Note it uses `window.storage`
(Claude artifact-specific) for persistence — outside claude.ai you'd swap that
for `localStorage`, IndexedDB, or a real backend database (see "Next steps").

## The 5 models — honest scorecard

| Model | Metric | Score | Framing |
|---|---|---|---|
| Diabetes | Accuracy / ROC-AUC | 74.5% / 0.82 | Binary: at-risk (pre-diabetic+diabetic) vs. not, from lifestyle only |
| Heart Disease | Accuracy / ROC-AUC | 91.4% / 0.95 | Binary, from clinical exam values (BP, cholesterol, ECG, etc.) |
| Stroke | Accuracy / ROC-AUC | 85.5% / 0.91 | Binary, from history + glucose/BMI |
| Obesity | Accuracy / ROC-AUC | 91.2% / 0.97 | Binary, from **lifestyle only** — Height/Weight deliberately excluded (see below) |
| Kidney Disease | CV-Accuracy / ROC-AUC | 99.3% / 0.9996 | Binary, basic history + optional lab values |

**Every number above was checked for the leakage traps that inflate a lot of public
health datasets** (synthetic oversampling creating near-duplicate rows across
train/test, or a target that's trivially derived from a feature already in the
data). Specifically:

- The diabetes, stroke, and heart-disease data you uploaded (`diabetes_data.csv`,
  `stroke_data.csv`, `hypertension_data.csv`) are heavily augmented with synthetic
  rows. A naive random split reported ~100% "accuracy" on two of them — that's
  leakage, not skill. We used a group-aware split (clustering near-duplicates so
  none cross the train/test boundary) to get the honest numbers above.
- `hypertension_data.csv` is actually the UCI Cleveland **Heart Disease** dataset
  (chest pain type, ECG results, etc.), not hypertension — the `target` column
  means "has heart disease." Built and labeled accordingly.
- Obesity's `Height`/`Weight` columns are excluded on purpose: `NObeyesdad` is
  literally BMI-derived, so including them would make the model a disguised BMI
  calculator (trivial, not predictive). The shipped model predicts obesity risk
  from lifestyle habits alone.
- Kidney disease genuinely needs lab values (creatinine, hemoglobin) for strong
  accuracy — same as real clinical CKD risk tools. The API marks those fields
  optional/advanced; basic history alone still gives a (weaker, appropriately
  low-confidence) estimate.

## API summary

All 5 endpoints follow the same pattern: `POST /predict/<disease>` with a JSON
body of real-world values (actual age, yes/no history, etc. — no survey codes
exposed to the frontend), returns:
```json
{
  "risk_label": "at_risk" | "lower_risk",
  "risk_probability": 0.0-1.0,
  "top_contributing_factors": ["...", "..."],
  "disclaimer": "..."
}
```

## Next steps
1. Add a combined Digital Twin health index that blends all 5 scores into one number
2. Move storage from artifact `window.storage` to a real backend database
   (PostgreSQL) so logs survive outside claude.ai and support multi-device sync
3. Add risk-trend history (store each assessment's timestamp + scores, chart
   the trend over weeks/months, not just the latest snapshot)
4. Deploy backend (Render/Railway) + frontend (Vercel) for a real hosted app
5. Add user accounts/auth so results are tied to a person, not a single session
