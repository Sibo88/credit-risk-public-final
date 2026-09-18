# Trustworthy AI Credit Risk — Public Web

Modern public website for the thesis demonstration system.

## Structure
- `frontend/`: React + Vite dashboard
- `backend/`: FastAPI + LightGBM API
- `backend/artifacts/research.json`: verified research values transcribed from the thesis
- `backend/models/`: supplied LightGBM models

## Run locally
### Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000/docs

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Open the Vite URL (normally http://localhost:5173).

## Notes
The website distinguishes verified thesis results from live calculations. The monitoring workflow can retrain a candidate model when a threshold-based degradation signal is detected. The candidate is compared with the current production model using holdout ROC-AUC; if the candidate is better, it is saved as a timestamped production model and becomes the active model. The thesis explicitly states that complete post-retraining conformal recalibration was not performed because a suitable recent calibration dataset was unavailable, so the application does not claim a new post-retraining q-hat.

Monitoring uploaded labelled datasets calculates all seven trustworthiness indicators using the installed reference/calibration artifacts. When a threshold-based degradation signal is detected, automatic retraining is enabled by default. The candidate is accepted only when its holdout ROC-AUC is higher than the current model; accepted candidates are activated as the production model. The verified research dashboard remains populated from the thesis evidence.

## Public deployment
1. Push this repository to GitHub.
2. Deploy `backend/` as a Render web service (or another Python host) and set `FRONTEND_ORIGINS` to the final Vercel URL.
3. Deploy `frontend/` to Vercel and set `VITE_API_URL` to the public backend URL.
4. Redeploy the frontend after setting the environment variable.
