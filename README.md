# Hostel Inventory Demand Prediction System

Predicts next week's demand (`Next_Week_Demand`) for hostel inventory items with a scikit-learn model, served by a Flask API and shown in an HTML/CSS/JS dashboard (Chart.js).

```
hostel-inventory-prediction/
├── backend/
│   ├── app.py                 Flask API (loads model.pkl once at start-up)
│   ├── train_model.py         trains + compares models, saves model.pkl / model_metadata.json
│   ├── cleaned_dataset.csv    <-- YOU put your dataset here
│   ├── model.pkl              <-- created by train_model.py
│   ├── model_metadata.json    <-- created by train_model.py
│   └── requirements.txt
├── frontend/  index.html · style.css · script.js
└── README.md
```

> `cleaned_dataset.csv`, `model.pkl` and `model_metadata.json` are not bundled: copy your cleaned dataset (exact columns you specified) into `backend/` and run the training step below to generate the model files.

## 1. Installation

```bash
cd hostel-inventory-prediction/backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: Flask, flask-cors, pandas, numpy, scikit-learn, joblib.

## 2. Train the model

```bash
python train_model.py
```

Prints MAE, RMSE and R² for Linear Regression, Random Forest and Gradient Boosting, selects the best by hold-out RMSE, and writes `model.pkl` and `model_metadata.json` (model name, feature names in order, target name, metrics, item encoding).

Design choices:
- **Preprocessing:** the dataset already one-hot encodes the items and categories, so the text `Item` column is dropped from the features (not encoded a second time). 22 features remain, in the CSV's order; `Next_Week_Demand` is never a feature.
- **Baseline item:** Bathing Soap has no `Item_*` column (all `Item_*` = 0 is its encoding). It is still a valid item in the API.
- **No leakage:** the split is chronological (weeks 1-64 train, 65-80 test), not random, because each row's lag features overlap neighbouring rows. The best model is then refit on all weeks for deployment; the saved metrics are the honest hold-out ones.
- Scale caveat: Drinking Water is in the thousands, so overall RMSE/R² is dominated by it. Per-item hold-out MAE is printed and saved as well.

## 3. Start the backend

```bash
python app.py        # http://127.0.0.1:5000
```

The model is loaded once at start-up; `/api/predict` never retrains.

## 4. Open the frontend

Either open `http://127.0.0.1:5000/` (Flask also serves the frontend), or open `frontend/index.html` directly in a browser (it calls the API at `http://127.0.0.1:5000`, allowed by CORS). Chart.js is loaded from a CDN, so internet access is needed for the charts.

## 5. API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/predict` | predict next week's demand for one item |
| GET | `/api/items` | item list (with category and latest values for pre-filling the form) |
| GET | `/api/dashboard` | KPIs, per-item predictions/status, alerts, chart data, recent predictions |
| GET | `/api/health` | liveness check |

**Example request**

```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d '{
  "week": 41, "item": "Detergent",
  "previous_week_consumption": 33.9, "two_weeks_ago_consumption": 34.4,
  "three_weeks_ago_consumption": 35.9, "current_stock": 48.4,
  "hostel_occupancy": 397, "month": 10 }'
```

**Example response** (the predicted value depends on your trained model)

```json
{
  "item": "Detergent",
  "predicted_next_week_demand": 35.8,
  "current_stock": 48.4,
  "stock_difference": 12.6,
  "recommended_purchase": 0,
  "status": "Sufficient"
}
```

Invalid input returns HTTP 400, e.g. `{"error": "Validation failed.", "details": ["'month' must be between 1 and 12.", ...]}`.

**Business rules (applied after the model prediction; the model does not predict status):**
`stock_difference = current_stock - predicted`; `recommended_purchase = max(predicted - current_stock, 0)`;
status is *Sufficient* if stock >= predicted, *Critical* if stock < 50% of predicted (`CRITICAL_RATIO` in `app.py`), otherwise *Low Stock*.

## 6. How the frontend talks to the backend

1. On load, `script.js` calls `GET /api/items` (fills the Item dropdown) and `GET /api/dashboard` (KPIs, tables, charts, alerts).
2. **Predict Demand** sends the form as JSON to `POST /api/predict`.
3. Flask validates the input, converts the chosen item into the exact one-hot `Item_*` / `Category_*` columns (learned from the dataset and stored in `model_metadata.json`), orders all columns as in training, and calls `model.predict`.
4. The JSON result is shown in the result card; the dashboard is refreshed so "Recent predictions" updates. JavaScript performs no demand calculations.

**Dashboard notes:** the dashboard/inventory/alerts predictions take the latest recorded week in the CSV and roll it forward one week (last week's actual demand becomes "previous week consumption"; stock and occupancy are the last recorded values). Recent predictions are kept in memory and reset when the server restarts. Charts exclude Drinking Water by default because of its scale; use the toggle in the top bar to include it.
