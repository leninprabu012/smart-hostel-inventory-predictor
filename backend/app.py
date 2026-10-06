"""Flask API for the Hostel Inventory Demand Prediction System."""
import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE = Path(__file__).resolve().parent
FRONTEND_DIR = BASE.parent / "frontend"
MODEL_PATH, META_PATH, DATA_PATH = BASE / "model.pkl", BASE / "model_metadata.json", BASE / "cleaned_dataset.csv"

# Business rule: stock below this fraction of predicted demand is "Critical".
CRITICAL_RATIO = 0.5
WATER = "Drinking Water"

for p in (MODEL_PATH, META_PATH, DATA_PATH):
    if not p.exists():
        sys.exit(f"Missing {p.name}. Run `python train_model.py` first "
                 f"(and make sure cleaned_dataset.csv is in backend/).")

# Loaded ONCE at start-up - never retrained per request.
model = joblib.load(MODEL_PATH)
meta = json.loads(META_PATH.read_text())
FEATURES = meta["feature_names"]
ITEM_ENCODING = meta["item_encoding"]
ITEMS = meta["items"]
history = pd.read_csv(DATA_PATH)
recent_predictions = deque(maxlen=20)

app = Flask(__name__)
CORS(app)

FIELDS = {  # request key -> (type, min, max)
    "week": (int, 1, None),
    "previous_week_consumption": (float, 0, None),
    "two_weeks_ago_consumption": (float, 0, None),
    "three_weeks_ago_consumption": (float, 0, None),
    "current_stock": (float, 0, None),
    "hostel_occupancy": (float, 0, None),
    "month": (int, 1, 12),
}
COLUMN_FOR = {
    "week": "Week", "previous_week_consumption": "Previous_Week_Consumption",
    "two_weeks_ago_consumption": "Two_Weeks_Ago_Consumption",
    "three_weeks_ago_consumption": "Three_Weeks_Ago_Consumption",
    "current_stock": "Current_Stock", "hostel_occupancy": "Hostel_Occupancy", "month": "Month",
}


def category_of(item):
    cats = [c.replace("Category_", "").replace("_", " ") for c, v in ITEM_ENCODING[item].items()
            if c.startswith("Category_") and v == 1]
    return cats[0] if cats else "Cleaning/Other"


def validate(payload):
    if not isinstance(payload, dict):
        return None, ["Request body must be a JSON object."]
    errors, clean = [], {}
    item = payload.get("item")
    if not isinstance(item, str) or not item.strip():
        errors.append("'item' is required.")
    elif item not in ITEM_ENCODING:
        errors.append(f"Unknown item '{item}'. Available items: {', '.join(ITEMS)}.")
    else:
        clean["item"] = item
    for key, (typ, lo, hi) in FIELDS.items():
        raw = payload.get(key)
        if raw is None or raw == "":
            errors.append(f"'{key}' is required.")
            continue
        if isinstance(raw, bool):
            errors.append(f"'{key}' must be a number.")
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{key}' must be a number.")
            continue
        if typ is int:
            if val != int(val):
                errors.append(f"'{key}' must be a whole number.")
                continue
            val = int(val)
        if val < lo or (hi is not None and val > hi):
            rng = f"between {lo} and {hi}" if hi is not None else f">= {lo}"
            errors.append(f"'{key}' must be {rng}.")
            continue
        clean[key] = val
    return (None, errors) if errors else (clean, [])


def to_frame(rows):
    """rows: list of validated dicts -> DataFrame with the exact training feature order."""
    records = []
    for r in rows:
        rec = {COLUMN_FOR[k]: r[k] for k in COLUMN_FOR}
        rec.update(ITEM_ENCODING[r["item"]])  # one-hot Item_* / Category_* from the selected item
        records.append(rec)
    return pd.DataFrame(records).reindex(columns=FEATURES)


def predict_many(rows):
    return [max(float(p), 0.0) for p in model.predict(to_frame(rows))]  # demand cannot be negative


def stock_status(stock, predicted):
    """Business rule applied AFTER the model prediction (the model does not predict status)."""
    if stock >= predicted:
        return "Sufficient"
    return "Critical" if stock < CRITICAL_RATIO * predicted else "Low Stock"


def summarize(item, stock, predicted):
    return {
        "item": item,
        "predicted_next_week_demand": round(predicted, 2),
        "current_stock": round(stock, 2),
        "stock_difference": round(stock - predicted, 2),
        "recommended_purchase": round(max(predicted - stock, 0), 2),
        "status": stock_status(stock, predicted),
    }


@app.get("/api/health")
def health():
    return jsonify(status="ok", model=meta["model_name"])


@app.get("/api/items")
def items():
    last_week = history["Week"].max()
    latest = history[history["Week"] == last_week].set_index("Item")
    out = []
    for name in ITEMS:
        row = latest.loc[name]
        out.append({
            "name": name, "category": category_of(name),
            "latest": {  # latest recorded values rolled forward one week, to pre-fill the form
                "week": int(last_week) + 1,
                "previous_week_consumption": float(row["Next_Week_Demand"]),
                "two_weeks_ago_consumption": float(row["Previous_Week_Consumption"]),
                "three_weeks_ago_consumption": float(row["Two_Weeks_Ago_Consumption"]),
                "current_stock": float(row["Current_Stock"]),
                "hostel_occupancy": float(row["Hostel_Occupancy"]),
                "month": int(row["Month"]),
            },
        })
    return jsonify(items=out)


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="Request body must be valid JSON (Content-Type: application/json)."), 400
    clean, errors = validate(payload)
    if errors:
        return jsonify(error="Validation failed.", details=errors), 400
    predicted = predict_many([clean])[0]
    result = summarize(clean["item"], clean["current_stock"], predicted)
    recent_predictions.appendleft({**result, "week": clean["week"],
                                   "timestamp": datetime.now().isoformat(timespec="seconds")})
    return jsonify(result)


@app.get("/api/dashboard")
def dashboard():
    """Roll the latest recorded week forward one week and predict each item with the saved model."""
    last_week = int(history["Week"].max())
    latest = history[history["Week"] == last_week]
    rows = []
    for _, r in latest.iterrows():
        rows.append({
            "item": r["Item"], "week": last_week + 1,
            "previous_week_consumption": float(r["Next_Week_Demand"]),
            "two_weeks_ago_consumption": float(r["Previous_Week_Consumption"]),
            "three_weeks_ago_consumption": float(r["Two_Weeks_Ago_Consumption"]),
            "current_stock": float(r["Current_Stock"]),
            "hostel_occupancy": float(r["Hostel_Occupancy"]), "month": int(r["Month"]),
        })
    preds = predict_many(rows)

    table, alerts = [], []
    for r, p in zip(rows, preds):
        s = summarize(r["item"], r["current_stock"], p)
        action = ("No action needed" if s["status"] == "Sufficient"
                  else f"Purchase {s['recommended_purchase']:g} units")
        table.append({**s, "category": category_of(r["item"]),
                      "previous_week_consumption": r["previous_week_consumption"],
                      "recommended_action": action})
        if s["status"] != "Sufficient":
            msg = (f"{r['item']} has insufficient current stock." if s["status"] == "Critical"
                   else f"{r['item']} may require additional stock next week.")
            alerts.append({"item": r["item"], "severity": s["status"], "message": msg,
                           "shortfall": s["recommended_purchase"]})
    alerts.sort(key=lambda a: a["severity"] != "Critical")

    # Weekly trend: demand realised in week w+1 is the target of row w.
    non_water = history[history["Item"] != WATER].groupby("Week")["Next_Week_Demand"].sum()
    water = history[history["Item"] == WATER].groupby("Week")["Next_Week_Demand"].sum()
    pred_non_water = sum(t["predicted_next_week_demand"] for t in table if t["item"] != WATER)
    pred_water = sum(t["predicted_next_week_demand"] for t in table if t["item"] == WATER)

    categories = {}
    for t in table:
        categories[t["category"]] = categories.get(t["category"], 0) + 1

    return jsonify(
        model={"name": meta["model_name"], "metrics": meta["evaluation_metrics"],
               "note": meta.get("evaluation_note")},
        forecast_week=last_week + 1,
        summary={
            "total_items": len(table),
            "items_requiring_attention": len(alerts),
            "total_current_stock": round(sum(t["current_stock"] for t in table), 1),
            "total_predicted_demand": round(sum(t["predicted_next_week_demand"] for t in table), 1),
        },
        items=table,
        alerts=alerts,
        category_distribution=categories,
        trend={
            "labels": [f"W{int(w) + 1}" for w in non_water.index],
            "total_excl_water": [round(float(v), 1) for v in non_water.values],
            "drinking_water": [round(float(v), 1) for v in water.values],
            "forecast_label": f"W{last_week + 2} (forecast)",
            "forecast_excl_water": round(pred_non_water, 1),
            "forecast_drinking_water": round(pred_water, 1),
        },
        recent_predictions=list(recent_predictions),
    )


# Optional convenience: also serve the frontend at http://127.0.0.1:5000/
@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:name>")
def static_files(name):
    return send_from_directory(FRONTEND_DIR, name)


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Not found."), 404


@app.errorhandler(500)
def server_error(_):
    return jsonify(error="Internal server error."), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
