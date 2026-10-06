"""
Train the Next_Week_Demand regression model.

    python train_model.py

Reads cleaned_dataset.csv, compares Linear Regression / Random Forest /
Gradient Boosting on a chronological hold-out (last 20% of weeks), picks the
best by RMSE, refits it on all rows and saves:

    model.pkl             - trained model (joblib)
    model_metadata.json   - model name, feature order, target, metrics, item encoding
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "cleaned_dataset.csv"
MODEL_PATH = BASE / "model.pkl"
META_PATH = BASE / "model_metadata.json"

TARGET = "Next_Week_Demand"
TEXT_COL = "Item"  # human-readable label; already represented by the Item_* one-hot columns
TEST_FRACTION = 0.2
SEED = 42

EXPECTED_COLUMNS = [
    "Week", "Item", "Previous_Week_Consumption", "Two_Weeks_Ago_Consumption",
    "Three_Weeks_Ago_Consumption", "Current_Stock", "Hostel_Occupancy", "Month",
    "Item_Detergent", "Item_Dishwashing_Liquid", "Item_Drinking_Water",
    "Item_Floor_Cleaner", "Item_Garbage_Bags", "Item_Handwash", "Item_Notebooks",
    "Item_Pens", "Item_Phenyl", "Item_Tissue_Paper", "Item_Toilet_Paper",
    "Category_Essentials", "Category_Hygiene", "Category_Personal_Care",
    "Category_Stationery", "Next_Week_Demand",
]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: {DATA_PATH} not found. Put your cleaned dataset there as cleaned_dataset.csv.")
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    if list(df.columns) != EXPECTED_COLUMNS:
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
        sys.exit(f"ERROR: dataset columns differ from the expected schema.\n missing={missing}\n extra={extra}")
    if df.isna().any().any():
        sys.exit("ERROR: dataset contains missing values; expected a cleaned dataset.")
    return df


def build_item_encoding(df: pd.DataFrame, encoded_cols: list) -> dict:
    """item name -> {one-hot/category column: value}, learned from the data itself.

    Note: an item whose Item_* columns are all 0 (e.g. Bathing Soap) is the implicit
    baseline of the one-hot encoding. It is kept as a valid item with all Item_* = 0.
    """
    encoding = {}
    for item, grp in df.groupby(TEXT_COL):
        vals = grp[encoded_cols].drop_duplicates()
        if len(vals) != 1:
            sys.exit(f"ERROR: item '{item}' has inconsistent one-hot encoding in the dataset.")
        encoding[item] = {c: int(vals.iloc[0][c]) for c in encoded_cols}
    return encoding


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main():
    df = load_data().sort_values(["Week", TEXT_COL]).reset_index(drop=True)

    # Features: everything except the target and the raw text label. The one-hot
    # Item_*/Category_* columns already encode Item, so we do NOT re-encode it.
    feature_names = [c for c in EXPECTED_COLUMNS if c not in (TARGET, TEXT_COL)]
    assert TARGET not in feature_names  # leakage guard

    X, y = df[feature_names], df[TARGET]

    # Chronological split (no shuffling): train on earlier weeks, test on the latest weeks.
    weeks = np.sort(df["Week"].unique())
    cutoff = weeks[int(len(weeks) * (1 - TEST_FRACTION))]
    train_mask = df["Week"] < cutoff
    X_train, X_test, y_train, y_test = X[train_mask], X[~train_mask], y[train_mask], y[~train_mask]
    print(f"Rows: {len(df)} | features: {len(feature_names)} | "
          f"train weeks {weeks[0]}-{cutoff - 1} ({len(X_train)} rows), "
          f"test weeks {cutoff}-{weeks[-1]} ({len(X_test)} rows)\n")

    candidates = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=SEED),
    }

    results = {}
    print(f"{'Model':<30}{'MAE':>10}{'RMSE':>12}{'R2':>9}")
    print("-" * 61)
    for name, est in candidates.items():
        est.fit(X_train, y_train)
        results[name] = metrics(y_test, est.predict(X_test))
        m = results[name]
        print(f"{name:<30}{m['MAE']:>10.3f}{m['RMSE']:>12.3f}{m['R2']:>9.4f}")

    best_name = min(results, key=lambda n: results[n]["RMSE"])
    print(f"\nBest model (lowest hold-out RMSE): {best_name}")

    # Per-item MAE for the best model (hold-out): items have very different scales.
    best_eval = candidates[best_name]
    per_item = (
        pd.DataFrame({"Item": df.loc[~train_mask, TEXT_COL], "abs_err": np.abs(y_test - best_eval.predict(X_test))})
        .groupby("Item")["abs_err"].mean().round(2).to_dict()
    )
    print("Hold-out MAE per item:", per_item)

    # Refit the winner on ALL rows for deployment; reported metrics remain the honest hold-out ones.
    final_model = candidates[best_name]
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)

    encoded_cols = [c for c in feature_names if c.startswith(("Item_", "Category_"))]
    metadata = {
        "model_name": best_name,
        "target_name": TARGET,
        "feature_names": feature_names,
        "evaluation_metrics": results[best_name],
        "all_model_metrics": results,
        "holdout_mae_per_item": per_item,
        "evaluation_note": f"Chronological hold-out: weeks >= {int(cutoff)} held out; final model refit on all weeks.",
        "items": sorted(df[TEXT_COL].unique().tolist()),
        "item_encoding": build_item_encoding(df, encoded_cols),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved {MODEL_PATH.name} and {META_PATH.name}")


if __name__ == "__main__":
    main()
