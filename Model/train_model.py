from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "house_data.csv"
MODEL_DIR = BASE_DIR / "Model"
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
METRICS_PATH = MODEL_DIR / "metrics.joblib"

BASE_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "grade",
    "condition",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]
TARGET = "price"
N_ESTIMATORS = 300
MIN_LOT_SIZE = 1


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()
    transformed["living_lot_ratio"] = transformed["sqft_living"] / transformed["sqft_lot"].clip(lower=MIN_LOT_SIZE)
    transformed["nearby_living_gap"] = transformed["sqft_living"] - transformed["sqft_living15"]
    return transformed


def train_and_save(data_path: Path = DATA_PATH) -> dict:
    data = pd.read_csv(data_path)
    x = engineer_features(data[BASE_FEATURES])
    y = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=42)
    model.fit(x_train_scaled, y_train)

    predictions = model.predict(x_test_scaled)
    metrics = {
        "r2_score": float(r2_score(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "feature_order": list(x.columns),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(metrics, METRICS_PATH)
    return metrics


if __name__ == "__main__":
    scores = train_and_save()
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"R2 Score: {scores['r2_score']:.4f}")
    print(f"MSE: {scores['mse']:.2f}")
