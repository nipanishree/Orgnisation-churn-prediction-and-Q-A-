import json
import os
from pathlib import Path

import joblib
import pandas as pd

from features.preprocessing import preprocess

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "models"))


class ChurnPredictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.model = joblib.load(models_dir / "churn_model.joblib")
        self.scaler = joblib.load(models_dir / "scaler.joblib")

        with open(models_dir / "churn_model_threshold.json") as f:
            self.threshold = json.load(f)["threshold"]
        with open(models_dir / "feature_columns.json") as f:
            self.feature_columns = json.load(f)

    def predict(self, raw: dict) -> dict:
        df = pd.DataFrame([raw])
        X = preprocess(df, self.scaler, self.feature_columns)

        probability = float(self.model.predict_proba(X)[0, 1])
        return {
            "churn": probability >= self.threshold,
            "churn_probability": round(probability, 4),
            "threshold": round(self.threshold, 4),
        }


predictor = ChurnPredictor()
