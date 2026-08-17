import pandas as pd

INTERNET_DEPENDENT_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
BINARY_MAPS = {"gender": {"Female": 0, "Male": 1}}
DEFAULT_BINARY_MAP = {"No": 0, "Yes": 1}

CONTRACT_MAP = {"Month-to-month": 0.0, "One year": 1.0, "Two year": 2.0}
NOMINAL_COLS = ["InternetService", "PaymentMethod"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def preprocess(raw_df: pd.DataFrame, scaler, feature_columns: list[str]) -> pd.DataFrame:
    """Mirrors the transformation steps from notebooks/02_preprocessing.ipynb (steps 3-9),
    using an already-fitted scaler instead of fitting a new one."""
    df = raw_df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    for col in INTERNET_DEPENDENT_COLS:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    for col in BINARY_COLS:
        df[col] = df[col].map(BINARY_MAPS.get(col, DEFAULT_BINARY_MAP))

    df["Contract"] = df["Contract"].map(CONTRACT_MAP)

    df = pd.get_dummies(df, columns=NOMINAL_COLS, prefix=NOMINAL_COLS, dtype=int)
    # reindex handles both column ordering and one-hot categories absent from this batch
    df = df.reindex(columns=feature_columns, fill_value=0)

    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return df
