# Customer Churn Prediction Project — Overview

## Dataset

The project uses the Telco Customer Churn dataset (`data/raw/Telco-Customer-Churn.csv`),
containing 7,043 customers and 21 columns, including demographics (gender, senior citizen
status, partner, dependents), account details (tenure, contract type, payment method,
paperless billing), subscribed services (phone, internet, online security, backup, device
protection, tech support, streaming TV and movies), billing amounts (monthly charges, total
charges), and the target label `Churn` (Yes/No).

## Exploratory Data Analysis (notebooks/01_eda.ipynb)

- The dataset has no duplicate rows or duplicate customer IDs.
- `TotalCharges` was loaded as text because 11 brand-new customers (tenure = 0) had blank
  values instead of numbers; these were converted to numeric and filled with 0.
- The target is imbalanced: about 26.5% of customers churned.
- Churned customers tend to have low tenure and higher monthly charges, and lower total
  charges overall since they leave early.
- The strongest churn drivers are month-to-month contracts, fiber optic internet, electronic
  check payment, and lack of tech support or online security. Two-year contracts and
  automatic payment methods are associated with much lower churn.

## Preprocessing (notebooks/02_preprocessing.ipynb)

The raw data is cleaned and encoded before modeling:
1. Drop the `customerID` identifier column.
2. Fix `TotalCharges` (coerce to numeric, fill missing values with 0).
3. Collapse redundant categories: `"No internet service"` and `"No phone service"` are
   merged into `"No"` across the affected columns, since they're fully implied by
   `InternetService` / `PhoneService`.
4. Binary-encode all Yes/No columns (and `gender`) to 0/1.
5. Ordinal-encode `Contract` (Month-to-month = 0, One year = 1, Two year = 2), since churn
   rate strictly decreases in that order.
6. One-hot encode the remaining nominal columns: `InternetService` and `PaymentMethod`.
7. Split into train/test (80/20), stratified on `Churn`, before any scaling to avoid leakage.
8. Scale the numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) with a
   `StandardScaler` fit only on the training set.

Outputs: `data/processed/train.csv`, `data/processed/test.csv`, and `models/scaler.joblib`.

## Model Training (notebooks/03_model_training.ipynb)

Three models were compared with 5-fold stratified cross-validation on ROC-AUC: Logistic
Regression, Random Forest, and Gradient Boosting. Gradient Boosting scored best (CV ROC-AUC
≈ 0.848) and was tuned further with a randomized hyperparameter search, reaching a test-set
ROC-AUC of 0.846.

Because Gradient Boosting doesn't support class weighting, its default 0.5 decision
threshold under-caught churners (recall 0.52). A better threshold (≈ 0.277) was chosen using
out-of-fold predictions on the training set only, raising churn recall to 0.77 on the test
set at some cost to precision — a trade-off favoring catching more churners over minimizing
false alarms.

The top predictors of churn, by feature importance, are `Contract`, `tenure`, and having
fiber optic internet.

Artifacts saved: `models/churn_model.joblib` (the trained Gradient Boosting model),
`models/churn_model_threshold.json` (the chosen decision threshold), and
`models/feature_columns.json` (the exact column order expected by the model).

## Serving (api/)

A FastAPI app in `api/main.py` exposes:
- `GET /health` — basic liveness check.
- `POST /predict` — accepts a customer's raw attributes (matching the original CSV
  columns, validated against known categories via Pydantic), runs them through the same
  preprocessing pipeline as training (`src/features/preprocessing.py`), and returns
  `{churn, churn_probability, threshold}`.

The preprocessing and prediction logic lives in `src/features/` and `src/models/` so it's
shared between the notebooks and the API rather than duplicated.
