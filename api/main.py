import io
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import accuracy_score, precision_score, recall_score

from api.schemas import (
    BatchPredictResponse,
    CustomerData,
    PredictionResponse,
    QARequest,
    QAResponse,
    UploadResponse,
)
from models.predict import predictor
from rag import ingest as rag_ingest
from rag.qa import get_qa_engine, refresh_knowledge_base

app = FastAPI(
    title="Churn Prediction & Q&A API",
    description="Predicts customer churn probability, and answers questions about the project "
    "using retrieval-augmented generation over documents/knowledge_base/.",
    version="0.1.0",
)

ALLOWED_DOC_EXTENSIONS = {".md", ".txt"}
MAX_DOC_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
MAX_DATASET_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
REQUIRED_PREDICT_COLUMNS = set(CustomerData.model_fields.keys())

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    return predictor.predict(customer.model_dump())


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if not name or Path(name).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    content = await file.read()
    if len(content) > MAX_DATASET_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"'{name}' exceeds the {MAX_DATASET_SIZE_BYTES // (1024 * 1024)}MB limit.",
        )

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:  # noqa: BLE001 - any parse failure should become a clean 400
        raise HTTPException(status_code=400, detail=f"Could not parse '{name}' as CSV.")

    if df.empty:
        raise HTTPException(status_code=400, detail=f"'{name}' has no rows.")

    missing = REQUIRED_PREDICT_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"'{name}' is missing required columns: {sorted(missing)}.",
        )

    customer_ids = df["customerID"].astype(str) if "customerID" in df.columns else None
    actual_churn = df["Churn"] if "Churn" in df.columns else None

    feature_df = df.drop(columns=[c for c in ("customerID", "Churn") if c in df.columns])

    try:
        result = predictor.predict_batch(feature_df)
    except Exception as e:  # noqa: BLE001 - any scoring failure should become a clean 400
        raise HTTPException(status_code=400, detail=f"Could not score '{name}': {e}")

    rows = []
    for i in range(len(result)):
        rows.append(
            {
                "row": i,
                "customer_id": customer_ids.iloc[i] if customer_ids is not None else None,
                "churn_probability": float(result["churn_probability"].iloc[i]),
                "churn": bool(result["churn"].iloc[i]),
                "actual_churn": actual_churn.iloc[i] if actual_churn is not None else None,
            }
        )

    summary = {
        "total_rows": len(result),
        "predicted_churn_count": int(result["churn"].sum()),
        "predicted_churn_rate": round(float(result["churn"].mean()), 4),
        "threshold": round(predictor.threshold, 4),
    }

    if actual_churn is not None:
        actual_bool = actual_churn.map({"Yes": True, "No": False})
        if actual_bool.isna().any():
            raise HTTPException(
                status_code=400,
                detail=f"'{name}' has a Churn column with values other than 'Yes'/'No'.",
            )
        summary["actual_churn_rate"] = round(float(actual_bool.mean()), 4)
        summary["accuracy"] = round(accuracy_score(actual_bool, result["churn"]), 4)
        summary["precision"] = round(precision_score(actual_bool, result["churn"]), 4)
        summary["recall"] = round(recall_score(actual_bool, result["churn"]), 4)

    return {"summary": summary, "rows": rows}


@app.post("/qa", response_model=QAResponse)
def ask_question(request: QARequest):
    engine = get_qa_engine()
    return engine.answer(request.question, top_k=request.top_k)


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    saved_names = []
    for file in files:
        # .name strips any directory components (including "../"), preventing
        # path traversal outside the knowledge base directory.
        name = Path(file.filename or "").name
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise HTTPException(status_code=400, detail=f"Invalid filename: '{file.filename}'.")

        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_DOC_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for '{name}'. Allowed: {sorted(ALLOWED_DOC_EXTENSIONS)}.",
            )

        content = await file.read()
        if len(content) > MAX_DOC_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"'{name}' exceeds the {MAX_DOC_SIZE_BYTES // (1024 * 1024)}MB limit.",
            )

        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail=f"'{name}' is not valid UTF-8 text.")

        destination = rag_ingest.KNOWLEDGE_BASE_DIR / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        saved_names.append(name)

    stats = rag_ingest.run()
    refresh_knowledge_base()

    return {
        "uploaded_files": saved_names,
        "num_documents": stats["num_documents"],
        "num_chunks": stats["num_chunks"],
    }


# Must be mounted last: this catches every path not matched by a route above,
# so any API route defined after this point would be unreachable.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
