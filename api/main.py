from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from api.schemas import (
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

ALLOWED_UPLOAD_EXTENSIONS = {".md", ".txt"}
MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024  # 5MB


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    return predictor.predict(customer.model_dump())


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
        if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for '{name}'. Allowed: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}.",
            )

        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"'{name}' exceeds the {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB limit.",
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
