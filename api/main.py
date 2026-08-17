from fastapi import FastAPI

from api.schemas import CustomerData, PredictionResponse, QARequest, QAResponse
from models.predict import predictor
from rag.qa import get_qa_engine

app = FastAPI(
    title="Churn Prediction & Q&A API",
    description="Predicts customer churn probability, and answers questions about the project "
    "using retrieval-augmented generation over documents/knowledge_base/.",
    version="0.1.0",
)


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
