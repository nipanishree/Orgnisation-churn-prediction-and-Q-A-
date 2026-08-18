from fastapi.testclient import TestClient

from api.main import app
from rag import ingest as rag_ingest
from rag.qa import refresh_knowledge_base

client = TestClient(app)


def _restore_knowledge_base():
    """Re-ingest and refresh the live index, undoing any test uploads."""
    rag_ingest.run()
    refresh_knowledge_base()

BASE_PAYLOAD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_high_risk_profile():
    response = client.post("/predict", json=BASE_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"churn", "churn_probability", "threshold"}
    assert body["churn"] is True
    assert 0 <= body["churn_probability"] <= 1


def test_predict_low_risk_profile():
    payload = dict(
        BASE_PAYLOAD, tenure=60, Contract="Two year",
        PaymentMethod="Credit card (automatic)", InternetService="No",
        MultipleLines="No", OnlineSecurity="No internet service",
        OnlineBackup="No internet service", DeviceProtection="No internet service",
        TechSupport="No internet service", StreamingTV="No internet service",
        StreamingMovies="No internet service", PaperlessBilling="No",
        MonthlyCharges=20.05, TotalCharges=1200.0,
    )
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["churn"] is False


def test_predict_rejects_invalid_category():
    response = client.post("/predict", json=dict(BASE_PAYLOAD, Contract="Invalid"))
    assert response.status_code == 422


def test_qa_answers_in_scope_question():
    response = client.post("/qa", json={"question": "What are the strongest predictors of churn?"})
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"question", "answer", "sources"}
    assert len(body["answer"]) > 0
    assert len(body["sources"]) > 0


def test_qa_declines_out_of_scope_question():
    response = client.post("/qa", json={"question": "What is the capital of France?"})
    assert response.status_code == 200

    body = response.json()
    assert body["sources"] == []
    assert "don't know" in body["answer"].lower()


def test_qa_rejects_empty_question():
    response = client.post("/qa", json={"question": ""})
    assert response.status_code == 422


def test_upload_document_and_query_live():
    test_file = rag_ingest.KNOWLEDGE_BASE_DIR / "_test_upload.md"
    try:
        content = b"# Test Fact\n\nThe test canary phrase is zorblatt-nine."
        response = client.post(
            "/documents/upload",
            files=[("files", ("_test_upload.md", content, "text/markdown"))],
        )
        assert response.status_code == 200
        assert response.json()["uploaded_files"] == ["_test_upload.md"]

        qa_response = client.post("/qa", json={"question": "What is the test canary phrase?"})
        assert qa_response.status_code == 200
        assert "zorblatt" in qa_response.json()["answer"].lower()
    finally:
        test_file.unlink(missing_ok=True)
        _restore_knowledge_base()


def test_upload_sanitizes_path_traversal_filename():
    response = client.post(
        "/documents/upload",
        files=[("files", ("../../escape_attempt.md", b"content", "text/markdown"))],
    )
    assert response.status_code == 200

    saved_name = response.json()["uploaded_files"][0]
    assert "/" not in saved_name and ".." not in saved_name

    saved_path = rag_ingest.KNOWLEDGE_BASE_DIR / saved_name
    assert saved_path.resolve().parent == rag_ingest.KNOWLEDGE_BASE_DIR.resolve()

    saved_path.unlink(missing_ok=True)
    _restore_knowledge_base()


def test_upload_rejects_disallowed_extension():
    response = client.post(
        "/documents/upload",
        files=[("files", ("malware.exe", b"binary", "application/octet-stream"))],
    )
    assert response.status_code == 400


def test_upload_rejects_oversized_file():
    response = client.post(
        "/documents/upload",
        files=[("files", ("big.txt", b"x" * (6 * 1024 * 1024), "text/plain"))],
    )
    assert response.status_code == 400


def test_upload_rejects_invalid_utf8():
    response = client.post(
        "/documents/upload",
        files=[("files", ("bad.txt", b"\xff\xfe\x00\x01", "text/plain"))],
    )
    assert response.status_code == 400


def test_serves_static_ui():
    response = client.get("/")
    assert response.status_code == 200
    assert "Churn Console" in response.text


def _sample_dataset_csv(n=20, with_churn=True):
    import pandas as pd

    df = pd.read_csv("data/raw/Telco-Customer-Churn.csv").head(n)
    if not with_churn:
        df = df.drop(columns=["Churn"])
    return df.to_csv(index=False).encode("utf-8")


def test_predict_batch_with_labels_computes_accuracy():
    response = client.post(
        "/predict/batch", files=[("file", ("sample.csv", _sample_dataset_csv(30), "text/csv"))]
    )
    assert response.status_code == 200

    body = response.json()
    assert body["summary"]["total_rows"] == 30
    assert body["summary"]["accuracy"] is not None
    assert len(body["rows"]) == 30
    assert body["rows"][0]["customer_id"] == "7590-VHVEG"


def test_predict_batch_without_labels_omits_accuracy():
    response = client.post(
        "/predict/batch",
        files=[("file", ("sample.csv", _sample_dataset_csv(10, with_churn=False), "text/csv"))],
    )
    assert response.status_code == 200
    assert response.json()["summary"]["accuracy"] is None


def test_predict_batch_rejects_missing_columns():
    response = client.post(
        "/predict/batch", files=[("file", ("bad.csv", b"a,b,c\n1,2,3", "text/csv"))]
    )
    assert response.status_code == 400
    assert "missing required columns" in response.json()["detail"]


def test_predict_batch_rejects_non_csv():
    response = client.post(
        "/predict/batch", files=[("file", ("data.txt", b"hello", "text/plain"))]
    )
    assert response.status_code == 400
