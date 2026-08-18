FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./

# Install dependencies before copying application code, so editing source
# files doesn't force a full dependency reinstall on every rebuild. setuptools
# requires src/ to exist on disk for [tool.setuptools.packages.find] even
# before any real packages are copied into it.
RUN mkdir -p src
# CPU-only torch first: PyPI's default Linux wheel pulls the full CUDA stack
# (~7GB of libraries this container never uses since there's no GPU here).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

# Pre-download the embedding/generation models so the image is self-contained
# and the first /qa request doesn't pay a cold-start download.
RUN python -c "from sentence_transformers import SentenceTransformer; from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); AutoTokenizer.from_pretrained('google/flan-t5-base'); AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')"

COPY src/ src/
COPY api/ api/
RUN pip install --no-cache-dir --no-deps .

COPY models/ models/
COPY data/vector_store/ data/vector_store/
COPY documents/knowledge_base/ documents/knowledge_base/

ENV PYTHONPATH=/app \
    MODEL_PATH=/app/models \
    VECTOR_STORE_PATH=/app/data/vector_store \
    KNOWLEDGE_BASE_PATH=/app/documents/knowledge_base \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
