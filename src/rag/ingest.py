import json
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "documents" / "knowledge_base"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 350
CHUNK_OVERLAP = 60


def load_documents(knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR) -> list[dict]:
    docs = []
    for path in sorted(knowledge_base_dir.glob("**/*")):
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        docs.append({"source": path.relative_to(knowledge_base_dir).as_posix(), "text": path.read_text(encoding="utf-8")})
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for doc in docs:
        for text in splitter.split_text(doc["text"]):
            chunks.append({"source": doc["source"], "text": text})
    return chunks


def build_index(chunks: list[dict], model: SentenceTransformer) -> faiss.Index:
    embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def run(knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR, vector_store_dir: Path = VECTOR_STORE_DIR) -> dict:
    docs = load_documents(knowledge_base_dir)
    if not docs:
        raise FileNotFoundError(f"No .md/.txt documents found in {knowledge_base_dir}")

    chunks = chunk_documents(docs)
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = build_index(chunks, model)

    vector_store_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(vector_store_dir / "index.faiss"))
    with open(vector_store_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    with open(vector_store_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"embedding_model": EMBEDDING_MODEL, "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
                    "num_documents": len(docs), "num_chunks": len(chunks)}, f, indent=2)

    return {"num_documents": len(docs), "num_chunks": len(chunks)}


if __name__ == "__main__":
    stats = run()
    print(f"Indexed {stats['num_documents']} document(s) into {stats['num_chunks']} chunks -> {VECTOR_STORE_DIR}")
