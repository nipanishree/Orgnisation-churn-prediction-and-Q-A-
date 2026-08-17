import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.ingest import EMBEDDING_MODEL, VECTOR_STORE_DIR


class Retriever:
    def __init__(self, vector_store_dir: Path = VECTOR_STORE_DIR):
        index_path = vector_store_dir / "index.faiss"
        chunks_path = vector_store_dir / "chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No vector store found at {vector_store_dir}. Run `python -m rag.ingest` first."
            )

        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")

        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({"source": chunk["source"], "text": chunk["text"], "score": float(score)})
        return results
