from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from rag.retriever import Retriever

GENERATION_MODEL = "google/flan-t5-base"
MIN_RELEVANCE_SCORE = 0.25  # cosine similarity (normalized embeddings); below this, context is treated as irrelevant

PROMPT_TEMPLATE = (
    "Answer the question using only the information in the context below. "
    "If the answer isn't in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


class QAEngine:
    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)

    def answer(self, question: str, top_k: int = 3) -> dict:
        results = self.retriever.retrieve(question, top_k=top_k)
        relevant = [r for r in results if r["score"] >= MIN_RELEVANCE_SCORE]

        if not relevant:
            return {
                "question": question,
                "answer": "I don't know — the knowledge base doesn't contain information relevant to this question.",
                "sources": [],
            }

        context = "\n\n".join(r["text"] for r in relevant)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output_ids = self.model.generate(**inputs, max_new_tokens=200)
        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "question": question,
            "answer": generated.strip(),
            "sources": [{"source": r["source"], "score": round(r["score"], 4)} for r in relevant],
        }


qa_engine: QAEngine | None = None


def get_qa_engine() -> QAEngine:
    global qa_engine
    if qa_engine is None:
        qa_engine = QAEngine()
    return qa_engine


def refresh_knowledge_base() -> None:
    """Reload the on-disk index into the already-loaded QA engine, if one exists,
    so newly ingested documents are queryable without restarting the process."""
    if qa_engine is not None:
        qa_engine.retriever.reload_index()
