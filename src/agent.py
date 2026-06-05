from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        context_blocks = []
        for index, result in enumerate(results, start=1):
            source = result.get("metadata", {}).get("source", result.get("metadata", {}).get("doc_id", "unknown"))
            context_blocks.append(
                f"[{index}] Source: {source}\n"
                f"Score: {result.get('score', 0):.4f}\n"
                f"Content: {result.get('content', '')}"
            )

        context = "\n\n".join(context_blocks) if context_blocks else "No relevant context was retrieved."
        prompt = (
            "You are a retrieval-augmented knowledge base assistant. "
            "Answer the question using only the retrieved context. "
            "If the context is insufficient, say that the available documents do not contain enough information.\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer:"
        )
        return self.llm_fn(prompt)
