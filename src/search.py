import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.vector_store import ChromadbStore

load_dotenv()


def create_llm() -> ChatOpenAI:
    """Create the NVIDIA-compatible chat model from environment variables."""
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("NVIDIA_BASE_URL")
    model_name = os.getenv("MODEL_NAME")

    missing = [
        name
        for name, value in (
            ("NVIDIA_API_KEY", api_key),
            ("NVIDIA_BASE_URL", base_url),
            ("MODEL_NAME", model_name),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing environment variables: " + ", ".join(missing)
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.1,
        max_tokens=1024,
    )


def _format_context(results: list[dict[str, Any]]) -> str:
    """Format retrieved chunks and their sources for the model."""
    return "\n\n".join(
        f"[Source {index}] {result['document']}\n"
        f"Source metadata: {result.get('metadata', {})}"
        for index, result in enumerate(results, start=1)
    )


def rag_simple(
    query: str,
    rag_retriever: Optional[ChromadbStore] = None,
    llm: Optional[ChatOpenAI] = None,
    top_k: int = 3,
) -> str:
    """Retrieve relevant chunks and answer the query using the LLM."""
    if not query.strip():
        raise ValueError("query must not be empty")

    retriever = rag_retriever or ChromadbStore()
    model = llm or create_llm()
    results = retriever.query(query, top_k=top_k)

    if not results:
        return "I could not find relevant information in the document collection."

    prompt = f"""You answer questions using only the provided document context.
If the answer is not in the context, say you do not know. Do not invent facts.
Cite the relevant source numbers in your answer, for example [Source 1].

Question:
{query}

Document context:
{_format_context(results)}
"""

    response = model.invoke(prompt)
    return str(response.content)


if __name__ == "__main__":
    store = ChromadbStore()
    store.load()
    answer = rag_simple(
        "What is a Decision Tree?",
        rag_retriever=store,
        top_k=3,
    )
    print(f"Answer:\n{answer}")
