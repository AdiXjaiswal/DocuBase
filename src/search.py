import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from vector_store import ChromadbStore

load_dotenv()

class SearchPipeline:
    def __init__(
        self, 
        api_key="NVIDIA_API_KEY",
        base_url="NVIDIA_BASE_URL",
        model_name="MODEL_NAME"
        ):
        self.api_key=os.getenv(api_key)
        self.base_url=os.getenv(base_url)
        self.model_name=os.getenv(model_name)
    
    def create_llm(self, temp: int = 0.1, tokens: int=1024) -> ChatOpenAI:
        """Create the NVIDIA-compatible chat model from environment variables."""
        return ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model_name,
            temperature=temp,
            max_tokens=tokens,
        )
    
    def _format_context(self, results: list[dict[str, Any]]) -> str:
        """Format retrieved chunks and their sources for the model."""
        return "\n\n".join(
            f"[Source {index}] {result['document']}\n"
            f"Source metadata: {result.get('metadata', {})}"
            for index, result in enumerate(results, start=1)
        )
    
    def rag_simple(
        self,
        query: str,
        rag_retriever: Optional[ChromadbStore] = None,
        llm: Optional[ChatOpenAI] = None,
        top_k: int = 5,
    ) -> str:
        """Retrieve relevant chunks and answer the query using the LLM."""
        if not query.strip():
            raise ValueError("query must not be empty")

        retriever = rag_retriever or ChromadbStore()
        model = llm or self.create_llm()
        results = retriever.query(query, top_k=top_k)

        if not results:
            return "I could not find relevant information in the document collection."

        prompt = f"""You answer questions using only the provided document context.
    If the answer is not in the context, say you do not know. Do not invent facts.
    Cite the relevant source numbers in your answer, for example [Source 1].

    Question:
    {query}

    Document context:
    {self._format_context(results)}
    """

        response = model.invoke(prompt)
        return str(response.content)


if __name__ == "__main__":
    store = ChromadbStore()
    store.load()
    search=SearchPipeline()
    answer = search.rag_simple(
        "What is a Multilayer Perceptron?",
        rag_retriever=store,
        top_k=6,
    )
    print(f"Answer:\n{answer}")
