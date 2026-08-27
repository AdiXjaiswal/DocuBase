from src import load_all_docs
from src import EmbeddingPipeline, ChromadbStore, SearchPipeline

if __name__ == "__main__":
    data_dir="data"
    # all_docs=load_all_docs(data_dir)
    store=ChromadbStore()
    # store.build_from_documents(all_docs)
    store.load()
    # print(store.query("What is Decision Tree", top_k=6))
    search=SearchPipeline()
    answer = search.rag_simple(
            "What is a Multilayer Perceptron?",
            rag_retriever=store,
            top_k=6,
        )
    print(f"Answer:\n{answer}")