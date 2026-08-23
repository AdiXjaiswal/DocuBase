from src.data_loader import load_all_docs
from src.embeddings import EmbeddingPipeline
from src.vector_store import ChromadbStore
# from src.

if __name__ == "__main__":
    data_dir="data"
    # all_docs=load_all_docs(data_dir)
    store=ChromadbStore()
    # store.build_from_documents(all_docs)
    store.load()
    print(store.query("What is Decision Tree", top_k=6))