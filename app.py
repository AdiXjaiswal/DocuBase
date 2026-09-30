from src import load_all_docs
from src import ChromadbStore, SearchPipeline

if __name__ == "__main__":
    data_dir="data"
    all_docs=load_all_docs(data_dir)
    store=ChromadbStore()
    store.build_from_documents(all_docs)
    store.load()
    # print(store.query("What is Decision Tree", top_k=6))
    search=SearchPipeline()
    while(True):
        query=input("Enter your query or type 'exit' to quit: ")
        if query.lower() == "exit":
            break
        answer = search.rag_simple(
            query,
            rag_retriever=store,
            top_k=6,
        )
        print(f"Answer:\n{answer}")