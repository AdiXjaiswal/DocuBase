from pathlib import Path
from typing import Any, List, Optional
from uuid import uuid4

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings import EmbeddingPipeline
from data_loader import load_all_docs

class ChromadbStore:
    def __init__(
        self,
        persist_dir: str = str("../data/vector_store"),
        collection_name: str = "test_collection",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = str(Path(persist_dir).resolve())
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Document embeddings for RAG"},
        )
        self.model = SentenceTransformer(embedding_model)
        print(f"Loaded embedding model: {embedding_model}")
        print(f"Using Chroma collection '{self.collection_name}' with {self.collection.count()} documents")

    def build_from_documents(self, documents: List[Any]):
        print(f"Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.gen_embeddings(chunks)
        metadatas = [dict(chunk.metadata) for chunk in chunks]
        self.add_embeddings(embeddings, metadatas, [chunk.page_content for chunk in chunks])
        print(f"Vector store built and saved to {self.persist_dir}")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: Optional[List[dict]] = None,
        documents: Optional[List[str]] = None,
    ):
        # embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional array")

        count = embeddings.shape[0]
        if metadatas is None:
            metadatas = [{} for _ in range(count)]
        if len(metadatas) != count:
            raise ValueError("Number of metadata entries must match embeddings")
        if documents is None:
            documents = [metadata.pop("text", "") for metadata in metadatas]
        if len(documents) != count:
            raise ValueError("Number of documents must match embeddings")

        self.collection.add(
            ids=[f"doc_{uuid4().hex}" for _ in range(count)],
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=documents,
        )
        print(f"[INFO] Added {count} vectors to Chroma collection.")

    def load(self):
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_collection(name=self.collection_name)
        print(f"[INFO] Loaded {self.collection.count()} documents from Chroma.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        collection_count = self.collection.count()
        if collection_count == 0:
            raise ValueError(
                f"Cannot search empty Chroma collection '{self.collection_name}'. "
                "Build the vector store before querying."
            )

        query_embedding = np.asarray(query_embedding, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise ValueError("query_embedding must contain exactly one vector")

        result = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(top_k, collection_count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "id": doc_id,
                "distance": distance,
                "document": document,
                "metadata": metadata,
            }
            for doc_id, distance, document, metadata in zip(
                result["ids"][0],
                result["distances"][0],
                result["documents"][0],
                result["metadatas"][0],
            )
        ]

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text])
        return self.search(query_emb, top_k=top_k)

    def clear(self):
        """Delete all documents from this Chroma collection."""
        ids = self.collection.get(include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)
        print(f"[INFO] Cleared Chroma collection '{self.collection_name}'.")

if __name__ == "__main__":
    # docs = load_all_docs("../data")
    store = ChromadbStore("../data/vector_store")
    # store.build_from_documents(docs)
    store.load()
    print(store.query("What is Decision Tree?", top_k=6))
    
    # files = [x for x in Path("../data").iterdir() if x.is_file()]
    # print(files)