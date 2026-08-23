import numpy as np
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
# from src.data_loader import load_all_docs

class EmbeddingPipeline:
    def __init__(self, model_name: str="all-MiniLM-L6-v2", chunk_size: int=1000, chunk_overlap: int=200):
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.model=SentenceTransformer(model_name)
        print(f"\nLoaded model: {model_name}")

    def chunk_documents(self, documents: List[Any]) ->List[Any]:
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks=splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks\n")
        
        return chunks

    def gen_embeddings(self, chunks: List[str]) -> np.ndarray:
        '''
        Generate embeddings for a list of texts.
        Args:
            texts (List[str]): List of text strings to embed.
        Returns:
            np.ndarray: Array of embeddings, with shape of (len(texts), embedding_dimension).
        '''
        if not self.model:
            raise ValueError("Model is not loaded.")
        texts=[chunk.page_content for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} texts.")
        embeddings=self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings