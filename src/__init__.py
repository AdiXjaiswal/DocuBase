"""
DocuBase - A document processing and retrieval system with vector embeddings.
Provides tools for loading documents, generating embeddings, and semantic search.
"""

from .data_loader import load_all_docs
from .embeddings import EmbeddingPipeline
from .vector_store import ChromadbStore
from .search import SearchPipeline

__version__ = "0.1.1"

__all__ = [
    "load_all_docs",
    "EmbeddingPipeline",
    "ChromadbStore",
    "SearchPipeline",
]