import sys
from pathlib import Path

# # Add src to python path
# sys.path.insert(0, r"j:\DocuBase\src")

from langchain_core.documents import Document
from src import ChromadbStore

def test_deduplication():
    # 1. Initialize temporary vector store
    # test_dir = Path("data/vector_store")
    store = ChromadbStore()
    store.clear()
    
    # 2. Create sample raw documents
    doc1_a = Document(page_content="Decision Tree is a supervised learning algorithm.", metadata={"source": "doc1.txt"})
    doc1_b = Document(page_content="It splits data into subsets based on features.", metadata={"source": "doc1.txt"})
    doc2 = Document(page_content="Random Forest is an ensemble method of Decision Trees.", metadata={"source": "doc2.txt"})
    
    print("\n--- Test 1: Initial Build ---")
    store.build_from_documents([doc1_a, doc1_b, doc2])
    print(f"Collection count: {store.collection.count()}")
    assert store.collection.count() == 3, f"Expected 3 chunks, got {store.collection.count()}"
    
    print("\n--- Test 2: Re-build without changes (Should Skip) ---")
    store.build_from_documents([doc1_a, doc1_b, doc2])
    print(f"Collection count: {store.collection.count()}")
    assert store.collection.count() == 3, f"Expected 3 chunks, got {store.collection.count()}"
    
    print("\n--- Test 3: Modify doc1 (Should replace old chunks) ---")
    doc1_modified = Document(page_content="Decision Trees split data into subsets using entropy or Gini impurity.", metadata={"source": "doc1.txt"})
    store.build_from_documents([doc1_modified, doc2])
    print(f"Collection count: {store.collection.count()}")
    assert store.collection.count() == 2, f"Expected 2 chunks after modification, got {store.collection.count()}"
    
    print("\n--- Test 4: Delete doc2 (Should purge doc2) ---")
    store.build_from_documents([doc1_modified])
    print(f"Collection count: {store.collection.count()}")
    assert store.collection.count() == 1, f"Expected 1 chunk after deletion, got {store.collection.count()}"
    
    print("\nSUCCESS: All vector store deduplication and update tests passed!")
    store.clear()

if __name__ == "__main__":
    test_deduplication()
