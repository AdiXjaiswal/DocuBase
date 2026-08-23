import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, JSONLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from pathlib import Path
from typing import List, Any


def load_all_docs(data_dir: str) -> List[Any]:
    '''
    Load all supported documents types and convert into LangChain document structure.
    Supported: pdf, csv, docx, txt, excel, json
    '''

    loaders_map={
        '.pdf': PyMuPDFLoader,
        '.csv': CSVLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader,
        '.json': JSONLoader,
        '.xlsx': UnstructuredExcelLoader
    }
    data_path=Path(data_dir)
    documents=[]

    for file in data_path.rglob("*"): # rglob("*") searches subfolders also
        if file.suffix.lower() in loaders_map:
            try:
                doc=loaders_map[file.suffix.lower()](str(file)).load() #load files using loader classes stored in loaders_map
                print(f"Loaded: {file}")
                documents.extend(doc)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents

# Example usage
if __name__ == "__main__":
    docs = load_all_docs("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)