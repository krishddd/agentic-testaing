import re
import os
import pickle
from typing import List
from langchain_core.documents import Document


def save_documents(documents, filename: str):
    """Saves extracted Document objects as a pickle file in the results directory."""
    result_path = os.path.join("results", 'document_data', filename)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "wb") as f:
        pickle.dump(documents, f)


def load_saved_docs(filename: str)-> List[Document]:
    """Loads saved Document objects from a pickle file."""
    result_path = os.path.join("results", 'document_data', filename)
    with open(result_path, "rb") as f:
        return pickle.load(f)