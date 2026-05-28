import os
# import lancedb
from typing import Any, List, Dict, Optional
from langchain_chroma import Chroma
from src.logging import logger


class ChromaDBVectorStoreManager:
    def __init__(self, embeddings: Any, db_path: str = "database/chromadb"):
        """
        Initialize the LanceDB Collection Manager.

        Args:
            embeddings: Embedding model instance.
            db_path: Path to the LanceDB database.
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.embeddings = embeddings

    def create_collection(self, documents: List[str], collection_name:str="rag_documents") -> Optional[Chroma]:
        """
        Create a new collection with documents.

        Args:
            documents: List of documents to add.
            mode: Mode for creating collection ('overwrite' or 'append').

        Returns:
            LanceDB vector store instance or None if creation fails.
        """
        if not documents:
            logger.error("Documents list cannot be empty.")
            return None
        print("++++++++",len(documents))
        try:
            vectorstore = Chroma.from_documents(
                documents=documents,
                collection_name=collection_name,
                embedding=self.embeddings,
                persist_directory=self.db_path,
            )
            logger.info(f"Created collection '{collection_name}' with {len(documents)} records.")
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return None


    def load_collection(self, collection_name:str) -> Optional[Chroma]:
        """
        Load an existing collection.

        Returns:
            LanceDB vector store instance or None if loading fails.
        """

        try:
            vectorstore = Chroma(
                # collection_name="example_collection",
                collection_name=collection_name,
                embedding_function=self.embeddings,
                # persist_directory="database/syx",
                persist_directory=self.db_path,
            )
            logger.info(f"Loaded collection '{collection_name}'.")
            return vectorstore
        except Exception as e:
            logger.error(f"Error loading collection: {e}")
            return None

