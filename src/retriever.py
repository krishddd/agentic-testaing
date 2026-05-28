import os
import pickle
import logging
from typing import List, Optional, Any

try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    try:
        from langchain_community.retrievers.ensemble import EnsembleRetriever
    except (ImportError, ModuleNotFoundError):
        # Lightweight fallback if neither package provides EnsembleRetriever
        from langchain_core.retrievers import BaseRetriever as _BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun as _CMFRR

        class EnsembleRetriever(_BaseRetriever):
            """Minimal EnsembleRetriever fallback: merges results from multiple retrievers."""
            retrievers: list
            weights: list = None

            def _get_relevant_documents(self, query: str, *, run_manager: _CMFRR = None):
                all_docs = []
                seen = set()
                for retriever in self.retrievers:
                    for doc in retriever.get_relevant_documents(query):
                        key = doc.page_content[:200]
                        if key not in seen:
                            seen.add(key)
                            all_docs.append(doc)
                return all_docs
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from src.logging import logger
from src.chroma_vectorstore import ChromaDBVectorStoreManager 
from src.model import Model


class ManualCrossEncoderReranker(BaseRetriever):
    """Manual reranker using sentence-transformers CrossEncoder directly.
    
    Inherits from BaseRetriever to be compatible with LangChain chains.
    """
    
    base_retriever: BaseRetriever
    cross_encoder: Any
    top_n: int = 5
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Retrieve documents and rerank with cross-encoder.
        
        This is the required method for BaseRetriever subclasses.
        """
        # Get initial documents from base retriever
        docs = self.base_retriever.get_relevant_documents(query)
        
        if not docs:
            return []
        
        # If we have fewer docs than top_n, return all
        if len(docs) <= self.top_n:
            return docs
        
        try:
            # Create query-document pairs for scoring
            pairs = [[query, doc.page_content] for doc in docs]
            
            # Get relevance scores from cross-encoder
            scores = self.cross_encoder.predict(pairs)
            
            # Combine docs with scores
            scored_docs = list(zip(docs, scores))
            
            # Sort by score (highest first)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_n and add scores to metadata
            reranked_docs = []
            for doc, score in scored_docs[:self.top_n]:
                doc.metadata['relevance_score'] = float(score)
                doc.metadata['reranked'] = True
                reranked_docs.append(doc)
            
            logger.info(f"[OK] Reranked {len(docs)} -> {len(reranked_docs)} docs (score range: {scores.min():.3f} to {scores.max():.3f})")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning top {self.top_n} original docs")
            return docs[:self.top_n]


class RetrieverManager:
    def __init__(self, embeddings: Any, config_dict, collection_name: str = "rag_documents"):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.db_type = config_dict["db_type"].lower()
        self.config_dict = config_dict
        self.vector_db = ChromaDBVectorStoreManager(embeddings=embeddings)
        self.vectorstore = self.vector_db.load_collection(collection_name=self.collection_name)
        
        if not self.vectorstore:
            raise ValueError(f"Failed to load vector store collection: {self.collection_name}")
        
        # Load cross-encoder for reranking
        try:
            model_obj = Model(config_dict)
            self.cross_encoder = model_obj.load_cross_encoder_model()
            logger.info("Cross-encoder loaded successfully for reranking")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}. Reranking will be disabled.")
            self.cross_encoder = None

    def _load_saved_docs(self) -> List[Document]:
        """Loads processed documents saved during ingestion."""
        path = "results/document_data/document_data.pkl"
        if not os.path.exists(path):
            logger.warning(f"No saved documents found for BM25 retriever at {path}")
            return []
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(f"Error loading pickled documents: {e}")
            return []
        
    def get_retriever(self, k: int = 10, enable_reranking: bool = True, top_n_rerank: int = 5) -> BaseRetriever:
        """
        Creates an enhanced retriever with:
        1. Hybrid retrieval (BM25 + Vector)
        2. Increased initial retrieval (k=10)
        3. Cross-encoder reranking
        4. Final top_n selection after reranking
        
        Args:
            k: Initial number of documents to retrieve (default: 10)
            enable_reranking: Whether to use cross-encoder reranking (default: True)
            top_n_rerank: Final number of documents after reranking (default: 5)
        """
        logger.info(f"Setting up enhanced retriever (k={k}, rerank={enable_reranking}, top_n={top_n_rerank})...")
        
        # Create vector retriever with larger k
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Load documents for BM25
        docs = self._load_saved_docs()
        
        if not docs:
            logger.warning("No documents for BM25, using only vector retriever.")
            base_retriever = vector_retriever
        else:
            # Create BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = k
            
            # Create ensemble retriever with balanced weights
            base_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]  # Equal weight for keyword and semantic search
            )
            logger.info(f"Ensemble retriever created with k={k}")
        
        # Add reranking if cross-encoder is available
        if enable_reranking and self.cross_encoder is not None:
            try:
                # Use manual reranker that inherits from BaseRetriever
                reranked_retriever = ManualCrossEncoderReranker(
                    base_retriever=base_retriever,
                    cross_encoder=self.cross_encoder,
                    top_n=top_n_rerank
                )
                
                logger.info(f"[OK] Reranking enabled: retrieving {k} docs, reranking to top {top_n_rerank}")
                return reranked_retriever
                
            except Exception as e:
                logger.error(f"Failed to create reranked retriever: {e}. Falling back to base retriever.")
                return base_retriever
        else:
            logger.info("Reranking disabled or unavailable, using base retriever")
            return base_retriever
    
    def get_custom_retriever(self, 
                            initial_k: int = 15,
                            final_top_n: int = 8,
                            bm25_weight: float = 0.5,
                            enable_reranking: bool = True) -> BaseRetriever:
        """
        Custom retriever configuration for specific use cases.
        
        Args:
            initial_k: Initial retrieval size
            final_top_n: Final number after reranking
            bm25_weight: Weight for BM25 (vector gets 1-bm25_weight)
            enable_reranking: Enable cross-encoder reranking
        """
        logger.info(f"Creating custom retriever: initial_k={initial_k}, top_n={final_top_n}")
        
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": initial_k})
        docs = self._load_saved_docs()
        
        if not docs:
            base_retriever = vector_retriever
        else:
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = initial_k
            
            base_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[bm25_weight, 1.0 - bm25_weight]
            )
        
        if enable_reranking and self.cross_encoder is not None:
            try:
                reranked_retriever = ManualCrossEncoderReranker(
                    base_retriever=base_retriever,
                    cross_encoder=self.cross_encoder,
                    top_n=final_top_n
                )
                logger.info(f"[OK] Custom reranking enabled: {initial_k} -> {final_top_n}")
                return reranked_retriever
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                return base_retriever
        
        return base_retriever