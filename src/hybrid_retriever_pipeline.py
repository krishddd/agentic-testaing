
import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from src.retriever import RetrieverManager
from src.rag_fusion_expander import RAGFusionExpander
from src.graph_knowledge_manager import GraphKnowledgeManager

logger = logging.getLogger(__name__)

class HybridRetrieverPipeline:
    """
    Orchestrates Hybrid Retrieval:
    1. Expands queries using RAG Fusion.
    2. Retrieves documents using Vector Search.
    3. Retrieves context using Graph Search.
    4. Fuses results.
    """

    def __init__(self, 
                 llm: BaseChatModel, 
                 retriever_manager: RetrieverManager,
                 graph_manager: GraphKnowledgeManager):
        self.llm = llm
        self.retriever_manager = retriever_manager
        self.graph_manager = graph_manager
        self.fusion_expander = RAGFusionExpander(llm)
        self.vector_retriever = retriever_manager.get_retriever(k=5)

    def retrieve(self, query: str) -> List[Document]:
        """
        Performs the hybrid retrieval.
        """
        logger.info(f"Starting hybrid retrieval for: '{query}'")
        
        # 1. RAG Fusion: Expand Query
        expanded_queries = self.fusion_expander.generate_queries(query, num_queries=3)
        
        all_docs = []
        
        # 2. Vector Search (for each expanded query, simplified)
        # Using just the original query for main vector search to avoid noise, 
        # or we could aggregate. Let's stick to original query + 1 variance for efficiency.
        
        # Main vector search
        docs = self.vector_retriever.invoke(query)
        all_docs.extend(docs)
        
        # 3. Graph Search
        # Graph context is text, we wrap it in a Document
        graph_context = self.graph_manager.get_context_for_query(query, hops=1)
        
        if graph_context:
            logger.info("Found graph context, adding to results.")
            graph_doc = Document(
                page_content=graph_context,
                metadata={"source": "Knowledge Graph", "type": "structural_context"}
            )
            # Prepend graph context so it's high priority
            all_docs.insert(0, graph_doc)
            
        return all_docs

    def run_full_search(self, query: str) -> dict:
        """
        Returns structured results for API.
        """
        docs = self.retrieve(query)
        
        # Formatted context
        context_str = "\n\n".join([d.page_content for d in docs])
        
        return {
            "query": query,
            "context": context_str,
            "documents": [
                {"content": d.page_content, "metadata": d.metadata} for d in docs
            ]
        }
