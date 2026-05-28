
import logging
from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class RAGFusionExpander:
    """
    Expands a user query into multiple variations to improve retrieval coverage.
    This is a technique known as 'RAG Fusion' or 'Query Expansion'.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def generate_queries(self, original_query: str, num_queries: int = 4) -> List[str]:
        """
        Generates alternative versions of the query.
        """
        logger.info(f"Expanding query: '{original_query}'")
        
        prompt = PromptTemplate(
            input_variables=["query", "num"],
            template="""You are an AI assistant helping with information retrieval. 
Generate {num} different search queries based on the user's original query.
Focus on different aspects, keywords, or related concepts to maximize search coverage.
Return ONLY the queries, one per line. Do not number them.

Original Query: {query}

Generated Queries:"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({"query": original_query, "num": num_queries})
            # Split by line and clean up
            queries = [q.strip() for q in response.split('\n') if q.strip()]
            logger.info(f"Generated {len(queries)} variations: {queries}")
            
            # Always include the original query!
            if original_query not in queries:
                queries.insert(0, original_query)
                
            return queries
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [original_query]
