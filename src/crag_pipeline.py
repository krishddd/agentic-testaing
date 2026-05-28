import os
import json
import re
# from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel
from src.logging import logger
from typing import List, Tuple, Optional, Any

# Load environment variables
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# --- Pydantic Models for Structured LLM Output ---
class RetrievalEvaluatorInput(BaseModel):
    """Model for capturing the relevance score of a document to a query."""
    score: float = Field(
        ...,
        description="Relevance score between 0 and 1, indicating how relevant the document is to the query."
    )

class DocumentScores(BaseModel):
    """A list of scores for multiple documents."""
    scores: List[float] = Field(
        ...,
        min_items=1,
        description="A list of relevance scores, one for each document."
    )

class QueryRewriterInput(BaseModel):
    """Model for capturing a rewritten query suitable for web search."""
    query: str = Field(..., description="The query rewritten for better web search results.")

class KnowledgeRefinementInput(BaseModel):
    """Model for extracting key points from a document."""
    key_points: str = Field(..., description="Key information extracted from the document in bullet-point form.")

class CRAGPipeline:
    """
    An optimized pipeline for Corrective RAG (CRAG) that handles document retrieval,
    batch evaluation, and knowledge refinement to improve response quality and speed.
    This is an excellent example of the Prompt Chaining pattern.
    """

    def __init__(self, llm: BaseChatModel, retriever: BaseRetriever, config: dict):
        """
        Initializes the CRAG Pipeline.
        """
        logger.info("--- Initializing CRAG Pipeline ---")
        self.llm = llm
        self.retriever = retriever
        self.lower_threshold = float(config.get('crag_lower_threshold', 0.3))
        self.upper_threshold = float(config.get('crag_upper_threshold', 0.7))
        self.search = DuckDuckGoSearchResults()

    def retrieve_documents(self, query: str) -> list[str]:
        """Uses the retriever to fetch documents."""
        docs = self.retriever.invoke(query)
        return [doc.page_content for doc in docs]

    def evaluate_documents(self, query: str, documents: list[str]) -> list[float]:
        """
        Evaluates the relevance of all documents in a single batch call to the LLM.
        This is much more efficient than evaluating them one by one.
        """
        if not documents:
            return []

        # Create a numbered list of documents for the prompt
        doc_list_str = "\n\n".join([f"--- Document {i+1} ---\n{doc}" for i, doc in enumerate(documents)])

        prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template="""You are an expert relevance evaluator. Assess each document below to determine how relevant it is to the user's query.
            Your response MUST be a JSON object containing a 'scores' array with one floating-point score (0.0 to 1.0) for each document.

            User Query: {query}

            Documents:
            {documents}
            """
        )
        chain = prompt | self.llm.with_structured_output(DocumentScores)
        try:
            result = chain.invoke({"query": query, "documents": doc_list_str})
            return result.scores
        except Exception as e:
            logger.error(f"Failed to evaluate documents in batch: {e}")
            # Fallback to individual scoring if batch fails
            return [self._retrieval_evaluator_single(query, doc) for doc in documents]

    def _retrieval_evaluator_single(self, query: str, document: str) -> float:
        """Fallback method to score a single document's relevance if batch processing fails."""
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="On a scale from 0 to 1, how relevant is this document to the query? "
                     "Query: {query}\nDocument: {document}\n"
        )
        chain = prompt | self.llm.with_structured_output(RetrievalEvaluatorInput)
        try:
            result = chain.invoke({"query": query, "document": document})
            return result.score
        except Exception as e:
            logger.error(f"Failed to evaluate single document: {e}")
            return 0.0

    def rewrite_query(self, query: str) -> str:
        """Rewrites the query for better web search results."""
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Rewrite the following query to be optimal for a web search engine:\n{query}"
        )
        chain = prompt | self.llm.with_structured_output(QueryRewriterInput)
        return chain.invoke({"query": query}).query.strip()

    def perform_web_search(self, query: str) -> tuple[str, list]:
        """Performs a web search and refines the knowledge."""
        rewritten_query = self.rewrite_query(query)
        web_results_str = self.search.run(rewritten_query)
        web_knowledge = self._knowledge_refinement(web_results_str)
        sources = self._parse_search_results(web_results_str)
        return web_knowledge, sources

    def _knowledge_refinement(self, document: str) -> str:
        """Extracts key points from a document."""
        prompt = PromptTemplate(
            input_variables=["document"],
            template="Extract the key information from the following text in clear bullet points:\n{document}"
        )
        chain = prompt | self.llm.with_structured_output(KnowledgeRefinementInput)
        return chain.invoke({"document": document}).key_points

    def _parse_search_results(self, results_string: str) -> list:
        """
        Robustly parses the string from DuckDuckGo search results.
        It first tries to parse it as JSON, and if that fails, it uses regex as a fallback.
        """
        try:
            results = json.loads(results_string)
            return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]
        except json.JSONDecodeError:
            logger.warning("JSON parsing of web search results failed. Falling back to regex.")
            pattern = re.compile(r"snippet: (.*?),\s*title: (.*?),\s*link: (.*?)(?=\s*snippet:|\Z)", re.DOTALL)
            matches = pattern.findall(results_string)
            if matches:
                return [(title.strip(), link.strip()) for snippet, title, link in matches]
            logger.error("Error parsing web search results with both JSON and regex.")
            return []

    def generate_response(self, query: str, knowledge: str, sources: list) -> str:
        """Generates the final answer based on the refined knowledge."""
        source_str = "\n".join([f"- {title}: {link}" if link else f"- {title}" for title, link in sources])
        prompt = PromptTemplate(
            input_variables=["query", "knowledge", "sources"],
            template="""Based on the provided knowledge, give a detailed and helpful answer to the user's query.
            List the sources you used at the end of your answer.

            Knowledge:
            {knowledge}

            Sources:
            {sources}

            Query: {query}

            Helpful Answer:"""
        )
        chain = prompt | self.llm
        response = chain.invoke({"query": query, "knowledge": knowledge, "sources": source_str})
        return response.content.strip()

    def invoke(self, query: str, web_search_results: Optional[Tuple[str, list]] = None) -> dict:
        """
        Main execution function for the CRAG pipeline.
        
        Args:
            query (str): The user's query.
            web_search_results (Optional[Tuple[str, list]]): Pre-computed web search results to avoid redundant searches.
        """
        logger.info(f"Processing query with CRAG: '{query}'")

        final_knowledge = ""
        sources = []
        action = ""
        max_score = 0.0

        retrieved_docs = self.retrieve_documents(query)

        if not retrieved_docs:
            logger.warning("No documents retrieved. Proceeding to web search.")
            action = "Incorrect (No documents found, performing web search)"
            if web_search_results:
                final_knowledge, sources = web_search_results
            else:
                final_knowledge, sources = self.perform_web_search(query)
        else:
            eval_scores = self.evaluate_documents(query, retrieved_docs)
            logger.info(f"Retrieved {len(retrieved_docs)} documents with scores: {eval_scores}")

            if eval_scores:
                max_score = max(eval_scores)
                best_doc_index = eval_scores.index(max_score)
                best_doc = retrieved_docs[best_doc_index]

                if max_score > self.upper_threshold:
                    action = "Correct (Using retrieved document)"
                    final_knowledge = best_doc
                    sources.append(("Retrieved document (Local Source)", ""))
                elif max_score < self.lower_threshold:
                    action = "Incorrect (Performing web search)"
                    if web_search_results:
                        final_knowledge, sources = web_search_results
                    else:
                        final_knowledge, sources = self.perform_web_search(query)
                else:
                    action = "Ambiguous (Combining retrieved document and web search)"
                    retrieved_knowledge = self._knowledge_refinement(best_doc)
                    if web_search_results:
                        web_knowledge, web_sources = web_search_results
                    else:
                        web_knowledge, web_sources = self.perform_web_search(query)
                    final_knowledge = f"From Local Documents:\n{retrieved_knowledge}\n\nFrom Web Search:\n{web_knowledge}"
                    sources = [("Retrieved document (Local Source)", "")] + web_sources
            else:
                action = "Incorrect (Evaluation failed, performing web search)"
                if web_search_results:
                    final_knowledge, sources = web_search_results
                else:
                    final_knowledge, sources = self.perform_web_search(query)

        logger.info(f"Action Taken: {action}")
        logger.info("Generating final response...")
        final_answer = self.generate_response(query, final_knowledge, sources)

        return {
            "final_answer": final_answer,
            "action_taken": action,
            "max_relevance_score": max_score,
            "sources": sources
        }
