import logging
import json
import re
from typing import Any, Type, List, Optional
from enum import Enum
import asyncio
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from pydantic import Field, BaseModel
from src.mcp import MCPManager

# Configure logging
logger = logging.getLogger(__name__)

# --- Fixed Pydantic Models ---
class SearchRoute(str, Enum):
    DOCUMENT_SEARCH = "document_search"
    WEB_SEARCH = "web_search"
    HYBRID_SEARCH = "hybrid_search"

class RouteDecision(BaseModel):
    route: SearchRoute
    reasoning: str

# --- Enhanced Query Decomposer ---
def create_query_decomposer(llm: BaseChatModel):
    """
    Creates a chain to break a query into sub-queries only when truly necessary.
    This aligns with the Prompt Chaining principle of breaking down complex tasks.
    """
    prompt = PromptTemplate(
        template="""You are an expert at analyzing queries to determine if they need decomposition.

**Rules for decomposition:**
1. SIMPLE queries (one clear question) should NOT be decomposed
2. ONLY decompose if the query contains multiple distinct, unrelated questions
3. Avoid decomposing queries that are naturally one question even if they seem complex

**Examples:**
- "What is Tesla's stock price?" → Tesla stock price (NO decomposition)
- "Weather in Tokyo today" → Weather in Tokyo today (NO decomposition)  
- "Tell me about Tesla's stock price AND the weather in London" → Tesla stock price, weather in London (decompose)
- "Compare the economies of Germany and Japan" → economy of Germany, economy of Japan (decompose)

**Query to analyze:** {query}

**Instructions:** - If the query is a single question (even if complex), return it unchanged
- If it contains multiple distinct questions, separate them with commas
- Output ONLY the query/queries, no explanations

Output:""",
        input_variables=["query"],
    )
    
    def smart_list_parser(text: str) -> List[str]:
        """Custom parser that handles both single queries and lists."""
        text = text.strip()
        if ',' in text:
            return [item.strip() for item in text.split(',') if item.strip()]
        else:
            return [text]
    
    return prompt | llm | StrOutputParser() | RunnableLambda(smart_list_parser)

# --- Simplified and More Accurate Query Router ---
def create_query_router(llm: BaseChatModel):
    """
    Creates a more accurate router that better understands when to use internal documents.
    """
    routing_prompt = PromptTemplate(
        template="""You are a query router. Determine the best search strategy based on this query.

**Available Sources:**
1. **Internal Documents**: WHO Management of Diabetes Mellitus Guidelines (1994)
   - Contains: diagnostic criteria, treatment protocols, complications, drug therapies
   - Covers: both IDDM and NIDDM diabetes management as of 1994

2. **Web Search**: Real-time information from the internet
   - Contains: current events, recent data, general knowledge, non-medical topics

**Routing Rules:**
- Use 'document_search' ONLY for diabetes-related medical questions that would be in 1994 WHO guidelines
- Use 'web_search' for everything else (stocks, weather, news, current events, non-diabetes topics)
- Use 'hybrid_search' ONLY when you need BOTH old guidelines AND current information

**Query:** {query}

Answer with ONE word only: document_search, web_search, or hybrid_search

Answer:""",
        input_variables=["query"],
    )
    return routing_prompt | llm | StrOutputParser()

class SmartSearchInput(BaseModel):
    """Input schema for the SmartSearchTool."""
    query: str = Field(description="The search query from the user.")

class SmartSearchTool(BaseTool):
    """
    Enhanced smart search tool with better error handling and more conservative decomposition.
    It orchestrates a chain of tools to answer a query.
    """
    name: str = "smart_search"
    description: str = "Advanced search tool for complex queries that need multiple information sources or when you're unsure which specific tool to use. Prefer specialized tools for simple, direct queries."
    args_schema: Type[BaseModel] = SmartSearchInput
    
    document_search_tool: Any
    web_search_connector_name: str
    mcp_manager: MCPManager
    query_router: Any
    query_decomposer: Any
    llm: BaseChatModel

    async def _execute_web_search(self, query: str) -> str:
        """Executes a web search via the MCP manager with enhanced fallback logging."""
        primary_engine = self.web_search_connector_name
        
        try:
            logger.info(f"Attempting web search with primary engine: {primary_engine}")
            response = await self.mcp_manager.execute_tool(primary_engine, {"query": query})
            
            if response.success:
                logger.info(f"Web search successful with {primary_engine}")
                return str(response.data)
            else:
                logger.warning(f"Primary engine {primary_engine} failed: {response.error}")
                
                # Try fallback search engines with detailed logging
                fallback_engines = ["tavily_web_search", "duckduckgo_web_search"]
                attempted_fallbacks = []
                
                for engine in fallback_engines:
                    if engine != primary_engine and engine in self.mcp_manager.connectors:
                        try:
                            logger.info(f"Attempting fallback to {engine}")
                            attempted_fallbacks.append(engine)
                            
                            fallback_response = await self.mcp_manager.execute_tool(engine, {"query": query})
                            
                            if fallback_response.success:
                                logger.info(f"Fallback successful with {engine}")
                                return str(fallback_response.data)
                            else:
                                logger.warning(f"Fallback engine {engine} also failed: {fallback_response.error}")
                        except Exception as fallback_error:
                            logger.error(f"Exception during fallback to {engine}: {fallback_error}")
                            continue
                
                # All attempts failed
                error_msg = (
                    f"Web search failed. Primary engine: {primary_engine} ({response.error}). "
                    f"Attempted fallbacks: {', '.join(attempted_fallbacks) if attempted_fallbacks else 'None available'}. "
                    f"All search engines exhausted."
                )
                logger.error(error_msg)
                return error_msg
                
        except Exception as e:
            logger.error(f"Critical error in web search for '{query}': {e}", exc_info=True)
            return f"Web search critical error: {str(e)}"

    async def _execute_document_search(self, query: str) -> str:
        """Executes document search with error handling."""
        try:
            logger.info(f"Executing document search for: {query}")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.document_search_tool.invoke, query)
            
            if result:
                logger.info(f"Document search successful for: {query}")
                return str(result)
            else:
                logger.warning(f"No relevant documents found for: {query}")
                return "No relevant documents found."
        except Exception as e:
            logger.error(f"Document search error for '{query}': {e}", exc_info=True)
            return f"Document search error: {str(e)}"

    async def _execute_hybrid_search(self, query: str) -> str:
        """Executes both document and web search concurrently."""
        try:
            logger.info(f"Executing hybrid search for: {query}")
            
            doc_task = self._execute_document_search(query)
            web_task = self._execute_web_search(query)
            
            doc_results, web_results = await asyncio.gather(doc_task, web_task, return_exceptions=True)
            
            # Handle potential exceptions with detailed logging
            if isinstance(doc_results, Exception):
                logger.error(f"Document search failed in hybrid mode: {doc_results}")
                doc_results = f"Document search failed: {str(doc_results)}"
            
            if isinstance(web_results, Exception):
                logger.error(f"Web search failed in hybrid mode: {web_results}")
                web_results = f"Web search failed: {str(web_results)}"
            
            logger.info(f"Hybrid search completed for: {query}")
            return f"**Internal Documents:**\n{doc_results}\n\n**Web Search:**\n{web_results}"
        except Exception as e:
            logger.error(f"Hybrid search error for '{query}': {e}", exc_info=True)
            return f"Hybrid search error: {str(e)}"

    async def _arun(self, query: str) -> str:
        """The asynchronous execution logic for the tool."""
        try:
            logger.info(f"SmartSearchTool invoked with query: {query}")
            
            # Step 1: Conservative decomposition
            sub_queries = self.query_decomposer.invoke({"query": query})
            
            # Fallback if decomposer fails
            if not sub_queries or not isinstance(sub_queries, list):
                logger.warning(f"Query decomposer failed, using original query")
                sub_queries = [query]
            
            logger.info(f"Query '{query}' decomposed into {len(sub_queries)} sub-queries: {sub_queries}")

            # Step 2: Route and execute each sub-query
            results = []
            for idx, sub_q in enumerate(sub_queries, 1):
                sub_q = sub_q.strip()
                if not sub_q:
                    logger.warning(f"Empty sub-query #{idx}, skipping")
                    continue

                try:
                    logger.info(f"Processing sub-query #{idx}/{len(sub_queries)}: {sub_q}")
                    
                    raw_decision = self.query_router.invoke({"query": sub_q}).strip().lower()
                    logger.info(f"Router decision for '{sub_q}': {raw_decision}")
                    
                    if "document_search" in raw_decision:
                        result = await self._execute_document_search(sub_q)
                    elif "hybrid_search" in raw_decision:
                        result = await self._execute_hybrid_search(sub_q)
                    else:  # Default to web search
                        result = await self._execute_web_search(sub_q)
                    
                    results.append(f"**Query:** {sub_q}\n**Result:** {result}")
                    logger.info(f"Sub-query #{idx} completed successfully")
                    
                except Exception as e:
                    logger.error(f"Error processing sub-query '{sub_q}': {e}", exc_info=True)
                    # Fallback to web search
                    logger.info(f"Attempting fallback web search for failed sub-query: {sub_q}")
                    result = await self._execute_web_search(sub_q)
                    results.append(f"**Query:** {sub_q}\n**Result:** {result}")
            
            if not results:
                logger.error("No results obtained from any sub-query")
                return "I apologize, but I couldn't retrieve any information for your query. Please try rephrasing."
            
            # Step 3: Synthesize results if multiple sub-queries
            if len(results) == 1:
                # Single query - return result directly without extra synthesis
                logger.info("Single sub-query result, returning directly")
                return results[0].split("**Result:** ", 1)[-1]
            else:
                # Multiple queries - synthesize
                logger.info(f"Synthesizing {len(results)} sub-query results")
                combined_results = "\n\n".join(results)
                
                synthesis_prompt = PromptTemplate(
                    template="""Synthesize the following search results to provide a comprehensive answer to the original query.

**Original Query:** {original_query}

**Search Results:**
{search_results}

**Instructions:**
- Provide a direct, factual answer
- Combine relevant information from all results
- If results conflict, mention the discrepancy
- Keep the response focused and concise

**Synthesized Answer:**""",
                    input_variables=["original_query", "search_results"],
                )
                
                synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
                
                final_answer = synthesis_chain.invoke({
                    "original_query": query,
                    "search_results": combined_results
                })
                
                logger.info("Synthesis completed successfully")
                return final_answer

        except Exception as e:
            logger.error(f"Critical error in smart search for query '{query}': {e}", exc_info=True)
            # Final fallback - try simple web search
            try:
                logger.info("Attempting final fallback web search")
                return await self._execute_web_search(query)
            except Exception as fallback_error:
                logger.error(f"Final fallback also failed: {fallback_error}", exc_info=True)
                return f"I apologize, but I encountered an error while searching for information about '{query}'. Please try rephrasing your question or try again later."

    def _run(self, query: str) -> str:
        """Synchronous wrapper for the async execution logic."""
        try:
            return asyncio.run(self._arun(query))
        except Exception as e:
            logger.error(f"Error in synchronous smart search: {e}", exc_info=True)
            return f"I encountered an error while processing your query. Please try again."

# --- Additional Helper Functions ---
def create_smart_search_with_fallbacks(document_search_tool, mcp_manager, llm, primary_web_search="tavily_web_search"):
    """
    Factory function to create a robust SmartSearchTool with all dependencies.
    """
    try:
        logger.info("Creating SmartSearchTool with fallbacks")
        query_router = create_query_router(llm)
        query_decomposer = create_query_decomposer(llm)
        
        tool = SmartSearchTool(
            document_search_tool=document_search_tool,
            web_search_connector_name=primary_web_search,
            mcp_manager=mcp_manager,
            query_router=query_router,
            query_decomposer=query_decomposer,
            llm=llm
        )
        
        logger.info("SmartSearchTool created successfully")
        return tool
    except Exception as e:
        logger.error(f"Failed to create SmartSearchTool: {e}", exc_info=True)
        raise