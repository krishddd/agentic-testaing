import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import json
import time
from functools import wraps
import os
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.chains import LLMMathChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_core.tools import Tool, BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import ShellTool
from langchain_core.runnables import RunnableConfig
from langchain_experimental.tools import PythonREPLTool
from src.retriever import RetrieverManager
from src.model import Model
from src.configuration import ConfigurationManager
from src.retriever import RetrieverManager
from src.smart_search import SmartSearchTool, create_query_router, create_query_decomposer
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.plan_and_execute_agent import create_plan_and_execute_graph
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from src.mcp import MCPManager, ConnectorResponse
from src.tool_schemas import (
    WebSearchInput, WikipediaInput, CalculatorInput, StockTickerInput,
    WeatherInput, NewsInput, ArXivInput,SymPySolverInput,DateTimeInput,TextAnalysisInput,WebScraperInput, FinalAnswerInput,SemanticScholarInput,PubMedInput,RedditSearchInput,YouTubeSearchInput,GitHubSearchInput
)
import json
import re
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from typing import List
from src.tool_schemas import FinalAnswerInput
from src.deep_research_agent import research as deep_research, ResearchConfig
from src.chain_of_thought_tracer import ChainOfThoughtTracer, ReasoningStepType
from src.tool_attribution import ToolAttributionTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def execute_deep_research(
    query: str,
    max_iterations: int = 3,
    config_dict: Optional[Dict] = None,
    cot_tracer: Optional[ChainOfThoughtTracer] = None,
    attribution_tracker: Optional[ToolAttributionTracker] = None
) -> Dict[str, Any]:
    """Executes deep research with MULTIPLE specialized tools."""
    try:
        if config_dict is None:
            config_manager = ConfigurationManager()
            config_dict = config_manager.configurations()

        model_obj = Model(config_dict)
        llm = model_obj.load_ollama_model()
        embedding = model_obj.load_ollama_embedding()
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        mcp_manager = MCPManager(llm=llm, tavily_api_key=tavily_api_key)
        retriever = RetrieverManager(embeddings=embedding, config_dict=config_dict).get_retriever()
        memory = _create_memory("buffer_window", llm)

        document_search_tool = ContextualRetrievalTool(
            retriever=retriever,
            conversation_memory=memory,
            llm=llm
        )

        # ========== CREATE MULTIPLE TOOLS ==========
        all_search_tools = {}
        
        # 1. Web Search (Primary)
        web_connectors = ["tavily_web_search", "duckduckgo_web_search"]
        for connector in web_connectors:
            if connector in mcp_manager.connectors:
                all_search_tools["web_search"] = SmartSearchTool(
                    name="web_search",
                    description="General web search",
                    document_search_tool=document_search_tool,
                    web_search_connector_name=connector,
                    fallback_engines=[c for c in web_connectors if c in mcp_manager.connectors],
                    mcp_manager=mcp_manager,
                    query_router=create_query_router(llm),
                    query_decomposer=create_query_decomposer(llm),
                    llm=llm
                )
                break
        
        # 2. Wikipedia
        if "wikipedia_search" in mcp_manager.connectors:
            async def wiki_search(query: str) -> str:
                response = await mcp_manager.execute_tool("wikipedia_search", {"query": query})
                return str(response.data) if response.success else response.error
            
            from langchain_core.tools import Tool
            all_search_tools["wikipedia"] = Tool(
                name="wikipedia_search",
                func=lambda q: asyncio.run(wiki_search(q)),
                description="Wikipedia encyclopedia search"
            )
        
        # 3. News
        if "news_headlines" in mcp_manager.connectors:
            async def news_search(query: str) -> str:
                response = await mcp_manager.execute_tool("news_headlines", {"topic": query})
                return str(response.data) if response.success else response.error
            
            from langchain_core.tools import Tool
            all_search_tools["news"] = Tool(
                name="news_headlines",
                func=lambda q: asyncio.run(news_search(q)),
                description="Recent news headlines"
            )
        
        # 4. ArXiv
        if "arxiv_search" in mcp_manager.connectors:
            async def arxiv_search(query: str) -> str:
                response = await mcp_manager.execute_tool("arxiv_search", {"query": query})
                return str(response.data) if response.success else response.error
            
            from langchain_core.tools import Tool
            all_search_tools["arxiv"] = Tool(
                name="arxiv_search",
                func=lambda q: asyncio.run(arxiv_search(q)),
                description="Academic papers search"
            )
        
        # 5. Document search
        all_search_tools["documents"] = document_search_tool
        
        logger.info(f"[OK] Available tools: {list(all_search_tools.keys())}")

        research_config = ResearchConfig(max_iterations=max_iterations, min_quality_score=7)
        logger.info(f"Starting deep research for: '{query}'")
        
        # Pass MULTIPLE tools
        result = await asyncio.to_thread(
            deep_research,
            query=query,
            search_tools=all_search_tools,  # ← CHANGED
            config=research_config,
            cot_tracer=cot_tracer,
            attribution_tracker=attribution_tracker
        )

        if 'final_answer' in result:
            result['final_answer'] = re.sub(r'\\n+', '\n', result['final_answer'])
        return result

    except Exception as e:
        logger.error(f"Error in deep research: {e}", exc_info=True)
        return {
            "final_answer": f"Error: {str(e)}",
            "agent_steps": [],
            "agent_type": "enhanced_deep_research",
            "metadata": {"error": str(e)}
        }
class PerformanceCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to track performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'total_calls': 0,
            'tool_usage': {},
            'response_times': [],
            'errors': 0,
            'start_time': None
        }
    
    def on_agent_action(self, action, **kwargs):
        tool_name = action.tool
        self.metrics['tool_usage'][tool_name] = self.metrics['tool_usage'].get(tool_name, 0) + 1
        self.metrics['total_calls'] += 1
        logger.info(f"Using tool: {tool_name}")
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        self.metrics['start_time'] = time.time()
    
    def on_chain_end(self, outputs, **kwargs):
        if self.metrics['start_time']:
            response_time = time.time() - self.metrics['start_time']
            self.metrics['response_times'].append(response_time)
    
    def on_chain_error(self, error, **kwargs):
        self.metrics['errors'] += 1
        logger.error(f"Chain error: {error}")

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}")
                        raise
                    wait_time = delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

class ContextualRetrievalTool(BaseTool):
    """
    Enhanced retrieval tool with contextual awareness and a robust query rewriter.
    This applies the 'Context Engineering' principle by using chat history to refine the query.
    """
    
    name: str = "contextual_document_search"
    description: str = "Searches internal documents using conversation context to improve the query."
    
    retriever: object
    conversation_memory: object
    llm: Any # Pass the LLM instance during initialization

    def _create_query_rewriter(self):
        """Creates a chain to rewrite queries based on context with a stricter prompt."""
        # MODIFIED: Stricter prompt to ensure only the query is returned
        rewrite_template = """Based on the chat history, rewrite the user's question to be a standalone, specific, and searchable query.
If the question is already specific, you can return it as is.

Chat History:
{chat_history}

User Question: {query}

IMPORTANT: Your output MUST be ONLY the rewritten query text and nothing else.
Rewritten Query:"""
        prompt = PromptTemplate(
            input_variables=["chat_history", "query"],
            template=rewrite_template,
        )
        return prompt | self.llm | StrOutputParser()

    def _run(self, query: str, run_manager=None) -> str:
        """Enhanced retrieval with query expansion and context"""
        try:
            rewriter_chain = self._create_query_rewriter()
            chat_history_str = "\n".join([msg.content for msg in self.conversation_memory.chat_memory.messages])
            
            rewritten_query = rewriter_chain.invoke({
                "chat_history": chat_history_str,
                "query": query
            }).strip()

            if rewritten_query.startswith('"') and rewritten_query.endswith('"'):
                rewritten_query = rewritten_query[1:-1]

            logger.info(f"Original query: '{query}' | Rewritten query: '{rewritten_query}'")
            
            docs = self.retriever.invoke(rewritten_query)
            
            if not docs:
                return "No relevant documents found for your query."
            
            results = [f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content[:500]}..." for i, doc in enumerate(docs[:5])]
            
            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"Error in contextual retrieval: {e}")
            return f"Error retrieving documents: {str(e)}"
    
    def _expand_query(self, query: str) -> str:
        """Expand query based on conversation context"""
        if self.conversation_memory and hasattr(self.conversation_memory, 'chat_memory'):
            recent_messages = self.conversation_memory.chat_memory.messages[-4:]  # Last 2 exchanges
            context = " ".join([msg.content for msg in recent_messages])
            
            if len(context) > 50:
                return f"{query} Context: {context[:200]}"
        
        return query

class CachingTool(BaseTool):
    """Tool wrapper with caching capabilities"""
    
    base_tool: BaseTool
    name: str
    description: str
    cache: dict = {}
    cache_ttl: int = 300
    
    def _run(self, query: str, run_manager=None) -> str:
        """Run with caching"""
        cache_key = hash(query)
        current_time = time.time()
        
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_ttl:
                logger.info(f"Cache hit for {self.base_tool.name}")
                return result
        
        result = self.base_tool._run(query, run_manager)
        self.cache[cache_key] = (result, current_time)
        
        return result

def create_custom_tools(llm, config_dict: Dict) -> List[Tool]:
    """Create custom tools with advanced capabilities."""
    tools = []
    
    try:
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        cached_wiki = CachingTool(
            base_tool=wikipedia, 
            name=f"cached_{wikipedia.name}",
            description=wikipedia.description,
            args_schema=WikipediaInput
        )
        tools.append(cached_wiki)
    except Exception as e:
        logger.warning(f"Could not initialize Wikipedia tool: {e}")
    
    if config_dict.get('enable_code_execution', False):
        try:
            python_repl = PythonREPLTool()
            tools.append(python_repl)
        except Exception as e:
            logger.warning(f"Could not initialize Python REPL: {e}")
    
    time_tool = Tool(
        name="datetime_info",
        description="Get current date, time, and perform date calculations",
        func=_get_datetime_info,
        args_schema=DateTimeInput
    )
    tools.append(time_tool)
    
    text_analysis_tool = Tool(
        name="text_analyzer",
        description="Analyze text for sentiment, keywords, readability, and other metrics. Requires 'text' parameter.",
        func=lambda x: _analyze_text(x, llm),
        args_schema=TextAnalysisInput
    )
    tools.append(text_analysis_tool)
    
    return tools

def _safe_math_calculation(expression: any, llm) -> str:
    """
    Enhanced mathematical calculation function that properly handles step substitutions.
    """
    try:
        if isinstance(expression, dict) and 'expression' in expression:
            expression_str = expression['expression']
        elif isinstance(expression, str):
            expression_str = expression
        else:
            return "Error: Invalid input type for calculator."

        current_year = datetime.now().year
        
        if '2023 - birth_year' in expression_str:
            expression_str = expression_str.replace('2023', str(current_year))
        if '2025 - birth_year' in expression_str:
            expression_str = expression_str.replace('2025', str(current_year))
            
        calc_match = re.match(r'(\d{4})\s*-\s*(\d{4})', expression_str)
        if calc_match:
            year1, year2 = int(calc_match.group(1)), int(calc_match.group(2))
            result = year1 - year2
            return f"Calculation result: {result}"
        
        math_prompt = PromptTemplate.from_template(
            """Solve this mathematical expression and provide only the numerical result.
            
            Expression: {expression}
            
            Provide only the final number, no explanations:"""
        )
        
        math_chain = LLMChain(llm=llm, prompt=math_prompt, output_parser=StrOutputParser())
        result_str = math_chain.invoke({"expression": expression_str})
        
        match = re.search(r'-?\d+\.?\d*', result_str)
        if match:
            return f"Calculation result: {match.group(0)}"
        else:
            return f"Could not extract numerical result from: {result_str}"
            
    except Exception as e:
        return f"Calculation error: {str(e)}"
    
def _get_datetime_info(query_input: Any) -> str:
    """
    Get date/time information. This version is robust against None or empty inputs.
    """
    now = datetime.now()
    query = ""
    if isinstance(query_input, dict):
        query = query_input.get('query')

    if not query:
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    query_lower = query.lower()
    if "date" in query_lower:
        return f"Current date: {now.strftime('%Y-%m-%d')}"
    elif "time" in query_lower:
        return f"Current time: {now.strftime('%H:%M:%S')}"
    else:
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def _analyze_text(text_input: Any, llm) -> str:
    """Analyze text properties"""
    try:
        if isinstance(text_input, dict):
            if 'text' in text_input:
                text = text_input['text']
            elif 'query' in text_input:
                text = text_input['query']
            else:
                text = next(iter(text_input.values())) if text_input else ""
        elif isinstance(text_input, str):
            text = text_input
        else:
            return "Invalid input format for text analyzer. Expected text string or dict with 'text' key."
        
        if not text or not isinstance(text, str):
            return "No valid text provided for analysis."
            
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        if sentence_count == 0:
            sentence_count = 1
        
        analysis = f"""Text Analysis Results:
- Word count: {word_count}
- Character count (with spaces): {char_count}
- Character count (without spaces): {char_count_no_spaces}
- Estimated sentence count: {sentence_count}
- Average words per sentence: {word_count / max(sentence_count, 1):.1f}
- Reading complexity: {'Simple' if word_count / max(sentence_count, 1) < 15 else 'Complex'}

Text analyzed: "{text}" """
        
        return analysis
        
    except Exception as e:
        return f"Text analysis error: {str(e)}"

def _get_react_system_prompt() -> str:
    """Gets the system prompt specifically for the ReAct (step-by-step) agent."""
    return """You are a helpful and knowledgeable assistant. Your goal is to answer the user's questions accurately by using the available tools effectively.

**Example Scenario (Weather):**
User Query: "what's the weather like in london?"
1.  **Thought:** I need to find the weather. I will use the `weather_forecast` tool.
2.  **Action:** Call `weather_forecast` with the query: "London".
3.  **Observation:** The tool returns the weather information.
4.  **Thought:** I have the complete answer now. I will format it with "Final Answer:".
5.  **Final Answer:** The current weather in London is [weather details].

**Tool Selection Strategy (Context Engineering):**
1. **Analyze the query** to understand what type of information is needed.
2. **Choose the most direct tool** for the task:
   - Use `weather_forecast` for weather queries
   - Use `yahoo_finance` for stock prices
   - Use `wikipedia_search` for encyclopedic information
   - Use `tavily_web_search` or `duckduckgo_web_search` for recent news/events
   - Use `contextual_document_search` for internal document queries
   - Use `advanced_calculator` for mathematical problems
   - Use `smart_search` only when you need to combine multiple information sources or when unsure which tool to use.
   This is a crucial step in the 'Context Engineering' pattern to ensure the agent uses the right tool for the job.

3. **Be efficient**: Don't use multiple tools if one can answer the question completely.
4. **Provide context**: When using search results, synthesize the information into a coherent answer.

**Final Answer Format:**
When you have enough information to answer the user's question, start your response with "Final Answer:" followed by your complete answer.
**Example Scenario (Stock Price):**
- Weather query: Use `weather_forecast` directly
- Stock price: Use `yahoo_finance` directly  
- Recent news: Use web search tools directly
- Complex research: May require multiple tools or `smart_search`
"""

def _get_plan_and_execute_system_prompt() -> str:
    """
    Returns the static template for the Plan-and-Execute agent's system prompt.
    This version includes a very strict rule to prevent the LLM from hallucinating
    the final answer within the plan itself.
    """
    return """You are an expert problem solver. Your job is to create a step-by-step plan to answer the user's query using the available tools.

**Available Tools:**
Here is a list of tools you can use. Each tool is described with its name, purpose, and the arguments it accepts.
---
{tools_description}
---

**Instructions & Rules:**
1.  **Analyze the User's Goal:** Deeply understand what the user is asking for.
2.  **Be Efficient:** Your primary goal is to answer the query in the FEWEST steps possible.
3.  **Use Direct Tools:** If a specific tool directly answers the query (e.g., `weather_forecast`), you MUST use that tool and only that tool.
4.  **Stop When Done:** Once a tool provides a complete answer, the plan's next and final step MUST be `final_answer`. Do not add more information-seeking steps.
5.  **Provide Valid Arguments:** You MUST provide all required arguments for a tool as specified in its schema.
6.  **Construct the Final Step:** The final step MUST be the `final_answer` tool. Its `summary` argument MUST ALWAYS be a reference to the output of a previous step, like `"$step1_output"`.
7.  **CRITICAL RULE:** DO NOT invent, hallucinate, or write the final answer in the plan itself. Your only job is to create the plan that FINDS the answer. The final answer will be determined by the tool's output during execution.
8.  **Output JSON Only:** Your entire response must be ONLY the JSON plan.

**JSON Plan Format:**
Your output must be a valid JSON list of dictionaries, where each dictionary represents one step and has the following keys:
- "step": An integer for the step number (e.g., 1, 2, 3).
- "tool": The exact name of the tool to use (e.g., "weather_forecast").
- "args": A dictionary of arguments for the tool (e.g., {{"city": "Mumbai"}}).
- "reasoning": A brief explanation of why this step is necessary.

Now, create the plan for the user's query."""

async def _arun_tool_wrapper(mcp_manager: MCPManager, tool_name: str, **kwargs) -> str:
    """Helper function to execute a tool via MCPManager and return a string result."""
    response: ConnectorResponse = await mcp_manager.execute_tool(tool_name, kwargs)
    return str(response.data) if response.success else response.error

def _create_connector_functions(mcp_manager: MCPManager, tool_name: str, schema_model: Any):
    """
    Factory to create the sync/async tool functions, correctly capturing the tool_name and schema.
    """
    async def _arun_connector(arg: Any = None, **kwargs):
        params = kwargs
        if arg is not None:
            first_arg_name = list(schema_model.__fields__.keys())[0]
            params[first_arg_name] = arg
        
        response: ConnectorResponse = await mcp_manager.execute_tool(tool_name, params)
        return str(response.data) if response.success else response.error

    def _run_connector(arg: Any = None, **kwargs):
        return asyncio.run(_arun_connector(arg, **kwargs))

    return _run_connector, _arun_connector

def create_mcp_tools(mcp_manager: MCPManager) -> List[Tool]:
    """
    Creates LangChain Tool objects that wrap MCP connectors, ensuring they
    support both synchronous and asynchronous invocation.
    """
    tools = []
    
    # Map tool names to their schemas and descriptions
    tool_map = {
        "tavily_web_search": (WebSearchInput, "Performs a web search to find up-to-date information."),
        "duckduckgo_web_search": (WebSearchInput, "Performs a web search to find up-to-date information."),
        "wikipedia_search": (WikipediaInput, "Searches Wikipedia for encyclopedic information."),
        "advanced_calculator": (CalculatorInput, "Performs mathematical calculations."),
        "yahoo_finance": (StockTickerInput, "Fetches the latest stock price for a company's ticker symbol."),
        "weather_forecast": (WeatherInput, "Gets the current weather for a specific city."),
        "news_headlines": (NewsInput, "Finds recent news headlines on a given topic."),
        "arxiv_search": (ArXivInput, "Searches ArXiv for academic papers."),
        "sympy_solver": (SymPySolverInput, "Solves algebraic equations."),
        "web_scraper": (WebScraperInput, "Scrapes a single article from the web on a given topic."),
        "semantic_scholar": (SemanticScholarInput, "Searches Semantic Scholar for academic papers and authors."),
        "pubmed_search": (PubMedInput, "Searches PubMed for medical and life sciences articles."),
        "reddit_search": (RedditSearchInput, "Searches Reddit for posts and comments on a given topic."),
        "youtube_search": (YouTubeSearchInput, "Searches YouTube for videos on a given topic."),
        "github_search": (GitHubSearchInput, "Searches GitHub for repositories and code snippets on a given topic."),
    }
    for connector_name, connector in mcp_manager.connectors.items():
        if connector_name in tool_map:
            schema, description = tool_map[connector_name]

            sync_func, async_func = _create_connector_functions(
                mcp_manager, connector_name, schema
            )

            tools.append(Tool(
                name=connector_name,
                func=sync_func,
                coroutine=async_func,
                description=description,
                args_schema=schema
            ))
    return tools
class UniversalAnswerSynthesizer:
    """
    Universal synthesizer that handles all types of queries and tool responses
    with intelligent detail level detection and adaptive response formatting
    """   
    def __init__(self, llm):
        self.llm = llm
        
        # Main synthesis prompt that handles all query types with DETAIL LEVEL awareness
        self.synthesis_prompt = PromptTemplate(
            input_variables=["query", "tool_data", "query_type", "detail_level"],
            template="""You are an AI assistant providing a clear, accurate answer to the user's query.

Query Type: {query_type}
Detail Level: {detail_level}
User Query: {query}

Raw Information from Tools:
{tool_data}

INSTRUCTIONS:
1. Analyze the user's query to understand exactly what they're asking for
2. **CRITICAL: Respect the detail level requirement:**
   - COMPREHENSIVE: Include ALL relevant details, multiple results, full descriptions
   - MODERATE: Include key details with some supporting information
   - CONCISE: Provide only the most essential information

3. For different query types, adapt your response format:
   - **List/Search queries (GitHub, research papers, news)**: 
     * COMPREHENSIVE: Show ALL retrieved items with full details
     * Format each item clearly with all available metadata
     * Include descriptions, statistics, links, topics, etc.
   
   - **Factual questions**: Direct answer with key details
   - **Calculations**: Show the result clearly
   - **Weather**: Current conditions and forecast
   - **Stock prices**: Current price and brief context
   - **Comparison queries**: Show detailed comparison of all items

4. Do not truncate information when detail_level is COMPREHENSIVE
5. Maintain proper formatting with bullet points, numbered lists, or sections for readability
6. Include URLs, links, and references when available

Do not include phrases like "Based on the information gathered". Just provide the answer naturally.

Answer:"""
        )
        
        # Query classification prompt with DETAIL LEVEL detection
        self.classifier_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Analyze this user query and provide two classifications:

Query: {query}

1. CATEGORY (choose one):
- factual: Questions about facts, people, places, events, definitions
- calculation: Mathematical problems, conversions, computations
- weather: Weather forecasts, conditions, climate information
- financial: Stock prices, market data, economic information
- news: Current events, recent news, breaking news
- research: Academic papers, studies, scientific information
- technical: Programming, software, technical explanations
- creative: Stories, poems, creative writing requests
- comparison: Comparing multiple items, pros/cons
- instruction: How-to questions, step-by-step guidance
- list_search: Finding repositories, papers, multiple items (GitHub, ArXiv, etc.)
- complex: Multi-part questions requiring several tools

2. DETAIL_LEVEL (choose one):
- COMPREHENSIVE: Query asks to "find", "list", "show all", "search for", "get repositories", wants multiple results
- MODERATE: Query asks for explanation, overview, general information
- CONCISE: Query asks simple yes/no, single fact, quick answer

Respond in this exact format:
CATEGORY: [category]
DETAIL_LEVEL: [detail_level]"""
        )
        
        self.synthesis_chain = self.synthesis_prompt | self.llm | StrOutputParser()
        self.classifier_chain = self.classifier_prompt | self.llm | StrOutputParser()
    
    def synthesize_final_answer(self, query: str, tool_responses: List[str], agent_steps: List = None) -> str:
        """
        Synthesize a final answer for any type of query using tool responses
        """
        try:
            # Classify the query type AND detail level
            query_type, detail_level = self._classify_query(query)
            
            logger.info(f"Query classification - Type: {query_type}, Detail: {detail_level}")
            
            # Process and combine tool responses (respecting detail level)
            processed_data = self._process_tool_responses(tool_responses, query_type, detail_level)
            
            # Generate the final answer
            synthesized = self.synthesis_chain.invoke({
                "query": query,
                "tool_data": processed_data,
                "query_type": query_type,
                "detail_level": detail_level
            })
            
            # Post-process for quality (less aggressive for comprehensive queries)
            final_answer = self._post_process_answer(synthesized.strip(), query_type, detail_level)
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Error in universal synthesis: {e}")
            return self._fallback_synthesis(query, tool_responses)
    
    def _classify_query(self, query: str) -> tuple[str, str]:
        """Classify the query to determine the best response format AND detail level"""
        try:
            classification = self.classifier_chain.invoke({"query": query}).strip()
            
            # Parse the response
            category_match = re.search(r'CATEGORY:\s*(\w+)', classification, re.IGNORECASE)
            detail_match = re.search(r'DETAIL_LEVEL:\s*(\w+)', classification, re.IGNORECASE)
            
            category = category_match.group(1).lower() if category_match else None
            detail_level = detail_match.group(1).upper() if detail_match else None
            
            # Validate category
            valid_types = ["factual", "calculation", "weather", "financial", "news", 
                          "research", "technical", "creative", "comparison", "instruction", 
                          "list_search", "complex"]
            
            if category not in valid_types:
                category = self._keyword_classify(query)
            
            # Validate detail level
            if detail_level not in ["COMPREHENSIVE", "MODERATE", "CONCISE"]:
                detail_level = self._detect_detail_level(query)
            
            return category, detail_level
                
        except Exception as e:
            logger.warning(f"Query classification failed: {e}")
            return self._keyword_classify(query), self._detect_detail_level(query)
    def _detect_detail_level(self, query: str) -> str:
        """Detect how much detail the user wants based on query keywords"""
        query_lower = query.lower()
        
        # Comprehensive indicators
        comprehensive_keywords = [
            'find', 'list', 'show all', 'search for', 'get', 'fetch',
            'repositories', 'papers', 'articles', 'results',
            'top', 'best', 'popular', 'latest', 'recent',
            'all', 'multiple', 'several', 'various'
        ]
        
        # Concise indicators
        concise_keywords = [
            'what is', 'who is', 'when', 'where',
            'yes or no', 'true or false', 'is it',
            'quick', 'brief', 'short'
        ]
        
        if any(kw in query_lower for kw in comprehensive_keywords):
            return "COMPREHENSIVE"
        elif any(kw in query_lower for kw in concise_keywords):
            return "CONCISE"
        else:
            return "MODERATE"
    
    def _keyword_classify(self, query: str) -> str:
        """Fallback classification using keywords"""
        query_lower = query.lower()
        
        # Check for list/search queries first (NEW)
        if any(word in query_lower for word in ['find', 'search', 'list', 'show', 'github', 'repository', 'repositories', 'papers']):
            return "list_search"
        elif any(word in query_lower for word in ['calculate', 'compute', 'math', '+', '-', '*', '/', '=']):
            return "calculation"
        elif any(word in query_lower for word in ['weather', 'temperature', 'forecast', 'rain', 'sunny']):
            return "weather"
        elif any(word in query_lower for word in ['stock', 'price', 'market', 'finance', 'trading']):
            return "financial"
        elif any(word in query_lower for word in ['news', 'latest', 'recent', 'headlines', 'breaking']):
            return "news"
        elif any(word in query_lower for word in ['paper', 'research', 'study', 'academic', 'arxiv']):
            return "research"
        elif any(word in query_lower for word in ['how to', 'steps', 'tutorial', 'guide', 'instructions']):
            return "instruction"
        elif any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference', 'better']):
            return "comparison"
        else:
            return "factual"
    
    def _process_tool_responses(self, tool_responses: List[str], query_type: str, detail_level: str) -> str:
        """Process and clean tool responses based on query type AND detail level"""
        if not tool_responses:
            return "No information was retrieved from the tools."
        
        processed_responses = []
        
        # Adjust max length based on detail level
        if detail_level == "COMPREHENSIVE":
            max_length = 8000  # Much higher limit for comprehensive answers
            max_items = 10     # Show more items
        elif detail_level == "MODERATE":
            max_length = 3000
            max_items = 5
        else:  # CONCISE
            max_length = 1000
            max_items = 3
        
        for response in tool_responses:
            if not response or len(response.strip()) < 10:
                continue
                
            # Clean up common tool artifacts
            cleaned = self._clean_tool_response(response)
            
            # Type-specific processing (now respects detail level)
            if query_type == "list_search":
                # NEW: Special handling for list/search results
                processed = self._extract_list_search_info(cleaned, detail_level, max_items)
            elif query_type == "calculation":
                processed = self._extract_calculation_result(cleaned)
            elif query_type == "weather":
                processed = self._extract_weather_info(cleaned)
            elif query_type == "financial":
                processed = self._extract_financial_info(cleaned)
            elif query_type == "news":
                processed = self._extract_news_info(cleaned, max_items)
            elif query_type == "research":
                processed = self._extract_research_info(cleaned, max_items)
            else:
                processed = self._extract_factual_info(cleaned, detail_level)
            
            if processed and len(processed.strip()) > 5:
                processed_responses.append(processed)
        
        # Combine all processed responses
        if not processed_responses:
            return "The tools provided information but it could not be processed properly."
        
        combined = "\n\n".join(processed_responses)
        
        # Only truncate if exceeds max_length
        if len(combined) > max_length:
            combined = combined[:max_length] + "\n\n[Information truncated for clarity]"
        
        return combined
    
    def _clean_tool_response(self, response: str) -> str:
        """General cleaning of tool responses"""
        # Remove common prefixes and metadata
        cleaned = response.replace("Page: ", "").replace("Summary: ", "")
        cleaned = re.sub(r'Source: [^\n]*\n?', '', cleaned)
        cleaned = re.sub(r'Document \d+[^\n]*\n?', '', cleaned)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    def _extract_list_search_info(self, response: str, detail_level: str, max_items: int) -> str:
        """
        NEW: Extract list/search results (GitHub repos, papers, etc.) with minimal truncation
        """
        # For list results, we want to preserve structure and details
        lines = response.split('\n')
        
        if detail_level == "COMPREHENSIVE":
            # Keep everything for comprehensive queries
            return response
        elif detail_level == "MODERATE":
            # Keep first max_items and reasonable detail
            # Try to identify item boundaries (usually numbered or with blank lines)
            items = []
            current_item = []
            item_count = 0
            
            for line in lines:
                # Check if this is a new item (starts with number or after blank line)
                if re.match(r'^\d+\.', line.strip()) and current_item:
                    item_count += 1
                    if item_count <= max_items:
                        items.append('\n'.join(current_item))
                    current_item = [line]
                else:
                    current_item.append(line)
            
            # Add last item
            if current_item and item_count < max_items:
                items.append('\n'.join(current_item))
            
            return '\n\n'.join(items) if items else response[:2000]
        else:  # CONCISE
            # Just first few items with minimal detail
            result_lines = []
            item_count = 0
            for line in lines:
                if re.match(r'^\d+\.', line.strip()):
                    item_count += 1
                    if item_count > max_items:
                        break
                result_lines.append(line)
            
            return '\n'.join(result_lines) if result_lines else response[:500]   
    def _extract_calculation_result(self, response: str) -> str:
        """Extract calculation results"""
        if "Calculation result:" in response:
            return response
        elif "result" in response.lower():
            return response
        else:
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if numbers:
                return f"Result: {numbers[-1]}"
        return response[:200]
    
    def _extract_weather_info(self, response: str) -> str:
        """Extract weather information"""
        if any(word in response.lower() for word in ['temperature', 'weather', 'forecast', 'sunny', 'rainy', 'cloudy']):
            return response[:400]
        return response[:200]
    
    def _extract_financial_info(self, response: str) -> str:
        """Extract financial/stock information"""
        if any(word in response.lower() for word in ['price', 'stock', 'usd', '$', 'market', 'trading']):
            return response[:300]
        return response[:200]
    
    def _extract_news_info(self, response: str, max_items: int = 5) -> str:
        """Extract news headlines and information"""
        lines = response.split('\n')
        important_lines = []
        
        for line in lines[:max_items * 3]:  # Adjusted based on max_items
            if line.strip() and len(line.strip()) > 10:
                important_lines.append(line.strip())
        
        return '\n'.join(important_lines) if important_lines else response[:500]
    
    def _extract_research_info(self, response: str, max_items: int = 5) -> str:
        """Extract research paper information"""
        if any(word in response.lower() for word in ['title:', 'authors:', 'abstract', 'paper']):
            limit = 600 * max_items // 5
            return response[:limit]
        return response[:300]
    
    def _extract_factual_info(self, response: str, detail_level: str = "MODERATE") -> str:
        """Extract factual information for general queries (now respects detail level)"""
        sentences = response.split('. ')
        
        # Adjust number of sentences based on detail level
        if detail_level == "COMPREHENSIVE":
            max_sentences = 10
            max_chars = 2000
        elif detail_level == "MODERATE":
            max_sentences = 5
            max_chars = 800
        else:  # CONCISE
            max_sentences = 3
            max_chars = 400
        
        important_sentences = []
        for sentence in sentences[:max_sentences]:
            if len(sentence.strip()) > 20:
                important_sentences.append(sentence.strip())
        
        result = '. '.join(important_sentences)
        if len(result) > max_chars:
            result = result[:max_chars] + "..."
        
        return result if result else response[:max_chars]
    
    def _post_process_answer(self, answer: str, query_type: str, detail_level: str) -> str:
        """Final post-processing of the answer (less aggressive for comprehensive queries)"""
        # Remove any remaining artifacts
        answer = re.sub(r'\[.*?\]', '', answer)
        
        # For comprehensive answers, preserve newlines for readability
        if detail_level == "COMPREHENSIVE":
            answer = re.sub(r'\n{3,}', '\n\n', answer)
        else:
            answer = re.sub(r'\n+', ' ', answer)
        
        answer = answer.strip()
        
        # Ensure proper sentence ending only if it's not a list format
        if answer and not answer.endswith(('.', '!', '?', ':', '\n')):
            if not re.search(r'[\d\-]$', answer):
                answer += '.'
        
        return answer
    
    def _fallback_synthesis(self, query: str, tool_responses: List[str]) -> str:
        """Simple fallback when advanced synthesis fails"""
        if not tool_responses:
            return "I wasn't able to find information to answer your question. Please try rephrasing it."
        
        for response in tool_responses:
            if response and len(response.strip()) > 20:
                cleaned = self._clean_tool_response(response)
                if len(cleaned) > 100:
                    truncated = cleaned[:200]
                    last_period = truncated.rfind('.')
                    if last_period > 50:
                        return truncated[:last_period + 1]
                    else:
                        return truncated + "..."
                else:
                    return cleaned
        
        return "I found some information but had trouble processing it clearly. Please try asking your question differently."
    
@retry_on_failure(max_retries=3)
def create_enhanced_rag_agent_executor(memory_type: str = "buffer_window", max_iterations: int = 10) -> AgentExecutor:
    try:
        config_manager = ConfigurationManager()
        config_dict = config_manager.configurations()
        model_obj = Model(config_dict)
        llm = model_obj.load_ollama_model()
        embedding = model_obj.load_ollama_embedding()
        logger.info("Models loaded successfully")

        memory = _create_memory(memory_type, llm)
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        mcp_manager = MCPManager(llm=llm, tavily_api_key=tavily_api_key)

        retriever = RetrieverManager(embeddings=embedding, config_dict=config_dict).get_retriever()
        document_search_tool = ContextualRetrievalTool(retriever=retriever, conversation_memory=memory, llm=llm)
        mcp_tools = create_mcp_tools(mcp_manager)
        custom_tools = create_custom_tools(llm, config_dict)
        primary_web_search = mcp_manager.get_primary_web_search_name()
        
        smart_search_tool = SmartSearchTool(
            name="smart_search",
            description="The primary tool for all information-seeking queries. Use it to find answers to questions, get real-time data like stock prices or weather, or search internal documents.",
            document_search_tool=document_search_tool,
            web_search_connector_name=primary_web_search,
            mcp_manager=mcp_manager,
            query_router=create_query_router(llm),
            query_decomposer=create_query_decomposer(llm),
            llm=llm
        )
        
        # Give agent access to all tools, with smart_search as backup
        all_tools = mcp_tools + custom_tools + [document_search_tool, smart_search_tool]


        prompt = ChatPromptTemplate.from_messages([
            ("system", _get_react_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, all_tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent, tools=all_tools, memory=memory, verbose=True, max_iterations=max_iterations,
            max_execution_time=120, early_stopping_method="force",
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        logger.info("Enhanced ReAct RAG agent with simplified toolset created successfully.")
        return agent_executor
    except Exception as e:
        logger.error(f"Failed to create enhanced RAG agent: {e}")
        raise

def create_graph_agent(memory_type: str = "buffer_window", checkpointer=None, interrupt_before=[]) -> Runnable:
    try:
        config_manager = ConfigurationManager()
        config_dict = config_manager.configurations()
        model_obj = Model(config_dict)
        llm = model_obj.load_ollama_model()
        embedding = model_obj.load_ollama_embedding()
        
        memory = _create_memory(memory_type, llm)
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        mcp_manager = MCPManager(llm=llm, tavily_api_key=tavily_api_key)
        
        retriever = RetrieverManager(embeddings=embedding, config_dict=config_dict).get_retriever()
        document_search_tool = ContextualRetrievalTool(retriever=retriever, conversation_memory=memory, llm=llm)
        
        primary_web_search = mcp_manager.get_primary_web_search_name()
        smart_search_tool = SmartSearchTool(
            document_search_tool=document_search_tool,
            web_search_connector_name=primary_web_search,
            mcp_manager=mcp_manager,
            query_router=create_query_router(llm),
            query_decomposer=create_query_decomposer(llm),
            llm=llm
        )

        mcp_tools = create_mcp_tools(mcp_manager)
        custom_tools = create_custom_tools(llm, config_dict)
        
        def final_answer_func(summary: str) -> str:
            """A tool that signals the end of the plan and returns the final answer."""
            return summary

        final_answer_tool = Tool(
            name="final_answer",
            func=final_answer_func,
            description="Provides the final, consolidated answer to the user's query. This must be the last step in any plan.",
            args_schema=FinalAnswerInput
        )

        all_tools = [smart_search_tool] + mcp_tools + custom_tools + [final_answer_tool]
        
        # 1. Build the detailed tool description string
        tool_details = []
        for tool in all_tools:
            schema_json = json.dumps(tool.args_schema.model_json_schema(), indent=2) if tool.args_schema else "{}"
            tool_details.append(
                f"Tool Name: `{tool.name}`\n"
                f"Description: {tool.description}\n"
                f"Arguments Schema:\n```json\n{schema_json}\n```"
            )
        tools_description = "\n---\n".join(tool_details)

        # 2. Get the prompt template from the new function
        system_prompt_template = _get_plan_and_execute_system_prompt()

        # 3. Create the prompt and use .partial() to safely inject the tool descriptions
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_template),
            ("human", "User Query: {input}")
        ]).partial(tools_description=tools_description)

        graph = create_plan_and_execute_graph(llm, all_tools, planner_prompt, checkpointer=checkpointer, interrupt_before=interrupt_before)
        logger.info("Plan-and-Execute graph agent with MCP created successfully.")
        return graph
    except Exception as e:
        logger.error(f"Failed to create graph agent: {e}")
        raise

def _create_memory(memory_type: str, llm) -> Any:
    if memory_type == "summary_buffer":
        # Explicitly set output_key to 'output' to resolve the warning
        return ConversationSummaryBufferMemory(llm=llm, max_token_limit=2000, memory_key="chat_history", return_messages=True, output_key='output')
    return ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True, output_key="output")
# ... (rest of the file)