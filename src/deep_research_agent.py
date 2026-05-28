import json
import re
import asyncio
from typing import TypedDict, List, Annotated, Literal, Optional, Dict, Any
from operator import add
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from datetime import datetime
import logging
import time
from dataclasses import dataclass
from src.model import Model
from src.configuration import ConfigurationManager
from src.chain_of_thought_tracer import ChainOfThoughtTracer, ReasoningStepType
from src.tool_attribution import ToolAttributionTracker
from typing import Optional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ResearchConfig:
    max_iterations: int = 4  # Allow more refinement cycles
    min_quality_score: int = 7
    max_queries_per_iteration: int = 6  # More queries = better coverage
    min_sources_threshold: int = 8  # Ensure diverse sources
    enable_deep_dive: bool = True  # For complex topics

# ===== UTILITY FUNCTIONS =====
def extract_json_from_llm_response(response_text: str) -> Dict[str, Any]:
    """Robustly extract JSON from LLM responses. Always returns a dict."""
    if not isinstance(response_text, str):
        response_text = str(response_text)
    
    def _ensure_dict(parsed):
        """Wrap non-dict results into a dict so callers can always use .get()."""
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            # LLM returned a JSON array — treat items as search_queries
            return {"search_queries": parsed}
        return {"raw_value": parsed}
    
    try:
        return _ensure_dict(json.loads(response_text.strip()))
    except json.JSONDecodeError:
        pass
    
    response_text = re.sub(r'^(Here\'s|Here is|The|This is).*?:', '', response_text, flags=re.IGNORECASE)
    
    json_block_patterns = [
        r'```json\s*([\{\[].*?[\}\]])\s*```',
        r'```\s*([\{\[].*?[\}\]])\s*```',
    ]
    
    for pattern in json_block_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                return _ensure_dict(json.loads(match.group(1)))
            except json.JSONDecodeError:
                continue
    
    try:
        response_text = response_text.strip()
        # Try object first
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = response_text[start:end]
            return _ensure_dict(json.loads(json_str))
    except json.JSONDecodeError:
        pass
    
    try:
        # Try array
        start = response_text.find('[')
        end = response_text.rfind(']') + 1
        if start != -1 and end > start:
            json_str = response_text[start:end]
            return _ensure_dict(json.loads(json_str))
    except json.JSONDecodeError:
        pass
    
    logger.error(f"Could not parse JSON. Response: {response_text[:200]}")
    return {"error": "Could not parse JSON", "raw_response": response_text[:500]}


# ===== STREAMLINED STATE DEFINITION =====
class ResearchState(TypedDict):
    """Simplified state focusing on essential data."""
    query: str
    query_type: Optional[str]
    key_entities: Optional[List[str]]
    search_queries: List[str]
    gathered_info: Annotated[List[dict], add]
    extracted_facts: List[dict]  # Simplified from knowledge graph
    draft_report: str
    final_answer: str
    sources: Annotated[List[dict], add]
    iteration: int
    quality_score: Optional[float]



class OptimizedDeepResearchAgent:
    """Streamlined research agent with explainability support."""
    
    def __init__(
        self, 
        llm: BaseChatModel, 
        search_tools: Dict[str, Any],  # ← CHANGED to dict
        config: Optional[ResearchConfig] = None,
        cot_tracer: Optional[ChainOfThoughtTracer] = None,
        attribution_tracker: Optional[ToolAttributionTracker] = None
    ):
        self.llm = llm
        self.search_tools = search_tools  # ← Store all tools
        self.config = config or ResearchConfig()
        self.max_iterations = self.config.max_iterations
        self.cot_tracer = cot_tracer
        self.attribution_tracker = attribution_tracker
    def _normalize_query(self, q) -> str:
        """Normalize a query item to a plain string (LLM may return dicts)."""
        if isinstance(q, str):
            return q
        if isinstance(q, dict):
            # Try common keys: query, text, search_query, question
            for key in ['query', 'text', 'search_query', 'question', 'title']:
                if key in q:
                    return str(q[key])
            # Fall back to first string value
            for v in q.values():
                if isinstance(v, str) and len(v) > 5:
                    return v
            return str(q)
        return str(q)

    def _dedup_queries(self, queries: list) -> list:
        """Remove duplicate or near-duplicate queries (IMP-5)."""
        seen = set()
        unique = []
        for q in queries:
            q_normalized = q.strip().lower()
            if q_normalized not in seen:
                seen.add(q_normalized)
                unique.append(q)
        return unique

    def _select_best_tool(self, query, state: ResearchState) -> str:
        """Intelligently select tool based on query keywords (IMP-3: expanded categories)."""
        query_lower = self._normalize_query(query).lower()
        
        # Academic queries -> ArXiv
        if any(word in query_lower for word in ['paper', 'study', 'academic', 'journal', 'thesis', 'citation']):
            if "arxiv" in self.search_tools:
                return "arxiv"
        
        # News / sentiment / opinion -> News tool
        if any(word in query_lower for word in ['news', 'latest', 'recent', 'today', 'headlines', 'sentiment', 'opinion', 'public reaction']):
            if "news" in self.search_tools:
                return "news"
        
        # Definitions/overview -> Wikipedia
        if any(word in query_lower for word in ['what is', 'who is', 'define', 'history', 'overview', 'meaning', 'origin']):
            if "wikipedia" in self.search_tools:
                return "wikipedia"
        
        # Internal documents
        if any(word in query_lower for word in ['document', 'internal', 'our', 'company', 'uploaded']):
            if "documents" in self.search_tools:
                return "documents"
        
        # Default to web search
        return "web_search"
    def analyze_and_plan(self, state: ResearchState) -> dict:
        """Combined analysis and query generation with CoT tracking."""
        logger.info("Analyzing query and generating search plan...")
        
        # CoT tracking - Query Analysis
        if self.cot_tracer:
            self.cot_tracer.add_query_analysis(
                query_type="research",
                complexity="multi-faceted",
                key_entities=[state['query']],
                intent="comprehensive_research",
                reasoning=f"Analyzing query '{state['query']}' to generate targeted search queries covering multiple perspectives"
            )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research strategist. Analyze the query and create a focused search plan.

**Query Generation Strategy:**
1. **Core Concept Queries** (2-3): Direct queries about the main topic
2. **Contextual Queries** (2-3): Historical background, origins, evolution
3. **Impact Queries** (2-3): Effects, consequences, implications
4. **Expert/Authority Queries** (1-2): Official sources, academic papers
5. **Comparative Queries** (1-2): How it compares to alternatives
6. **Recent Developments** (1-2): Latest news, updates, trends

CRITICAL: Generate 8-12 diverse queries covering ALL angles above
Return ONLY valid JSON:
{{
  "query_type": "comparison|factual|explanation|how-to",
  "key_entities": ["Entity1", "Entity2"],
  "search_queries": [
    "focused query 1",
    "focused query 2",
    "focused query 3"
  ]
}}"""),
            ("user", "Query: {query}\n\nRespond with JSON only:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": state['query']})
        analysis = extract_json_from_llm_response(response)
        
        # Ensure analysis is always a dict with expected keys
        if not isinstance(analysis, dict) or "error" in analysis:
            logger.warning("Analysis failed or unexpected format, using fallback")
            analysis = {
                "query_type": "factual",
                "key_entities": [state['query']],
                "search_queries": [state['query']]
            }
        
        queries = analysis.get('search_queries', [state['query']])[:self.config.max_queries_per_iteration]
        
        # Normalize all queries to plain strings (LLM may return dicts)
        queries = [self._normalize_query(q) for q in queries]
        # Filter out empty or too-short queries
        queries = [q for q in queries if q and len(q) > 3] or [state['query']]
        # IMP-5: Dedup queries
        queries = self._dedup_queries(queries)
        
        # Normalize key_entities too (BUG-2: LLM may return dicts)
        raw_entities = analysis.get('key_entities', [])
        entities = [self._normalize_query(e) for e in raw_entities] if raw_entities else []
        
        # CoT tracking - Decision on search strategy
        if self.cot_tracer:
            self.cot_tracer.add_decision(
                decision=f"Generated {len(queries)} targeted search queries",
                reasoning=f"Breaking down the research query into {len(queries)} specific searches to ensure comprehensive coverage from multiple angles. Query type identified as '{analysis.get('query_type')}' with key entities: {', '.join(str(e) for e in entities[:3])}",
                confidence=0.85,
                alternatives=[
                    "Single broad search (less comprehensive)",
                    f"More granular queries ({self.config.max_queries_per_iteration * 2} queries, slower)"
                ]
            )
        
        # IMP-4: Progress logging
        current_iter = state.get('iteration', 0) + 1
        logger.info(f"[Step 1/4] Iteration {current_iter}: Query type: {analysis.get('query_type')} | Entities: {len(entities)} | Queries: {len(queries)}")
        
        return {
            "query_type": analysis.get('query_type', 'factual'),
            "key_entities": entities,
            "search_queries": queries,
            "iteration": current_iter  # BUG-4: properly increment
        }

    def execute_searches(self, state: ResearchState) -> dict:
        """Execute searches with INTELLIGENT tool selection and proper URL extraction."""
        # IMP-4: Progress logging
        logger.info(f"[Step 2/4] Executing {len(state['search_queries'])} searches across {len(self.search_tools)} tools...")
        
        async def run_searches():
            tasks = []
            tool_selections = []
            
            for query in state['search_queries']:
                # Normalize query to string
                query_str = self._normalize_query(query)
                # Select best tool for this specific query
                selected_tool = self._select_best_tool(query_str, state)
                tool_selections.append(selected_tool)
                logger.info(f" '{query_str[:50]}...' -> {selected_tool}")
                
                tool = self.search_tools.get(selected_tool, self.search_tools["web_search"])
                tasks.append(asyncio.to_thread(tool.invoke, query_str))
            
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            gathered = []
            sources = []
            successful_searches = 0
            
            for idx, (raw_query, result, tool_name) in enumerate(zip(state['search_queries'], search_results, tool_selections)):
                # BUG-3: Normalize query in source metadata
                query = self._normalize_query(raw_query)
                if isinstance(result, Exception):
                    logger.warning(f"Search failed: {result}")
                    continue
                
                result_str = str(result)
                successful_searches += 1
                
                # ========== FIX: Proper URL extraction from JSON ==========
                urls = []
                try:
                    # Try parsing as JSON first (Tavily returns structured JSON)
                    result_json = json.loads(result_str) if isinstance(result_str, str) else result
                    
                    # Handle Tavily format: {"results": [{"url": "...", "title": "...", "content": "..."}]}
                    if isinstance(result_json, dict) and "results" in result_json:
                        for item in result_json["results"][:3]:
                            if "url" in item:
                                urls.append({
                                    "url": item["url"],
                                    "title": item.get("title", "Untitled"),
                                    "snippet": item.get("content", "")[:200]
                                })
                    # Handle list format
                    elif isinstance(result_json, list):
                        for item in result_json[:3]:
                            if isinstance(item, dict) and "url" in item:
                                urls.append({
                                    "url": item["url"],
                                    "title": item.get("title", "Untitled"),
                                    "snippet": item.get("content", "")[:200]
                                })
                except (json.JSONDecodeError, TypeError):
                    # Fallback: regex for plain text
                    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                    found_urls = re.findall(url_pattern, result_str)
                    urls = [{"url": url, "title": "Source", "snippet": ""} for url in found_urls[:3]]
                
                # If no URLs found, create placeholder
                if not urls:
                    urls = [{"url": f"search_{idx+1}", "title": query[:50], "snippet": result_str[:200]}]
                
                # Add to sources with metadata
                for i, url_info in enumerate(urls):
                    sources.append({
                        "id": f"[{idx+1}.{i+1}]",
                        "url": url_info["url"],
                        "title": url_info.get("title", "Untitled"),
                        "query": query,
                        "tool": tool_name
                    })
                
                gathered.append({
                    "query": query,
                    "results": result_str[:2000],
                    "source_ids": [f"[{idx+1}.{i+1}]" for i in range(len(urls))],
                    "tool_used": tool_name
                })
            
            return gathered, sources, successful_searches

        # BUG-1: Use nest_asyncio-safe approach instead of asyncio.run()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                newly_gathered_info, new_sources, success_count = loop.run_until_complete(run_searches())
            else:
                newly_gathered_info, new_sources, success_count = asyncio.run(run_searches())
        except RuntimeError:
            # Fallback: create new event loop in thread
            new_loop = asyncio.new_event_loop()
            try:
                newly_gathered_info, new_sources, success_count = new_loop.run_until_complete(run_searches())
            finally:
                new_loop.close()
        
        logger.info(f"[Step 2/4] Gathered {len(newly_gathered_info)} results from {len(new_sources)} sources ({success_count} successful)")
        
        return {
            "gathered_info": newly_gathered_info,
            "sources": new_sources
        }

    def extract_relevant_facts(self, state: ResearchState) -> dict:
        """Extract ONLY relevant facts based on the original query with reasoning tracking."""
        # IMP-4: Progress logging
        logger.info(f"[Step 3/4] Extracting relevant facts from {len(state['gathered_info'])} search results...")
        
        # CoT tracking - Starting fact extraction
        if self.cot_tracer:
            self.cot_tracer.add_reasoning_step(
                step_type=ReasoningStepType.RESULT_EVALUATION,
                reasoning=f"Analyzing {len(state['gathered_info'])} search results to extract key facts directly relevant to '{state['query']}'",
                input_data={
                    "num_results": len(state['gathered_info']),
                    "query_type": state.get('query_type', 'unknown'),
                    "key_entities": state.get('key_entities', [])
                },
                output_data={},
                confidence_score=0.8
            )
        
        # Combine all search results
        all_results = "\n\n---\n\n".join([
            f"Query: {item['query']}\n{item['results'][:1500]}"
            for item in state['gathered_info'][:5]  # Max 5 results
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract ONLY the key facts that directly answer the user's query.

Rules:
- Focus on facts relevant to the original query
- Include source IDs for each fact
- Limit to 8-12 most important facts
- For comparisons: extract facts for EACH entity
- Ignore tangential information

Return ONLY valid JSON:
{{
  "facts": [
    {{
      "statement": "Clear, specific fact",
      "entity": "Which entity this is about",
      "source_ids": ["[1.1]"],
      "relevance": "high|medium"
    }}
  ]
}}"""),
            ("user", """Original Query: {query}
Query Type: {query_type}
Key Entities: {entities}

Search Results:
{results}

Extract relevant facts as JSON:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "query": state['query'],
            "query_type": state.get('query_type', 'factual'),
            # BUG-2: Normalize key_entities to strings
            "entities": ", ".join(str(e) if isinstance(e, str) else self._normalize_query(e) for e in (state.get('key_entities') or [])),
            "results": all_results[:8000]  # Hard limit
        })
        
        extracted = extract_json_from_llm_response(response)
        
        if not isinstance(extracted, dict) or "error" in extracted:
            logger.warning("Fact extraction failed or unexpected format, using fallback")
            extracted = {"facts": []}
        
        facts = extracted.get('facts', [])[:12]  # Max 12 facts
        
        # CoT tracking - Fact extraction evaluation
        if self.cot_tracer:
            high_relevance_count = sum(1 for f in facts if f.get('relevance') == 'high')
            self.cot_tracer.add_result_evaluation(
                result=f"Extracted {len(facts)} facts",
                evaluation=f"Successfully identified {high_relevance_count} high-relevance facts and {len(facts) - high_relevance_count} medium-relevance facts",
                quality_score=min(0.9, 0.5 + (len(facts) / 20)),  # Score based on fact count
                reasoning=f"Extracted {len(facts)} relevant facts from {len(state['gathered_info'])} search results. Facts are categorized by relevance and linked to sources for verification."
            )
        
        logger.info(f"Extracted {len(facts)} relevant facts")
        
        return {"extracted_facts": facts}

    def generate_final_report(self, state: ResearchState) -> dict:
        """Generate report with better source formatting."""
        # IMP-4: Progress logging
        logger.info(f"[Step 4/4] Generating final report from {len(state.get('extracted_facts', []))} facts and {len(state.get('sources', []))} sources...")
        
        facts_str = json.dumps(state.get('extracted_facts', []), indent=2)
        sources_str = "\n".join([f"{s['id']} {s['url']}" for s in state.get('sources', [])])
        
        # CoT tracking - Starting synthesis
        if self.cot_tracer:
            self.cot_tracer.add_synthesis_step(
                sources=[s['url'] for s in state.get('sources', [])[:5]],
                synthesis_method="comprehensive_report_generation",
                reasoning=f"Synthesizing {len(state.get('extracted_facts', []))} facts from {len(state.get('sources', []))} sources into a coherent, well-structured report addressing '{state['query']}'",
                output="Generating structured research report with citations"
            )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research writer. Create a clear, well-structured report.

**Formatting Rules:**

For COMPARISONS (multiple entities):
1. Brief Introduction (4-5 sentences)
2. Comparison sections organized by aspects/dimensions:
   - ## Aspect 1
     - ### Entity A: [facts with citations]
     - ### Entity B: [facts with citations]
3. Conclusion (4-5 paragraphs)

For OTHER queries:
1. Clear introduction
2. Main findings (use ## headers for major sections)
3. Supporting details with bullet points
4. Brief conclusion

**Content Rules:**
- Use inline citations: "Fact here [1.1]"
- Bold key terms
- Be concise and direct
- If information is missing, state it clearly
- DO NOT add a sources section (will be added automatically)

Write the complete report now in Markdown:"""),
            ("user", """Query: {query}
Query Type: {query_type}

Extracted Facts:
{facts}

Available Sources:
{sources}

Generate the report:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        draft = chain.invoke({
            "query": state['query'],
            "query_type": state.get('query_type', 'factual'),
            "facts": facts_str,
            "sources": sources_str
        })
        
        # Clean up formatting
        draft = draft.replace('\\n', '\n')
        draft = re.sub(r'\n{3,}', '\n\n', draft)
        
        # ========== BETTER SOURCE FORMATTING ==========
        sources_md = "\n\n---\n\n## Sources\n\n"
        sources_by_tool = {}
        
        for src in state.get('sources', []):
            tool = src.get('tool', 'web_search')
            if tool not in sources_by_tool:
                sources_by_tool[tool] = []
            sources_by_tool[tool].append(src)
        
        for tool, tool_sources in sources_by_tool.items():
            sources_md += f"### {tool.replace('_', ' ').title()}\n"
            for src in tool_sources:
                title = src.get('title', 'Untitled')
                url = src.get('url', '')
                source_id = src.get('id', '')
                
                # Skip API endpoints
                if 'api.tavily.com' in url or url.startswith('search_'):
                    sources_md += f"{source_id} **{title}** (Retrieved via {tool})\n"
                else:
                    sources_md += f"{source_id} [{title}]({url})\n"
            sources_md += "\n"
        
        final_report = f"{draft.strip()}{sources_md.strip()}"
        return {"final_answer": final_report, "quality_score": 8.0}

    def should_continue(self, state: ResearchState) -> Literal["analyze_and_plan", "generate_final_report"]:
        """Simplified decision logic with reasoning tracking."""
        iteration = state['iteration']
        quality_score = state.get('quality_score', 0)
        
        # Always finalize after max iterations
        if iteration >= self.max_iterations:
            if self.cot_tracer:
                self.cot_tracer.add_decision(
                    decision="Finalize report - max iterations reached",
                    reasoning=f"Reached maximum iteration limit ({self.max_iterations}). Proceeding to generate final report with gathered information.",
                    confidence=1.0,
                    alternatives=[]
                )
            logger.info("Max iterations reached, finalizing")
            return "generate_final_report"
        
        # If we have good facts, finalize
        facts_count = len(state.get('extracted_facts', []))
        if facts_count >= 6:  # Threshold for "enough information"
            if self.cot_tracer:
                self.cot_tracer.add_decision(
                    decision="Finalize report - sufficient information gathered",
                    reasoning=f"Gathered {facts_count} facts which exceeds the threshold of 6. This provides sufficient information to generate a comprehensive report.",
                    confidence=0.85,
                    alternatives=[f"Continue for {self.max_iterations - iteration} more iterations"]
                )
            logger.info(f"Sufficient facts gathered ({facts_count}), finalizing")
            return "generate_final_report"
        
        # Otherwise, do one more iteration
        if self.cot_tracer:
            self.cot_tracer.add_decision(
                decision="Continue research - gather more information",
                reasoning=f"Only {facts_count} facts gathered (threshold: 6). Continuing for iteration {iteration + 1} to gather more comprehensive information.",
                confidence=0.75,
                alternatives=["Finalize with current information"]
            )
        logger.info("Gathering additional information")
        return "analyze_and_plan"

# ===== OPTIMIZED GRAPH CONSTRUCTION =====
def build_optimized_research_graph(
    llm: BaseChatModel, 
    search_tools: Dict[str, Any],  # ← CHANGED
    config: Optional[ResearchConfig] = None,
    cot_tracer: Optional[ChainOfThoughtTracer] = None,
    attribution_tracker: Optional[ToolAttributionTracker] = None
):
    agent = OptimizedDeepResearchAgent(llm, search_tools, config, cot_tracer, attribution_tracker)
    
    # ... rest of graph construction remains the same ...
    
    graph = StateGraph(ResearchState)
    
    # Only 4 nodes instead of 9!
    graph.add_node("analyze_and_plan", agent.analyze_and_plan)
    graph.add_node("execute_searches", agent.execute_searches)
    graph.add_node("extract_facts", agent.extract_relevant_facts)
    graph.add_node("generate_report", agent.generate_final_report)
    
    # Linear flow with one conditional
    graph.set_entry_point("analyze_and_plan")
    graph.add_edge("analyze_and_plan", "execute_searches")
    graph.add_edge("execute_searches", "extract_facts")
    
    graph.add_conditional_edges(
        "extract_facts",
        agent.should_continue,
        {
            "analyze_and_plan": "analyze_and_plan",
            "generate_final_report": "generate_report"
        }
    )
    
    graph.add_edge("generate_report", END)
    
    return graph.compile()


# ===== MAIN INTERFACE =====
def research(
    query: str, 
    search_tools: Dict[str, Any],  # ← CHANGED
    config: Optional[ResearchConfig] = None,
    cot_tracer: Optional[ChainOfThoughtTracer] = None,
    attribution_tracker: Optional[ToolAttributionTracker] = None
) -> dict:
    """Optimized research interface with explainability and MULTIPLE tools."""
    
    # Define llm FIRST
    config_manager = ConfigurationManager()
    config_dict = config_manager.configurations()
    model_obj = Model(config_dict)
    llm = model_obj.load_ollama_model()

    logger.info(f"Starting optimized research for: '{query}'")
    logger.info(f"Available tools: {list(search_tools.keys())}")  # ← Added this
    start_time = datetime.now()
    
    # NOW we can use llm
    graph = build_optimized_research_graph(
        llm, 
        search_tools, 
        config,
        cot_tracer=cot_tracer,
        attribution_tracker=attribution_tracker
    )
    
    initial_state = {
        "query": query,
        "query_type": None,
        "key_entities": None,
        "search_queries": [],
        "gathered_info": [],
        "extracted_facts": [],
        "draft_report": "",
        "final_answer": "",
        "sources": [],
        "iteration": 0,
        "quality_score": None
    }
    
    final_state = graph.invoke(initial_state)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Research complete in {processing_time:.2f}s")
    
    # IMP-6: Include sources and facts in metadata for richer API responses
    all_sources = final_state.get('sources', [])
    all_facts = final_state.get('extracted_facts', [])
    
    return {
        "final_answer": final_state.get('final_answer', "Unable to generate answer."),
        "agent_steps": [],
        "agent_type": "optimized_deep_research",
        "metadata": {
            "query_timestamp": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "iterations_used": final_state.get('iteration', 0),
            "query_type": final_state.get('query_type', 'unknown'),
            "sources_count": len(all_sources),
            "facts_extracted": len(all_facts),
            "sources": all_sources[:20],  # Top 20 sources
            "key_facts": [f.get('statement', str(f)) for f in all_facts[:12]]
        }
    }