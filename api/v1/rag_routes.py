from fastapi import APIRouter, HTTPException, Query, UploadFile, File, BackgroundTasks, Body, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, AnyUrl
from typing import Optional, List, Dict, Any, Literal, Union
from enum import Enum
import shutil, os, time, datetime, json, uuid, re, logging, asyncio

from src.configuration import ConfigurationManager
from src.rag import main
from src.data_ingestion import DataIngestion
from src.agent import create_enhanced_rag_agent_executor, create_graph_agent , UniversalAnswerSynthesizer,execute_deep_research
from src.model import Model
import pandas as pd
from pathlib import Path
from src.retriever import RetrieverManager
from src.crag_pipeline import CRAGPipeline 
from src.evaluation import RetrievalEvaluator
from src.test_data_generation import TestDataGeneration
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult
from src.chain_of_thought_tracer import ChainOfThoughtTracer, ReasoningStepType
from src.tool_attribution import ToolAttributionTracker
from src.explainability_integration import ExplainabilityCallback
from src.prompt_safety_guard import create_safety_guard, ThreatLevel

from src.document_generator import (
    EnhancedDocumentGenerator,
    GenerationConfig,
    GeneratedDocument,
    DocumentType,
    RetrievalStrategy,
    QualityMetrics
)
from src.advanced_mindmap import MindMapResponse, create_mindmap_from_text
from src.rag import RAG
# NEW: Import enhanced RAG and mindmap RAG integration
from src.enhanced_rag import (
    EnhancedRAG,
    RAGConfig,
    EnhancedRAGResponse,
    CitationInfo as RAGCitationInfo,
    QualityMetrics as RAGQualityMetrics
)
from src.mindmap_rag_integration import (
    RAGMindMapGenerator,
    RAGMindMapResponse,
    MindMapQualityMetrics,
    generate_mindmap_from_rag
)
from src.reasoning_agent.core import ReasoningAgentSystem
from src.configuration import AgentConfig
import asyncio
from src.gemini_url_connector import GeminiURLAnalyzer
from pydantic import AnyUrl, Field
# --- Import standard logging ---
import logging

# --- Set up logger ---
logger = logging.getLogger(__name__)
# # Storage for generation history
rag_router = APIRouter(prefix="/api/rag", tags=["Document Generation"])
reasoning_router = APIRouter(prefix="/api/reasoning", tags=["Reasoning Agent"])
# Singleton for reasoning agent
_reasoning_agent: Optional[ReasoningAgentSystem] = None

def get_reasoning_agent() -> ReasoningAgentSystem:
    global _reasoning_agent
    if _reasoning_agent is None:
        config = AgentConfig()
        _reasoning_agent = ReasoningAgentSystem(config)
    return _reasoning_agent
generation_history = {}
batch_jobs = {}
# Constants that were missing
RESPONSE_LOG_DIR = "api_responses"
SECURITY_LOG_DIR = "security_logs"



class SourceGuide(BaseModel):
    """Pydantic model for a single file's source guide."""
    file_name: str
    source_guide: Optional[str] = None
    error: Optional[str] = None

class SourceGuideResponse(BaseModel):
    """Response model for the source guides endpoint."""
    domain: str
    source_guides: List[SourceGuide]
class DeepResearchResponse(BaseModel):
    final_answer: str
    agent_type: str = "enhanced_deep_research"
    metadata: Dict[str, Any] = {}
    follow_up_questions: List[str] = []
    knowledge_gaps: List[str] = []
    sources_count: int = 0
    iterations_used: int = 0
    epistemic_metrics: Optional[Dict[str, Any]] = None
    safety_check: Optional[Dict[str, Any]] = None
# Add these Pydantic models after your existing models
class DomainSelectionRequest(BaseModel):
    """Request model for domain selection"""
    domain_name: str

class AvailableDomainsResponse(BaseModel):
    """Response showing available domains"""
    available_domains: List[str]
    active_domain: Optional[str]
    initialized_domains: List[str]
    message: str

class DocumentContextWithMindMap(BaseModel):
    """Enhanced context with optional mind map"""
    source: str
    page: int
    content_snippet: str
    rerank_score: Optional[float] = None
    doc_type: Optional[str] = None

class DocumentQueryResponseWithMindMap(BaseModel):
    """Response model with optional mind map."""
    answer: str
    retrieved_context: List[DocumentContextWithMindMap]
    metadata: Dict[str, Any]
    mindmap: Optional[MindMapResponse] = None  # Optional mind map

class GeneratedDocumentWithMindMap(BaseModel):
    """Extended GeneratedDocument with optional mind map"""
    # All fields from GeneratedDocument
    content: str
    document_type: str
    quality_metrics: Optional[Dict[str, Any]] = None
    sources_used: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    mindmap: Optional[MindMapResponse] = None  # Add mind map
# --- 1. Define Structured Response Models (Your models are great!) ---
class AgentStep(BaseModel):
    step_number: int
    action: str
    action_input: Dict[str, Any]
    observation: str
    thought: str = ""


class EpistemicLoopMetric(BaseModel):
    """Metrics for a single Active Inference loop"""
    loop: int
    action: str
    action_type: str = Field(description="epistemic (info-seeking) or pragmatic (goal-achieving)")
    efe_score: float = Field(description="Expected Free Energy — lower = better for pragmatic, higher = more info gain")
    confidence: float = Field(description="Agent confidence (0-1)")
    entropy: float = Field(description="Shannon entropy H — how uncertain the belief state is")
    vfe: float = Field(description="Variational Free Energy — model surprise, lower = better fit")
    surprisal: float = Field(description="Prediction error — high = potential hallucination")
    is_hallucination: bool = Field(description="Whether surprisal exceeded threshold")
    info_gain: float = Field(description="Epistemic value — expected information gain from this action")
    pragmatic_value: float = Field(description="Pragmatic value — expected goal progress from this action")
    beliefs: Dict[str, Dict[str, float]] = Field(description="Bayesian belief distributions after this loop")
    concentration: Dict[str, float] = Field(description="Dirichlet concentration (evidence strength) per factor")

class EpistemicDashboard(BaseModel):
    """Clear epistemic agent performance metrics"""
    # ── Final State ──
    final_confidence: float = Field(0.0, description="Final agent confidence (0-1)")
    final_entropy: float = Field(0.0, description="Final entropy H — lower = more certain")
    final_vfe: float = Field(0.0, description="Final Variational Free Energy")
    total_loops: int = Field(0, description="Number of Active Inference loops executed")
    converged: bool = Field(False, description="Whether the agent converged before max iterations")
    convergence_reason: str = Field("", description="Why the agent stopped: confidence_stable, answer_synthesized, pragmatic_action_succeeded, max_iterations_reached")
    
    # ── Final Beliefs ──
    final_beliefs: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Final Bayesian belief distributions: file_status, user_intent, risk_level"
    )
    
    # ── Per-Loop Trace ──
    loop_trace: List[EpistemicLoopMetric] = Field(
        default_factory=list,
        description="Per-loop metrics trace showing how EFE, confidence, entropy evolved"
    )
    
    # ── Security Metrics ──
    security: Dict[str, Any] = Field(
        default_factory=dict,
        description="Security module metrics: injection_score, injection_blocked, exfiltration_blocked, escalation_detected, belief_drift, cross_validation_agreement"
    )

class SafetyCheckInfo(BaseModel):
    """Safety check information to include in responses"""
    is_safe: bool
    threat_level: str
    violation_type: str
    confidence_score: float
    explanation: str
    recommendations: List[str] = []
from langgraph.checkpoint.memory import MemorySaver

class ChatQueryRequest(BaseModel):
    query: str
    use_crag: bool = False
    agent_type: str = "react"
    include_steps: bool = True
    enable_hitl: bool = False  # Enable Human-in-the-Loop

class ChatResponse(BaseModel):
    final_answer: str
    agent_steps: List[AgentStep] = []
    agent_type: str
    metadata: Dict[str, Any] = {}
    epistemic_metrics: Optional[EpistemicDashboard] = None
    safety_check: SafetyCheckInfo
class ExplainabilityResponse(BaseModel):
    final_answer: str
    agent_steps: List[AgentStep] = []
    metadata: Dict[str, Any] = {}
    explainability_results: Dict[str, Any] = {}

# --- NEW: Pydantic models for the /doc_question endpoint ---
class DocumentContext(BaseModel):
    """Pydantic model for a single retrieved document context."""
    source: str
    page: int
    content_snippet: str
    rerank_score: Optional[float] = None  # ADD THIS LINE
    doc_type: Optional[str] = None        # ADD THIS LINE

class DocumentQueryResponse(BaseModel):
    """Response model for the document query endpoint."""
    answer: str
    retrieved_context: List[DocumentContext]
    metadata: Dict[str, Any]

# Add these new Pydantic models after your existing models

class DocumentSourceGuide(BaseModel):
    """Source guide for a single document"""
    file_name: str
    source_guide: str

class DomainSourceGuidesResponse(BaseModel):
    """Response containing source guides for all documents in a domain"""
    domain_name: str
    collection_name: str
    total_documents: int
    documents: List[DocumentSourceGuide]
    generation_timestamp: str
    processing_time_seconds: float

# Add these Pydantic models
class UserProfileResponse(BaseModel):
    """Enhanced user profile with memory previews"""
    preferences: List[Dict[str, Any]] = []
    knowledge: List[Dict[str, Any]] = []
    context: List[Dict[str, Any]] = []
    goals: List[Dict[str, Any]] = []
    statistics: Dict[str, int] = {}
    
    # New preview fields
    trajectories_preview: List[Dict[str, Any]] = Field(default_factory=list, description="Sample successful reasoning patterns")
    session_previews: List[Dict[str, Any]] = Field(default_factory=list, description="Recent conversation summaries")

class MemoryRetrievalRequest(BaseModel):
    """Request to retrieve memories"""
    query: str
    memory_types: Optional[List[str]] = None
    top_k: int = Field(5, ge=1, le=20)
    min_relevance: float = Field(0.7, ge=0.0, le=1.0)

class MemoryRetrievalResponse(BaseModel):
    """Response with retrieved memories"""
    query: str
    memories_found: int
    memories: List[Dict[str, Any]]
class URLAnalysisRequest(BaseModel):
    """Request model for analyzing URLs via Gemini"""
    query: Optional[str] = Field(
        None, 
        description="Query containing one or more URLs (auto-detected)"
    )
    url: Optional[str] = Field(
        None, 
        description="Single URL to analyze"
    )
    urls: Optional[List[str]] = Field(
        None, 
        description="Multiple URLs to compare"
    )
    instruction: Optional[str] = Field(
        None,
        description="Custom instruction for Gemini (e.g., 'Extract key findings', 'Compare pricing', etc.)"
    )

class URLAnalysisResponse(BaseModel):
    """Response from Gemini URL analysis"""
    success: bool
    content: Optional[str] = None
    urls_analyzed: List[str] = Field(default_factory=list)
    grounding_used: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
class GeminiURLConnectorFixed:
    """
    Fixed version of GeminiURLConnector that properly integrates with FastAPI.
    Uses Gemini 2.0 Flash with google_search grounding for URL analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.name = "gemini_url_analyzer"
        self.timeout = 30.0
        
        try:
            self.analyzer = GeminiURLAnalyzer(api_key)
            logger.info("[GEMINI] Initialized GeminiURLConnectorFixed successfully")
        except Exception as e:
            logger.error(f"[GEMINI] Failed to initialize: {e}")
            raise
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute URL analysis using Gemini.
        
        Args:
            params: Dictionary with 'query', 'url', 'urls', or 'instruction'
        
        Returns:
            Dictionary with success, content, urls_analyzed, error, metadata
        """
        try:
            logger.info(f"[GEMINI] Executing with params: {list(params.keys())}")
            
            # Case 1: Direct URL analysis
            if 'url' in params and params['url']:
                url = params['url'].strip()
                if not url or url == "string":
                    return {
                        'success': False,
                        'error': 'Invalid URL provided'
                    }
                
                instruction = params.get('instruction', 'Analyze this URL and extract key information')
                logger.info(f"[GEMINI] Single URL mode: {url}")
                result = await self.analyzer.analyze_url(url, instruction)
            
            # Case 2: Multiple URL comparison
            elif 'urls' in params and params['urls']:
                urls = [u.strip() for u in params['urls'] if isinstance(u, str) and u.strip() and u != "string"]
                if not urls:
                    return {
                        'success': False,
                        'error': 'No valid URLs provided in list'
                    }
                
                instruction = params.get('instruction', f'Compare these {len(urls)} URLs side by side')
                logger.info(f"[GEMINI] Multi-URL mode: {len(urls)} URLs")
                result = await self.analyzer.compare_urls(urls, instruction)
            
            # Case 3: Query with embedded URLs (natural language)
            elif 'query' in params and params['query']:
                query = params['query'].strip()
                logger.info(f"[GEMINI] Query mode: {query[:100]}...")
                result = await self.analyzer.analyze_query_with_urls(query)
            
            else:
                logger.warning("[GEMINI] No valid input provided")
                return {
                    'success': False,
                    'error': 'Must provide either query, url, or urls parameter'
                }
            
            # Convert GeminiURLResponse to dict
            if result.success:
                logger.info(f"[GEMINI] ✓ Success - {len(result.urls_analyzed)} URLs analyzed")
                return {
                    'success': True,
                    'content': result.content,
                    'urls_analyzed': result.urls_analyzed or [],
                    'grounding_used': result.grounding_metadata is not None,
                    'metadata': {
                        'model': 'gemini-2.5-flash-exp',
                        'grounding_metadata': result.grounding_metadata
                    }
                }
            else:
                logger.error(f"[GEMINI] ✗ Failed: {result.error}")
                return {
                    'success': False,
                    'error': result.error or 'Unknown error occurred',
                    'urls_analyzed': result.urls_analyzed or []
                }
        
        except Exception as e:
            logger.error(f"[GEMINI] Exception in execute: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Gemini execution error: {str(e)}'
            }
# --- 2. Create the Custom Callback Handler for ReAct ---
class UniversalAgentStepCapture(BaseCallbackHandler):
    """Universal callback handler for all query and tool types"""
    
    def __init__(self, llm, query: str):
        self.llm = llm
        self.query = query
        self.steps = []
        self.current_step = 1
        self.final_answer = None
        self.synthesizer = UniversalAnswerSynthesizer(llm)
        self.tool_outputs = []

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Capture the agent's action and thoughts."""
        self.steps.append(
            AgentStep(
                step_number=self.current_step,
                action=action.tool,
                action_input=action.tool_input,
                observation="",
                thought=action.log.strip() if action.log else f"Using {action.tool} tool"
            )
        )

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Capture the tool's output and store for synthesis."""
        output_str = str(output)
        
        # Store truncated version for display
        if self.steps:
            display_output = output_str[:300] + "..." if len(output_str) > 300 else output_str
            self.steps[-1].observation = display_output
        
        # Store full output for synthesis
        self.tool_outputs.append(output_str)
        self.current_step += 1
        
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Synthesize final answer using the universal synthesizer."""
        # Use the synthesizer to create a proper final answer
        self.final_answer = self.synthesizer.synthesize_final_answer(
            self.query, 
            self.tool_outputs,
            self.steps
        )
                            
from src.app_state import app_state

# Helper to ensure initialization
def ensure_initialized():
    """Raises HTTPException if app is not initialized"""
    if not app_state.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="Application is still initializing. Please try again in a moment."
        )
def log_security_incident(query: str, safety_result: Any, endpoint: str):
    """Log security incidents for monitoring and auditing"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SECURITY_LOG_DIR, f"security_incident_{timestamp}.json")
    
    incident_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "endpoint": endpoint,
        "query": query,
        "threat_level": safety_result.threat_level.value,
        "violation_type": safety_result.violation_type.value,
        "confidence_score": safety_result.confidence_score,
        "explanation": safety_result.explanation,
        "details": safety_result.details,
        "recommendations": safety_result.recommendations
    }
    
    # FIX: Add the missing file write
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(incident_data, f, indent=4)
    
    print(f"Security incident logged: {filename}")

# Request/Response Models
class DocumentGenerationRequest(BaseModel):
    """Request model for document generation."""
    topic: str = Field(..., description="Main topic or question")
    document_type: DocumentType
    persona: Optional[str] = Field("expert assistant", description="Persona for generation")
    tone: Optional[str] = Field("professional", description="Tone of the document")
    length: Optional[Literal["short", "medium", "long"]] = "medium"
    target_audience: Optional[str] = Field("general", description="Target audience")
    language_level: Optional[Literal["beginner", "intermediate", "advanced"]] = "intermediate"
    num_sources: int = Field(5, ge=1, le=20, description="Number of sources to retrieve")
    include_citations: bool = Field(True, description="Include citation markers")
    include_visual_suggestions: bool = Field(False, description="Suggest visual elements")
    adaptive_retrieval: bool = Field(True, description="Use adaptive retrieval")
    multi_stage_generation: bool = Field(False, description="Use multi-stage refinement")
    max_iterations: int = Field(1, ge=1, le=5, description="Max refinement iterations")
    additional_instructions: Optional[str] = None


class DocumentRefinementRequest(BaseModel):
    """Request model for document refinement."""
    generation_id: str = Field(..., description="ID of document to refine")
    feedback: str = Field(..., description="User feedback for refinement")
    regenerate_sections: Optional[List[str]] = Field(None, description="Specific sections to regenerate")


class BatchGenerationRequest(BaseModel):
    """Request model for batch generation."""
    requests: List[DocumentGenerationRequest] = Field(..., max_items=10)
    shared_retrieval: bool = Field(True, description="Share retrieval across similar topics")
    parallel: bool = Field(False, description="Process in parallel (async)")


class ExportRequest(BaseModel):
    """Request model for document export."""
    generation_id: str
    format: Literal["markdown", "html", "json"] = "markdown"


class GenerationHistoryItem(BaseModel):
    """History item for generated documents."""
    generation_id: str
    timestamp: str
    request: DocumentGenerationRequest
    response: GeneratedDocument


class BatchJobStatus(BaseModel):
    """Status of a batch generation job."""
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    total: int
    completed: int
    results: List[GeneratedDocument] = Field(default_factory=list)
    error: Optional[str] = None
class SubjectDomain(str, Enum):
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    ECONOMICS = "economics"
    PSYCHOLOGY = "psychology"
    STATISTICS = "statistics"
    SOCIAL_SCIENCE = "social_science"
    GENERAL = "general"


class ReasoningStep(BaseModel):
    step_num: int
    thought: str
    action: str
    observation: str

class ReasoningResponse(BaseModel):
    success: bool
    query: str
    subject: str
    problem_type: str
    subtype: Optional[str] = None
    answer: str
    steps: List[ReasoningStep] = []
    metadata: Dict[str, Any] = {}
def generate_mindmap_from_content(
    content: str,
    topic: str,
    max_nodes: int = 50,
    depth: str = "balanced"
) -> Optional[MindMapResponse]:
    """
    Generate a mind map from content using the app's mind map generator.
    
    Args:
        content: The text content to create mind map from
        topic: The central topic
        max_nodes: Maximum nodes in the map
        depth: Depth preference ("shallow", "balanced", "deep")
    
    Returns:
        MindMapResponse or None if generation fails
    """
    try:
        # Use the global mind map generator
        mindmap = app_state.mindmap_generator.generate_from_text(
            text_content=content,
            topic=topic,
            depth_preference=depth
        )
        logger.info(f"Mind map generated: {mindmap.statistics.total_nodes} nodes, {mindmap.statistics.total_edges} edges")
        return mindmap
    except Exception as e:
        logger.error(f"Failed to generate mind map: {e}", exc_info=True)
        return None


# In-memory storage for generation history (use DB in production)
generation_history: Dict[str, GenerationHistoryItem] = {}

@rag_router.get("/domains", response_model=AvailableDomainsResponse, 
    summary="Check Available Databases")
def get_available_domains():
    """
    **STEP 1: Check available domain databases**
    
    - Lists all domain folders found in dataset/
    - Shows which domain is currently active
    - Shows which domains have been initialized (have vector stores)
    
    **Use this first to see available domains before data ingestion.**
    """
    ensure_initialized()
    
    try:
        config_manager = ConfigurationManager()
        available = config_manager.get_available_domains()
        initialized = list(app_state.retrievers.keys())
        
        if not available:
            message = "No domain folders found in dataset/. Create folders like: dataset/healthcare/, dataset/finance/"
        elif not initialized:
            message = f"Found {len(available)} domains but none initialized. Use /generate-embedding to initialize a domain."
        else:
            message = f"Found {len(available)} domains, {len(initialized)} initialized. Active: {app_state.active_domain or 'None'}"
        
        return AvailableDomainsResponse(
            available_domains=available,
            active_domain=app_state.active_domain,
            initialized_domains=initialized,
            message=message
        )
    except Exception as e:
        logger.error(f"Error getting domains: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/select-domain", summary="Select/Switch Active Database")
def select_domain(request: DomainSelectionRequest):
    """
    **STEP 2: Select which domain database to use**
    
    - Switch to a different domain database
    - If domain not initialized, you'll be prompted to run /generate-embedding first
    - All subsequent queries will use this domain until switched
    
    **Example:** `{"domain_name": "healthcare"}`
    """
    ensure_initialized()
    
    config_manager = ConfigurationManager()
    available = config_manager.get_available_domains()
    
    if request.domain_name not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{request.domain_name}' not found. Available domains: {available}"
        )
    
    # Check if domain is initialized
    if request.domain_name not in app_state.retrievers:
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Domain '{request.domain_name}' exists but not initialized.",
                "action_required": "Please run /generate-embedding with this domain_name to initialize it.",
                "domain_name": request.domain_name,
                "status": "not_initialized"
            }
        )
    
    try:
        # Switch to the domain
        app_state.set_active_domain(request.domain_name)
        app_state.dataingestion_obj = app_state.dataingestion_objs[request.domain_name]
        
        # Update RAG chain with new retriever
        rag_pipeline = RAG(model=app_state.llm_model, retriever=app_state.retriever)
        app_state.chain_obj = rag_pipeline.chain
        
        # Update CRAG pipeline
        domain_config = config_manager.get_domain_config(request.domain_name)
        app_state.crag_pipeline = CRAGPipeline(
            llm=app_state.llm_model,
            retriever=app_state.retriever,
            config=domain_config
        )
        
        # Update document generator
        app_state.doc_generator = EnhancedDocumentGenerator(
            llm=app_state.llm_model,
            retriever=app_state.retriever,
            embedding_model=app_state.embedding
        )
        
        logger.info(f"Switched to domain: {request.domain_name}")
        
        return {
            "message": f"Successfully switched to domain: {request.domain_name}",
            "active_domain": app_state.active_domain,
            "data_path": domain_config['data_path'],
            "collection_name": domain_config['collection_name'],
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error switching domain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@rag_router.post("/generate-embedding", summary="Initialize Database for Domain")
def generate_embedding(domain_name: str = Query(..., description="Domain name to initialize (folder name in dataset/)")):
    """
    **STEP 3: Generate embeddings and create vector database for a domain**
    
    - Processes all PDFs in dataset/{domain_name}/
    - Creates vector store in database/chromadb/{domain_name}/
    - Automatically sets this domain as active
    
    **Example:** `/generate-embedding?domain_name=healthcare`
    
    **Before running:**
    1. Ensure dataset/{domain_name}/ folder exists with PDF files
    2. Run /domains to see available domains
    """
    ensure_initialized()
    
    try:
        # Check if domain exists
        config_manager = ConfigurationManager()
        available_domains = config_manager.get_available_domains()
        
        if domain_name not in available_domains:
            raise HTTPException(
                status_code=404,
                detail=f"Domain folder '{domain_name}' not found in dataset/. Available: {available_domains}"
            )
        
        logger.info(f"Starting data ingestion for domain: {domain_name}")
        
        # Get domain-specific configuration
        domain_config = config_manager.get_domain_config(domain_name)
        
        # Create or get data ingestion object for this domain
        if domain_name not in app_state.dataingestion_objs:
            app_state.dataingestion_objs[domain_name] = DataIngestion(
                domain_config,
                embedding_model=app_state.embedding
            )
        
        ingestion_obj = app_state.dataingestion_objs[domain_name]
        
        # Run the ingestion pipeline
        vectorstore = ingestion_obj.ingestion_main()
        
        if vectorstore:
            # Create retriever for this domain
            retriever_manager = RetrieverManager(
                embeddings=app_state.embedding,
                config_dict=domain_config
            )
            
            app_state.retrievers[domain_name] = retriever_manager.get_retriever(
                k=15,
                enable_reranking=True,
                top_n_rerank=8
            )
            
            app_state.vectorstores[domain_name] = vectorstore
            
            # Automatically set as active domain
            app_state.set_active_domain(domain_name)
            app_state.dataingestion_obj = ingestion_obj
            
            # Update RAG chain
            rag_pipeline = RAG(model=app_state.llm_model, retriever=app_state.retriever)
            app_state.chain_obj = rag_pipeline.chain
            
            # Update CRAG pipeline
            app_state.crag_pipeline = CRAGPipeline(
                llm=app_state.llm_model,
                retriever=app_state.retriever,
                config=domain_config
            )
            
            # Update document generator
            app_state.doc_generator = EnhancedDocumentGenerator(
                llm=app_state.llm_model,
                retriever=app_state.retriever,
                embedding_model=app_state.embedding
            )
            
            logger.info(f"Successfully initialized and activated domain: {domain_name}")
            
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"Data ingestion completed successfully for domain: {domain_name}",
                    "domain_name": domain_name,
                    "collection_name": domain_config['collection_name'],
                    "vector_db_path": domain_config['db_vector_path'],
                    "data_path": domain_config['data_path'],
                    "active_domain": app_state.active_domain,
                    "status": "success",
                    "next_step": "You can now query this domain using any query endpoint, or switch to another domain using /select-domain"
                }
            )
        else:
            logger.error(f"Data ingestion returned None for domain: {domain_name}")
            raise HTTPException(
                status_code=500,
                detail="Data ingestion failed. Check logs for details."
            )
            
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Domain folder not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in /generate-embedding for domain {domain_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during data ingestion: {str(e)}"
        )

@rag_router.get("/source-guides", response_model=SourceGuideResponse, summary="Get All Source Guides for Active Domain")
def get_all_source_guides(
    max_text_for_summary: int = Query(8000, description="Maximum number of characters to feed to LLM for summarization per file.")
):
    """
    **Get all unique PDF filenames and a generated source guide for each.**

    - Uses the currently active domain's vector store.
    - Retrieves all unique `source_file` names.
    - For each file, it retrieves the first few chunks of text (sorted by chunk_id).
    - Uses an LLM to generate a ~300-500 word "source guide" (summary) based on that text.
    - This can be slow if there are many files, as it involves multiple LLM calls.
    """
    ensure_active_domain()
    logger.info(f"[DOMAIN: {app_state.active_domain}] Starting source guide generation...")
    start_time = time.time()

    try:
        # First try to get from vectorstores dict
        vectorstore = app_state.vectorstores.get(app_state.active_domain)
        
        # If not in dict but we have a data ingestion object, get it from there
        if not vectorstore and app_state.dataingestion_obj:
            logger.info(f"Vectorstore not in cache, retrieving from data ingestion object...")
            try:
                # Access the vectorstore directly from the data ingestion object
                from src.retriever import RetrieverManager
                config_manager = ConfigurationManager()
                domain_config = config_manager.get_domain_config(app_state.active_domain)
                
                retriever_manager = RetrieverManager(
                    embeddings=app_state.embedding,
                    config_dict=domain_config
                )
                vectorstore = retriever_manager.vectorstore
                
                # Cache it for future use
                app_state.vectorstores[app_state.active_domain] = vectorstore
                logger.info(f"Successfully retrieved and cached vectorstore for domain: {app_state.active_domain}")
            except Exception as e:
                logger.error(f"Failed to retrieve vectorstore: {e}")
        
        if not vectorstore:
            raise HTTPException(
                status_code=404, 
                detail=f"No vectorstore found for domain '{app_state.active_domain}'. Please run /generate-embedding first to initialize the domain."
            )

        llm = app_state.llm_model
        if not llm:
            raise HTTPException(status_code=503, detail="LLM model is not initialized.")

        # 1. Get all unique file names
        all_metadata = vectorstore.get(include=["metadatas"])
        unique_files = sorted(list(set(
            meta.get('source_file') for meta in all_metadata.get('metadatas', []) if meta.get('source_file')
        )))
        
        logger.info(f"Found {len(unique_files)} unique files: {unique_files}")

        if not unique_files:
            return SourceGuideResponse(domain=app_state.active_domain, source_guides=[])

        response_list = []

        # 2. Generate guide for each file
        for file_name in unique_files:
            try:
                # Get chunks for this specific file
                # We sort by chunk_id to try and get the first chunks
                # This relies on 'chunk_id' being consistently added during ingestion
                file_chunks = vectorstore.get(
                    where={"source_file": file_name},
                    include=["documents", "metadatas"]
                )
                
                documents = file_chunks.get('documents', [])
                metadatas = file_chunks.get('metadatas', [])
                
                if not documents:
                    logger.warning(f"No documents found for file: {file_name}")
                    response_list.append(SourceGuide(
                        file_name=file_name,
                        source_guide=None,
                        error="No text chunks found for this file in the database."
                    ))
                    continue

                # Combine documents and metadatas, then sort by chunk_id if possible
                chunk_data = []
                for doc, meta in zip(documents, metadatas):
                    chunk_data.append({
                        "text": doc,
                        "chunk_id": meta.get('chunk_id', float('inf')) # Use inf if no id
                    })
                
                # Sort by chunk_id
                sorted_chunks = sorted(chunk_data, key=lambda x: x['chunk_id'])
                
                # Concatenate text from sorted chunks up to the limit
                concatenated_text = ""
                for chunk in sorted_chunks:
                    current_len = len(concatenated_text)
                    chunk_len = len(chunk['text'])
                    if current_len + chunk_len > max_text_for_summary:
                        # If adding the whole chunk is too much, try to add a part of it
                        remaining_space = max_text_for_summary - current_len
                        if remaining_space > 100: # Only add if it's a meaningful amount
                             concatenated_text += chunk['text'][:remaining_space]
                        break # Stop after this chunk
                    
                    concatenated_text += chunk['text'] + "\n\n"
                
                if not concatenated_text:
                     logger.warning(f"No text could be concatenated for file: {file_name}")
                     response_list.append(SourceGuide(
                        file_name=file_name,
                        source_guide=None,
                        error="Text chunks were found but could not be concatenated."
                    ))
                     continue

                # 3. Call LLM to summarize
                prompt = f"""You are an expert summarization assistant. Based on the initial text provided from a document named '{file_name}', generate a 'source guide' of approximately 300-500 words. 

This guide should:
1.  Start with a clear statement of the document's main purpose or central topic.
2.  Summarize the key sections, arguments, or information presented in the initial text.
3.  Identify the target audience if possible (e.g., researchers, students, general public).
4.  Conclude with a statement about what the reader can expect to learn from the full document.

Maintain a professional and informative tone.

DOCUMENT TEXT:
---
{concatenated_text}
---

SOURCE GUIDE:
"""
                
                logger.info(f"Generating guide for {file_name} using {len(concatenated_text)} chars...")
                # Assuming llm.invoke returns a response object with a 'content' attribute
                llm_response = llm.invoke(prompt)
                guide_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)


                response_list.append(SourceGuide(
                    file_name=file_name,
                    source_guide=guide_text
                ))

            except Exception as e_file:
                logger.error(f"Failed to generate guide for {file_name}: {e_file}", exc_info=True)
                response_list.append(SourceGuide(
                    file_name=file_name,
                    source_guide=None,
                    error=f"An unexpected error occurred: {str(e_file)}"
                ))

        end_time = time.time()
        logger.info(f"Source guide generation finished in {end_time - start_time:.2f} seconds.")
        
        return SourceGuideResponse(
            domain=app_state.active_domain,
            source_guides=response_list
        )

    except Exception as e:
        logger.error(f"Error in /source-guides endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating source guides: {str(e)}"
        )
# ============================================================================
# ALL EXISTING ENDPOINTS (Modified to show active domain info)
# ============================================================================

# Modify helper function to check active domain
def ensure_active_domain():
    """Ensures an active domain is set before querying"""
    ensure_initialized()
    if not app_state.active_domain:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "No active domain set",
                "available_domains": app_state.available_domains,
                "initialized_domains": list(app_state.retrievers.keys()),
                "action": "Use GET /domains to see available domains, then POST /generate-embedding or POST /select-domain"
            }
        )

@rag_router.post("/generate-test-data", summary="Generate Test Data for Evaluation")
def generate_test_data(
    test_size: int = Query(20, description="Number of question-answer pairs to generate."),
    output_path: str = Query("generation_evaluation_dataset.json", description="Path to save the generated JSON file.")
):
    """
    Generates a test dataset of question-answer pairs from the documents in the 'dataset' folder.
    """
    ensure_initialized()
    try:
        test_data_generator = TestDataGeneration(model=app_state.llm_model)
        generated_file = test_data_generator.generator(test_size=test_size, output_path=output_path)
        
        return {
            "message": "Test data generated successfully.",
            "file_location": os.path.abspath(generated_file),
            "test_size": test_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during test data generation: {str(e)}")

@rag_router.post(
    "/gemini/analyze",
    response_model=URLAnalysisResponse,
    summary="🔗 Gemini URL Analyzer — Single URL Analysis",
    description="""
    **Analyze a single URL with Gemini AI and optional custom instructions.**
    
    **Powered by:**
    • Gemini 2.5 Flash (newest model with google_search)
    • Google Search grounding (real-time web access)
    • Smart content extraction and analysis
    
    **Example Use Cases:**
    1. Summarize an article
    2. Extract structured data from a webpage
    3. Analyze product details
    4. Get insights from documentation
    
    **Request format:**
    ```json
    {
        "url": "https://example.com/article",
        "instruction": "Summarize the main points"
    }
    ```
    
    **Response includes:**
    • Analyzed content from Gemini
    • URL processed
    • Grounding metadata (when available)
    • Error handling with clear messages
    """
)
async def gemini_analyze_url(
    url: str = Body(..., description="URL to analyze"),
    instruction: Optional[str] = Body(None, description="Custom instruction for analysis (optional)"),
):
    """
    Analyze a single URL using Gemini AI.
    Accepts one URL per request with optional custom instructions.
    """
    
    # Check if app is initialized
    if not app_state.is_initialized():
        raise HTTPException(
            status_code=503, 
            detail="System is still initializing. Please try again in a moment."
        )
    
    logger.info(f"[GEMINI ENDPOINT] Analyzing URL: {url}")
    
    try:
        # Initialize connector
        connector = GeminiURLConnectorFixed()
        
        # Build params
        params = {
            'url': url,
        }
        
        if instruction:
            params['instruction'] = instruction
        
        # Execute analysis
        result = await connector.execute(params)
        
        # Build response
        if result['success']:
            return URLAnalysisResponse(
                success=True,
                content=result.get('content'),
                urls_analyzed=[url],
                grounding_used=result.get('grounding_used', False),
                metadata=result.get('metadata', {})
            )
        else:
            return URLAnalysisResponse(
                success=False,
                error=result.get('error', 'Unknown error occurred'),
                urls_analyzed=[url]
            )
    
    except Exception as e:
        logger.error(f"[GEMINI ENDPOINT] Unhandled exception: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Gemini analysis failed: {str(e)}"
        )


# ============================================================================
# 4. HELPER FUNCTION FOR INTEGRATION
# ============================================================================

async def enhance_query_with_gemini_url(query: str) -> tuple[str, Optional[str]]:
    """
    Utility function to detect a URL in query and use Gemini to analyze it.
    Can be used in other endpoints to enhance queries automatically.
    
    Args:
        query: User query potentially containing a URL
    
    Returns:
        (enhanced_query, url_analysis) tuple
    """
    try:
        import os
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            logger.warning("[GEMINI] GOOGLE_API_KEY not set, skipping URL enhancement")
            return query, None
        
        # Initialize analyzer
        analyzer = GeminiURLAnalyzer(api_key)
        urls = analyzer.extract_urls(query)
        
        if not urls:
            logger.info("[GEMINI] No URLs found in query")
            return query, None
        
        # Use only the first URL found
        url = urls[0]
        logger.info(f"[GEMINI] Processing URL: {url}")
        
        # Analyze URL with Gemini
        result = await analyzer.analyze_query_with_urls(query)
        
        if result.success and result.content:
            # Create enhanced query with URL analysis
            enhanced_query = (
                f"Original query: {query}\n\n"
                f"Gemini analysis of URL:\n"
                f"{result.content}\n\n"
                f"Please use this analysis to answer the user's question."
            )
            
            logger.info("[GEMINI] ✓ Successfully enhanced query with URL analysis")
            return enhanced_query, result.content
        else:
            logger.warning(f"[GEMINI] URL analysis failed: {result.error}")
            return query, None
    
    except Exception as e:
        logger.error(f"[GEMINI] Error in enhance_query_with_gemini_url: {e}", exc_info=True)
        return query, None
# Replace your existing /chat-query endpoint in rag_routes.py with this:

def _build_epistemic_dashboard(agent_metrics: Optional[Dict]) -> Optional["EpistemicDashboard"]:
    """Build EpistemicDashboard from agent.get_metrics() output."""
    if not agent_metrics:
        return None
    try:
        loop_trace = []
        for lm in agent_metrics.get("loops", []):
            loop_trace.append(EpistemicLoopMetric(
                loop=lm["loop"],
                action=lm["action"],
                action_type=lm.get("action_type", "unknown"),
                efe_score=lm.get("efe_score", 0.0),
                confidence=lm.get("confidence", 0.0),
                entropy=lm.get("entropy", 0.0),
                vfe=lm.get("vfe", 0.0),
                surprisal=lm.get("surprisal", 0.0),
                is_hallucination=lm.get("is_hallucination", False),
                info_gain=lm.get("info_gain", 0.0),
                pragmatic_value=lm.get("pragmatic_value", 0.0),
                beliefs=lm.get("beliefs", {}),
                concentration=lm.get("concentration", {}),
            ))
        
        return EpistemicDashboard(
            final_confidence=agent_metrics.get("final_confidence", 0.0),
            final_entropy=agent_metrics.get("final_entropy", 0.0),
            final_vfe=agent_metrics.get("final_vfe", 0.0),
            total_loops=agent_metrics.get("total_loops", 0),
            converged=agent_metrics.get("converged", False),
            convergence_reason=agent_metrics.get("convergence_reason", ""),
            final_beliefs=agent_metrics.get("final_beliefs", {}),
            loop_trace=loop_trace,
            security=agent_metrics.get("security", {}),
        )
    except Exception as e:
        print(f"[EpistemicDashboard] Error: {e}")
        return None

@rag_router.post("/chat-query", response_model=ChatResponse)
async def generate_response(
    query: str,
    use_crag: bool = Query(False, description="Set to true to use the CRAG pipeline"),
    agent_type: str = Query("react", description="Agent type to use: 'react', 'plan_and_execute', or 'epistemic'"),
    include_steps: bool = Query(True, description="Include intermediate agent steps in response"),
    comprehensive: bool = Query(False, description="Force comprehensive detailed answers (useful for list/search queries)"),
    enable_hitl: bool = Query(False, description="Enable Human-in-the-Loop mode (halts before execution)")
):
    """Universal endpoint that handles all query types with proper LLM synthesis.
    
    **NEW: Set comprehensive=True for detailed results when finding/listing items**
    
    Examples:
    - "Find GitHub repositories for TensorFlow" with comprehensive=True → Shows ALL repos with full details
    - "What is TensorFlow?" with comprehensive=False → Concise answer
    """
    ensure_initialized()
    try:
        start_time = time.time()
        
        # ===== SAFETY CHECK: Validate input prompt =====
        print(f"[SAFETY] Checking input safety for query: {query[:100]}...")
        input_safety_result = app_state.safety_guard.check_prompt_safety(query)
        
        # Log security incidents
        if not input_safety_result.is_safe:
            log_security_incident(query, input_safety_result, "/chat-query")
        
        # Block requests with HIGH or CRITICAL threat levels
        if input_safety_result.threat_level in [ThreatLevel.LOW,ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            print(f"[SAFETY] Blocking request due to {input_safety_result.threat_level} threat level")
            
            return ChatResponse(
                final_answer="I cannot process this request as it appears to contain potentially harmful or malicious content. Please rephrase your query in a safe and appropriate manner.",
                agent_steps=[],
                agent_type=agent_type,
                metadata={
                    "query_timestamp": datetime.datetime.fromtimestamp(start_time).isoformat(),
                    "blocked": True,
                    "reason": "Safety violation detected"
                },
                safety_check=SafetyCheckInfo(
                    is_safe=input_safety_result.is_safe,
                    threat_level=input_safety_result.threat_level.value,
                    violation_type=input_safety_result.violation_type.value,
                    confidence_score=input_safety_result.confidence_score,
                    explanation=input_safety_result.explanation,
                    recommendations=input_safety_result.recommendations
                )
            )
        
        # Proceed with MEDIUM or LOW threat levels with warning
        if input_safety_result.threat_level == ThreatLevel.MEDIUM:
            print(f"[SAFETY] Proceeding with MEDIUM threat level - monitoring response")
        
        # ===== NORMAL PROCESSING =====
        final_answer = None
        agent_steps = []
        full_response_to_log = None
        agent_type_used = agent_type
        
        if use_crag:
            agent_type_used = "crag"
            print("--- Using CRAG Pipeline ---")
            response = app_state.crag_pipeline.invoke(query)
            full_response_to_log = response
            
            # Use synthesizer for CRAG responses too
            synthesizer = UniversalAnswerSynthesizer(app_state.llm_model)
            raw_answer = response.get("final_answer", str(response))
            final_answer = synthesizer.synthesize_final_answer(query, [raw_answer])
            
            if include_steps:
                agent_steps.append(AgentStep(
                    step_number=1,
                    action="crag_pipeline",
                    action_input={"query": query},
                    observation=f"Action: {response.get('action_taken')}, Score: {response.get('max_relevance_score')}",
                    thought="Evaluating document relevance and augmenting with web search if necessary."
                ))
        
        elif agent_type == "plan_and_execute":
            print("--- Using Plan-and-Execute Graph Agent ---")
            graph_input = {"input": query, "past_steps": []}

            if enable_hitl:
                print("--- Enabling HITL for Plan-and-Execute ---")
                # Create a fresh agent for this request with HITL enabled
                # Note: unique thread_id for session tracking would be handled here
                checkpointer = MemorySaver()
                hitl_agent = create_graph_agent(checkpointer=checkpointer, interrupt_before=["executor"])
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}
                response_state = hitl_agent.invoke(graph_input, config)
                print(f"HITL Execution paused or completed. Thread ID: {thread_id}")
            else:
                response_state = app_state.graph_agent.invoke(graph_input)
            
            # Extract all tool outputs from plan execution
            tool_outputs = []
            if "past_steps" in response_state:
                tool_outputs = [str(observation) for _, observation in response_state["past_steps"]]
            
            # Use universal synthesizer with comprehensive flag
            synthesizer = UniversalAnswerSynthesizer(app_state.llm_model)
            
            # Override detail level if comprehensive flag is set
            if comprehensive:
                # Manually set to comprehensive mode
                query_type = "list_search"
                detail_level = "COMPREHENSIVE"
                processed_data = synthesizer._process_tool_responses(tool_outputs, query_type, detail_level)
                final_answer = synthesizer.synthesis_chain.invoke({
                    "query": query,
                    "tool_data": processed_data,
                    "query_type": query_type,
                    "detail_level": detail_level
                })
            else:
                final_answer = synthesizer.synthesize_final_answer(query, tool_outputs)
            
            full_response_to_log = response_state

            if include_steps and "past_steps" in response_state:
                for i, (action, observation) in enumerate(response_state["past_steps"]):
                    agent_steps.append(AgentStep(
                        step_number=i + 1,
                        action=action.tool,
                        action_input=action.tool_input,
                        observation=str(observation)[:300] + "..." if len(str(observation)) > 300 else str(observation),
                        thought=f"Executing step {i+1} of the generated plan."
                    ))
        elif agent_type == "epistemic":
            agent_type_used = "epistemic"
            print("--- Using Epistemic Active Inference Agent ---")
            
            from src.epistemic_agent.enhanced_agent import EnhancedEpistemicAgent
            
            # Create epistemic agent instance
            epistemic_agent = EnhancedEpistemicAgent(
                max_iterations=5
            )
            
            # Capture steps via callback
            epistemic_steps_raw = []
            def on_epistemic_step(step_data):
                epistemic_steps_raw.append(step_data)
            
            epistemic_agent.on_step = on_epistemic_step
            
            # Run the async epistemic agent
            epistemic_response = await epistemic_agent.run(query)
            
            final_answer = epistemic_response if isinstance(epistemic_response, str) else str(epistemic_response)
            
            # Capture epistemic metrics from agent
            _epistemic_agent_metrics = epistemic_agent.get_metrics()
            
            full_response_to_log = {
                "epistemic_response": final_answer,
                "epistemic_steps": len(epistemic_steps_raw),
                "epistemic_metrics": _epistemic_agent_metrics
            }
            
            # Map epistemic AgentStep objects to the endpoint's AgentStep model
            if include_steps:
                for i, es in enumerate(epistemic_steps_raw):
                    agent_steps.append(AgentStep(
                        step_number=i + 1,
                        action=getattr(es, 'action_name', 'unknown'),
                        action_input={
                            "type": getattr(es, 'action_type', 'unknown').value if hasattr(getattr(es, 'action_type', None), 'value') else str(getattr(es, 'action_type', 'unknown')),
                            "efe_score": round(getattr(es, 'efe_score', 0.0), 4),
                            "confidence": round(getattr(es, 'confidence', 0.0), 4),
                        },
                        observation=str(getattr(es, 'observation', ''))[:300],
                        thought=f"Loop {getattr(es, 'loop_number', i+1)}: EFE={getattr(es, 'efe_score', 0.0):.2f}, Confidence={getattr(es, 'confidence', 0.0):.1%}"
                    ))

        else:
            agent_type_used = "react"
            print("--- Using ReAct Agent Executor ---")
            
            # Use universal callback handler
            step_capture = UniversalAgentStepCapture(app_state.llm_model, query)
            callbacks = [step_capture] if include_steps else []
            
            # Execute with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = app_state.agent_executor.invoke(
                        {"input": query},
                        {"callbacks": callbacks}
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        response = {"output": f"Technical difficulties processing: {query}"}
                    else:
                        continue
            
            # Get synthesized final answer
            if comprehensive:
                # Force comprehensive synthesis
                synthesizer = UniversalAnswerSynthesizer(app_state.llm_model)
                query_type = "list_search"
                detail_level = "COMPREHENSIVE"
                processed_data = synthesizer._process_tool_responses(step_capture.tool_outputs, query_type, detail_level)
                final_answer = synthesizer.synthesis_chain.invoke({
                    "query": query,
                    "tool_data": processed_data,
                    "query_type": query_type,
                    "detail_level": detail_level
                })
            else:
                final_answer = step_capture.final_answer
            
            # Fallback synthesis if callback didn't work
            if not final_answer:
                synthesizer = UniversalAnswerSynthesizer(app_state.llm_model)
                raw_outputs = [response.get("output", "")]
                if response.get("intermediate_steps"):
                    raw_outputs.extend([str(step[1]) for step in response["intermediate_steps"] if len(step) >= 2])
                final_answer = synthesizer.synthesize_final_answer(query, raw_outputs)
            
            agent_steps = step_capture.steps
            full_response_to_log = response

        # Final quality check
        if not final_answer or len(final_answer.strip()) < 5:
            final_answer = "I wasn't able to provide a complete answer. Please try rephrasing your question."
        
        # ===== SAFETY CHECK: Validate output =====
        print("[SAFETY] Checking output safety...")
        output_safety_result = app_state.safety_guard.check_output_safety(final_answer, query)
        
        # Log if output is unsafe
        if not output_safety_result.is_safe:
            log_security_incident(
                f"OUTPUT for query: {query}", 
                output_safety_result, 
                "/chat-query"
            )
        
        # Block unsafe outputs
        if input_safety_result.threat_level in [ThreatLevel.LOW,ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            print(f"[SAFETY] Blocking unsafe output - {output_safety_result.threat_level}")
            final_answer = "I generated a response, but it appears to contain content that doesn't meet safety guidelines. Please try rephrasing your query."
        
        end_time = time.time() 
        
        # Save interaction log
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RESPONSE_LOG_DIR, f"full_interaction_{timestamp}.json")
        with open(filename, "w", encoding="utf-8") as f:
            log_data = {
                "query": query, 
                "agent_type": agent_type_used, 
                "final_answer": final_answer,
                "agent_steps": [s.model_dump() for s in agent_steps], 
                "full_response": full_response_to_log,
                "comprehensive_mode": comprehensive,
                "input_safety_check": {
                    "is_safe": input_safety_result.is_safe,
                    "threat_level": input_safety_result.threat_level.value,
                    "violation_type": input_safety_result.violation_type.value,
                    "confidence": input_safety_result.confidence_score,
                    "explanation": input_safety_result.explanation
                },
                "output_safety_check": {
                    "is_safe": output_safety_result.is_safe,
                    "threat_level": output_safety_result.threat_level.value,
                    "violation_type": output_safety_result.violation_type.value,
                    "confidence": output_safety_result.confidence_score,
                    "explanation": output_safety_result.explanation
                },
                "metadata": {
                    "query_timestamp": datetime.datetime.fromtimestamp(start_time).isoformat(),
                    "completion_timestamp": datetime.datetime.fromtimestamp(end_time).isoformat()
                }
            }
            json.dump(log_data, f, indent=4, default=str)
        
        # Build epistemic dashboard (only for epistemic agent)
        total_time = end_time - start_time
        epistemic_dashboard = None
        if agent_type_used == "epistemic":
            epistemic_dashboard = _build_epistemic_dashboard(
                full_response_to_log.get("epistemic_metrics") if isinstance(full_response_to_log, dict) else None
            )
        
        return ChatResponse(
            final_answer=final_answer,
            agent_steps=agent_steps,
            agent_type=agent_type_used,
            metadata={
                "query_timestamp": datetime.datetime.fromtimestamp(start_time).isoformat(),
                "completion_timestamp": datetime.datetime.fromtimestamp(end_time).isoformat(),
                "total_time": total_time,
                "comprehensive_mode": comprehensive
            },
            epistemic_metrics=epistemic_dashboard,
            safety_check=SafetyCheckInfo(
                is_safe=input_safety_result.is_safe and output_safety_result.is_safe,
                threat_level=max(
                    input_safety_result.threat_level.value,
                    output_safety_result.threat_level.value
                ),
                violation_type=input_safety_result.violation_type.value if not input_safety_result.is_safe else output_safety_result.violation_type.value,
                confidence_score=(input_safety_result.confidence_score + output_safety_result.confidence_score) / 2,
                explanation=f"Input: {input_safety_result.explanation}; Output: {output_safety_result.explanation}",
                recommendations=list(set(input_safety_result.recommendations + output_safety_result.recommendations))
            )
        )
    except Exception as e:
        print(f"Error in /chat-query endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        return ChatResponse(
            final_answer="I encountered an error while processing your question. Please try again.",
            agent_steps=[],
            agent_type=agent_type,
            metadata={
                "error": str(e), 
                "query_timestamp": datetime.datetime.fromtimestamp(time.time()).isoformat(),
                "completion_timestamp": datetime.datetime.fromtimestamp(time.time()).isoformat()
            },
            safety_check=SafetyCheckInfo(
                is_safe=True,
                threat_level="safe",
                violation_type="none",
                confidence_score=0.0,
                explanation="Safety check skipped due to error",
                recommendations=[]
            )
        )

class DeepResearchRequest(BaseModel):
    query: str = Field(..., description="The research question or topic to investigate")
    max_iterations: int = Field(5, ge=3, le=8, description="Maximum research iterations (3-8, default: 5)")
    include_full_report: bool = Field(True, description="Include full formatted report in response")
    agent_type: str = Field("enhanced_deep_research", description="Agent for final synthesis: enhanced_deep_research, react, plan_and_execute, epistemic")
    comprehensive: bool = Field(True, description="Use comprehensive synthesis mode")

@rag_router.post("/deep-research", response_model=DeepResearchResponse)
async def deep_research_endpoint(
    request: DeepResearchRequest
):
    """
    Performs comprehensive deep research on a given query using multiple information sources.
    
    This endpoint:
    - Analyzes the query type (comparison, timeline, explanation, etc.)
    - Generates structured research queries
    - Executes searches across multiple sources
    - Builds a knowledge graph
    - Performs fact verification
    - Generates a structured, well-cited report
    - Suggests follow-up questions
    
    Ideal for:
    - Complex research questions
    - Comparative analyses (e.g., "Compare Tokyo and Mumbai public transportation")
    - Multi-faceted topics requiring diverse sources
    - Academic or professional research
    """
    ensure_initialized()  # Add this to ensure app is initialized
    
    try:
        start_time = time.time()
        
        query = request.query
        max_iterations = request.max_iterations
        include_full_report = request.include_full_report

        logger.info(f"--- Starting Deep Research for: '{query}' ---")
        logger.info(f"Max iterations: {max_iterations}")
        
        # FIX: Get config_dict from app_state
        config_dict = app_state.config_dict
        
        if not config_dict:
            raise HTTPException(
                status_code=500,
                detail="Configuration not loaded. Application may not be properly initialized."
            )
        
        # Execute deep research
        result = await execute_deep_research(
            query=query,
            max_iterations=max_iterations,
            config_dict=config_dict
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Extract metadata
        metadata = result.get("metadata", {})
        final_answer = result.get("final_answer", "")
        
        # Save interaction log
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RESPONSE_LOG_DIR, f"deep_research_{timestamp}.json")
        
        log_data = {
            "query": query,
            "agent_type": "enhanced_deep_research",
            "final_answer": final_answer,
            "metadata": metadata,
            "processing_time": processing_time,
            "timestamp": timestamp
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4, default=str)
        
        logger.info(f"Deep research completed in {processing_time:.2f}s")
        
        # ===== AGENT-BASED FINAL SYNTHESIS =====
        agent_type_used = request.agent_type
        epistemic_dashboard = None
        safety_info = None
        
        if agent_type_used == "epistemic" and hasattr(app_state, 'epistemic_agent') and app_state.epistemic_agent:
            logger.info("[DEEP-RESEARCH] Using Epistemic agent for final synthesis")
            try:
                epistemic_query = f"Based on the following research findings, provide a comprehensive analysis:\n\n{final_answer[:6000]}"
                epistemic_result = app_state.epistemic_agent.run(epistemic_query, comprehensive=True)
                
                if isinstance(epistemic_result, dict):
                    epistemic_answer = epistemic_result.get("output", epistemic_result.get("final_answer", final_answer))
                    if epistemic_answer and len(epistemic_answer) > 50:
                        final_answer = epistemic_answer
                    
                    _metrics = epistemic_result.get("epistemic_metrics")
                    if _metrics:
                        epistemic_dashboard = _build_epistemic_dashboard(_metrics)
                    
                    agent_type_used = "epistemic_deep_research"
            except Exception as e:
                logger.warning(f"Epistemic synthesis failed, using original: {e}")
                agent_type_used = "enhanced_deep_research"
        
        elif agent_type_used in ["react", "plan_and_execute"] and hasattr(app_state, 'llm_model') and app_state.llm_model:
            logger.info(f"[DEEP-RESEARCH] Using {agent_type_used} agent for final synthesis")
            try:
                synthesizer = UniversalAnswerSynthesizer(app_state.llm_model)
                synthesized = synthesizer.synthesize_final_answer(query, [final_answer])
                if synthesized and len(synthesized) > 50:
                    final_answer = synthesized
                agent_type_used = f"{agent_type_used}_deep_research"
            except Exception as e:
                logger.warning(f"{agent_type_used} synthesis failed, using original: {e}")
                agent_type_used = "enhanced_deep_research"
        
        # Safety check on final answer
        if hasattr(app_state, 'safety_guard') and app_state.safety_guard:
            try:
                output_safety = app_state.safety_guard.check_output_safety(final_answer, query)
                safety_info = {
                    "is_safe": output_safety.is_safe,
                    "threat_level": output_safety.threat_level.value,
                    "violation_type": output_safety.violation_type.value,
                    "confidence_score": output_safety.confidence_score,
                    "explanation": output_safety.explanation
                }
            except Exception as e:
                logger.warning(f"Safety check failed: {e}")
        
        # Prepare response
        response = DeepResearchResponse(
            final_answer=final_answer if include_full_report else "Report generated successfully. Set include_full_report=true to view full content.",
            agent_type=agent_type_used,
            metadata={
                "query_timestamp": metadata.get("query_timestamp", ""),
                "processing_time_seconds": metadata.get("processing_time_seconds", processing_time),
                "query_type": metadata.get("query_type", "unknown"),
                "log_file": filename,
                "active_domain": app_state.active_domain,
                "sources_used": metadata.get("sources_count", 0),
                "synthesis_agent": agent_type_used
            },
            follow_up_questions=metadata.get("follow_up_questions", []),
            knowledge_gaps=metadata.get("knowledge_gaps_in_final_metadata", []),
            sources_count=metadata.get("sources_count", 0),
            iterations_used=metadata.get("iterations_used", 0),
            epistemic_metrics=epistemic_dashboard,
            safety_check=safety_info
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in /deep-research endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Deep research failed: {str(e)}"
        )
    

@rag_router.post("/deep-research-with-explainability", response_model=ExplainabilityResponse)
async def deep_research_with_explainability(
    query: str = Query(..., description="The research question or topic to investigate"),
    max_iterations: int = Query(5, ge=3, le=8, description="Maximum research iterations"),
    include_cot_visualization: bool = Query(True, description="Include Chain-of-Thought visualization"),
    include_attribution_report: bool = Query(True, description="Include tool attribution report")
):
    """
    **Performs deep research with comprehensive explainability tracking**
    
    This endpoint combines the power of deep research with full explainability:
    - Chain-of-Thought tracing showing research reasoning process
    - Tool attribution tracking which searches contributed what
    - Source citations for transparency
    - Quality metrics and decision points
    
    Perfect for:
    - Understanding how research conclusions were reached
    - Auditing research methodology
    - Educational purposes showing research process
    - Building trust through transparency
    """
    ensure_initialized()
    
    try:
        start_time = time.time()
        
        # Safety check
        logger.info(f"[SAFETY] Checking input safety for deep research: {query[:100]}...")
        input_safety_result = app_state.safety_guard.check_prompt_safety(query)
        
        if not input_safety_result.is_safe:
            log_security_incident(query, input_safety_result, "/deep-research-with-explainability")
        
        if input_safety_result.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            logger.info(f"[SAFETY] Blocking request due to {input_safety_result.threat_level} threat level")
            
            return ExplainabilityResponse(
                final_answer="I cannot process this request as it appears to contain potentially harmful content.",
                agent_steps=[],
                metadata={
                    "query_timestamp": datetime.datetime.fromtimestamp(start_time).isoformat(),
                    "blocked": True,
                    "reason": "Safety violation detected"
                },
                explainability_results={},
                safety_check=SafetyCheckInfo(
                    is_safe=input_safety_result.is_safe,
                    threat_level=input_safety_result.threat_level.value,
                    violation_type=input_safety_result.violation_type.value,
                    confidence_score=input_safety_result.confidence_score,
                    explanation=input_safety_result.explanation,
                    recommendations=input_safety_result.recommendations
                )
            )
        
        # Initialize explainability trackers
        trace_id = app_state.cot_tracer.start_trace(
            query=query,
            agent_type="deep_research",
            metadata={"max_iterations": max_iterations}
        )
        
        attr_id = app_state.attribution_tracker.start_tracking(
            query=query,
            metadata={"agent_type": "deep_research"}
        )
        
        logger.info(f"--- Starting Deep Research with Explainability ---")
        logger.info(f"CoT Trace ID: {trace_id}")
        logger.info(f"Attribution ID: {attr_id}")
        
        # Get config
        config_dict = app_state.config_dict
        
        if not config_dict:
            raise HTTPException(
                status_code=500,
                detail="Configuration not loaded."
            )
        
        # Execute deep research WITH explainability trackers
        result = await execute_deep_research(
            query=query,
            max_iterations=max_iterations,
            config_dict=config_dict,
            cot_tracer=app_state.cot_tracer,  # NEW: Pass trackers
            attribution_tracker=app_state.attribution_tracker  # NEW
        )
        
        final_answer = result.get("final_answer", "")
        
        # Safety check output
        logger.info("[SAFETY] Checking output safety...")
        output_safety_result = app_state.safety_guard.check_output_safety(final_answer, query)
        
        if not output_safety_result.is_safe:
            log_security_incident(f"OUTPUT for query: {query}", output_safety_result, "/deep-research-with-explainability")
        
        if output_safety_result.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            logger.info(f"[SAFETY] Blocking unsafe output - {output_safety_result.threat_level}")
            final_answer = "I generated a response, but it doesn't meet safety guidelines."
        
        # Finalize explainability tracking
        cot_trace = app_state.cot_tracer.end_trace(final_decision=final_answer[:200])
        attribution_report = app_state.attribution_tracker.finalize_report(final_answer)
        
        # Build explainability results
        explainability_output = {
            "summary": {
                "research_quality": cot_trace.reasoning_quality_score,
                "total_reasoning_time": cot_trace.total_reasoning_time,
                "reasoning_steps": len(cot_trace.reasoning_steps),
                "tools_used": len(attribution_report.tools_used),
                "sources_consulted": len(attribution_report.source_attributions)
            }
        }
        
        # Add CoT visualization
        if include_cot_visualization:
            explainability_output["chain_of_thought"] = {
                "trace_id": trace_id,
                "visualization": app_state.cot_tracer.visualize_trace(trace_id),
                "summary": app_state.cot_tracer.get_trace_summary(trace_id),
                "quality_score": cot_trace.reasoning_quality_score
            }
        
        # Add attribution report
        if include_attribution_report:
            explainability_output["tool_attribution"] = {
                "report_id": attr_id,
                "visual_report": app_state.attribution_tracker.generate_visual_report(attr_id),
                "citations": app_state.attribution_tracker.generate_citation_text(attr_id),
                "answer_composition": attribution_report.answer_composition,
                "tool_usage_summary": attribution_report.tool_usage_summary
            }
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Save detailed log
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RESPONSE_LOG_DIR, f"deep_research_explainability_{timestamp}.json")
        
        with open(filename, "w", encoding="utf-8") as f:
            log_data = {
                "query": query,
                "agent_type": "deep_research_with_explainability",
                "final_answer": final_answer,
                "explainability_results": {
                    "cot_trace_id": trace_id,
                    "attribution_report_id": attr_id,
                    "reasoning_quality": cot_trace.reasoning_quality_score,
                    "total_tools_used": len(attribution_report.tools_used)
                },
                "input_safety_check": {
                    "is_safe": input_safety_result.is_safe,
                    "threat_level": input_safety_result.threat_level.value,
                    "confidence": input_safety_result.confidence_score
                },
                "output_safety_check": {
                    "is_safe": output_safety_result.is_safe,
                    "threat_level": output_safety_result.threat_level.value,
                    "confidence": output_safety_result.confidence_score
                },
                "metadata": {
                    "query_timestamp": datetime.datetime.fromtimestamp(start_time).isoformat(),
                    "completion_timestamp": datetime.datetime.fromtimestamp(end_time).isoformat(),
                    "total_time": processing_time
                }
            }
            json.dump(log_data, f, indent=4, default=str)
        
        logger.info(f"Deep research with explainability completed in {processing_time:.2f}s")
        
        return ExplainabilityResponse(
            final_answer=final_answer,
            agent_steps=[],  # Deep research doesn't use traditional steps
            metadata={
                "query_timestamp": datetime.datetime.fromtimestamp(start_time).isoformat(),
                "completion_timestamp": datetime.datetime.fromtimestamp(end_time).isoformat(),
                "total_execution_time": processing_time,
                "iterations_used": result.get("metadata", {}).get("iterations_used", 0),
                "sources_count": result.get("metadata", {}).get("sources_count", 0),
                "log_file": filename,
                "cot_trace_id": trace_id,
                "attribution_report_id": attr_id
            },
            explainability_results=explainability_output,
            safety_check=SafetyCheckInfo(
                is_safe=input_safety_result.is_safe and output_safety_result.is_safe,
                threat_level=max(input_safety_result.threat_level.value, output_safety_result.threat_level.value),
                violation_type=input_safety_result.violation_type.value if not input_safety_result.is_safe else output_safety_result.violation_type.value,
                confidence_score=(input_safety_result.confidence_score + output_safety_result.confidence_score) / 2,
                explanation=f"Input: {input_safety_result.explanation}; Output: {output_safety_result.explanation}",
                recommendations=list(set(input_safety_result.recommendations + output_safety_result.recommendations))
            )
        )
        
    except Exception as e:
        logger.error(f"Error in /deep-research-with-explainability: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Deep research failed: {str(e)}")

@rag_router.post("/chat-query-with-explainability", response_model=ExplainabilityResponse)
def generate_response_with_explainability(
    query: str,
    agent_type: str = Query("react", description="Agent type to use: 'react' or 'plan_and_execute'"),
    include_steps: bool = Query(True, description="Include intermediate agent steps in response"),
    include_cot_visualization: bool = Query(True, description="Include Chain-of-Thought visualization"),
    include_attribution_report: bool = Query(True, description="Include tool attribution report")
):
    """
    Enhanced endpoint that provides comprehensive explainability through:
    1. Chain-of-Thought tracing - Shows the reasoning process
    2. Tool Attribution - Shows which tools contributed what information
    3. Source citations - Provides transparency about information sources
    """
    ensure_initialized()
    try:
        start_time = time.time()
        
        # ===== SAFETY CHECK: Validate input prompt =====
        print(f"[SAFETY] Checking input safety for query: {query[:100]}...")
        input_safety_result = app_state.safety_guard.check_prompt_safety(query)  
        
        # Log security incidents
        if not input_safety_result.is_safe:
            log_security_incident(query, input_safety_result, "/chat-query-with-explainability")
        
        # Block requests with HIGH or CRITICAL threat levels
        if input_safety_result.threat_level in [ThreatLevel.LOW,ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            print(f"[SAFETY] Blocking request due to {input_safety_result.threat_level} threat level")
            
            # FIX: Add safety_check field
            return ExplainabilityResponse(
                final_answer="I cannot process this request as it appears to contain potentially harmful or malicious content. Please rephrase your query in a safe and appropriate manner.",
                agent_steps=[],
                metadata={
                    "query_timestamp": datetime.datetime.fromtimestamp(start_time).isoformat(),
                    "blocked": True,
                    "reason": "Safety violation detected"
                },
                explainability_results={},
                safety_check=SafetyCheckInfo(  # Already present, good!
                    is_safe=input_safety_result.is_safe,
                    threat_level=input_safety_result.threat_level.value,
                    violation_type=input_safety_result.violation_type.value,
                    confidence_score=input_safety_result.confidence_score,
                    explanation=input_safety_result.explanation,
                    recommendations=input_safety_result.recommendations
                )
            )
        
        # Proceed with MEDIUM or LOW threat levels
        if input_safety_result.threat_level == ThreatLevel.MEDIUM:
            print(f"[SAFETY] Proceeding with MEDIUM threat level - monitoring response")
        # ===== NORMAL PROCESSING =====
        
        # Initialize explainability callback
        explainability_callback = ExplainabilityCallback(
            query=query,
            agent_type=agent_type,
            cot_tracer=app_state.cot_tracer,  # CHANGED
            attribution_tracker=app_state.attribution_tracker  # CHANGED
        )
        
        # Prepare callbacks
        step_capture = UniversalAgentStepCapture(app_state.llm_model, query)
        callbacks = [step_capture, explainability_callback] if include_steps else [explainability_callback]
        
        final_answer = None
        agent_steps = []
        
        # Execute agent based on type
        if agent_type == "plan_and_execute":
            print("--- Using Plan-and-Execute with Explainability ---")
            graph_input = {"input": query, "past_steps": []}
            response_state = app_state.graph_agent.invoke(graph_input)
            
            # Extract tool outputs
            tool_outputs = []
            if "past_steps" in response_state:
                tool_outputs = [str(observation) for _, observation in response_state["past_steps"]]
            
            # Synthesize answer
            synthesizer = UniversalAnswerSynthesizer(app_state.llm_model)
            final_answer = synthesizer.synthesize_final_answer(query, tool_outputs)
            
            # Build agent steps
            if include_steps and "past_steps" in response_state:
                for i, (action, observation) in enumerate(response_state["past_steps"]):
                    agent_steps.append(AgentStep(
                        step_number=i + 1,
                        action=action.tool,
                        action_input=action.tool_input,
                        observation=str(observation)[:300] + "..." if len(str(observation)) > 300 else str(observation),
                        thought=f"Executing step {i+1} of the plan"
                    ))
        else:
            print("--- Using ReAct Agent with Explainability ---")
            
            # Execute with callbacks
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = app_state.agent_executor.invoke(
                        {"input": query},
                        {"callbacks": callbacks}
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        response = {"output": f"Error processing: {query}"}
                    else:
                        continue
            
            # Get synthesized answer
            final_answer = step_capture.final_answer
            
            if not final_answer:
                synthesizer = UniversalAnswerSynthesizer(app_state.llm_model)
                raw_outputs = [response.get("output", "")]
                if response.get("intermediate_steps"):
                    raw_outputs.extend([str(step[1]) for step in response["intermediate_steps"] if len(step) >= 2])
                final_answer = synthesizer.synthesize_final_answer(query, raw_outputs)
            
            agent_steps = step_capture.steps
        
        # Quality check
        if not final_answer or len(final_answer.strip()) < 5:
            final_answer = "I wasn't able to provide a complete answer. Please try rephrasing your question."
        # ===== SAFETY CHECK: Validate output =====
        print("[SAFETY] Checking output safety...")
        output_safety_result = app_state.safety_guard.check_output_safety(final_answer, query)
        
        # Log if output is unsafe
        if not output_safety_result.is_safe:
            log_security_incident(
                f"OUTPUT for query: {query}", 
                output_safety_result, 
                "/chat-query-with-explainability"
            )
        
        # Block unsafe outputs
        if input_safety_result.threat_level in [ThreatLevel.LOW,ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            print(f"[SAFETY] Blocking unsafe output - {output_safety_result.threat_level}")
            final_answer = "I generated a response, but it appears to contain content that doesn't meet safety guidelines. Please try rephrasing your query."
         
        # Finalize explainability tracking
        explainability_results = explainability_callback.finalize_tracking(final_answer)
        
        # Build explainability response
        explainability_output = {
            "summary": explainability_results["summary"],
        }
        
        # Add Chain-of-Thought visualization if requested
        if include_cot_visualization:
            cot_trace_id = explainability_results["chain_of_thought"]["trace_id"]
            explainability_output["chain_of_thought"] = {
                "trace_id": cot_trace_id,
                "visualization": app_state.cot_tracer.visualize_trace(cot_trace_id),  # CHANGED
                "summary": app_state.cot_tracer.get_trace_summary(cot_trace_id),
                "full_trace": explainability_results["chain_of_thought"]
            }
        
        # Add Tool Attribution report if requested
        if include_attribution_report:
            attr_report_id = explainability_results["tool_attribution"]["report_id"]
            explainability_output["tool_attribution"] = {
                "report_id": attr_report_id,
                "visual_report": app_state.attribution_tracker.generate_visual_report(attr_report_id),
                "citations": app_state.attribution_tracker.generate_citation_text(attr_report_id),
                "full_report": explainability_results["tool_attribution"]
            }
        
        end_time = time.time()
        
        # Save detailed log
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RESPONSE_LOG_DIR, f"explainability_interaction_{timestamp}.json")
        
        with open(filename, "w", encoding="utf-8") as f:
            log_data = {
                "query": query,
                "agent_type": agent_type,
                "final_answer": final_answer,
                "agent_steps": [s.model_dump() for s in agent_steps],
                "explainability_results": explainability_results,
                "input_safety_check": {
                    "is_safe": input_safety_result.is_safe,
                    "threat_level": input_safety_result.threat_level.value,
                    "violation_type": input_safety_result.violation_type.value,
                    "confidence": input_safety_result.confidence_score,
                    "explanation": input_safety_result.explanation
                },
                "output_safety_check": {
                    "is_safe": output_safety_result.is_safe,
                    "threat_level": output_safety_result.threat_level.value,
                    "violation_type": output_safety_result.violation_type.value,
                    "confidence": output_safety_result.confidence_score,
                    "explanation": output_safety_result.explanation
                },
                "metadata": {
                    "query_timestamp": datetime.datetime.fromtimestamp(start_time).isoformat(),
                    "completion_timestamp": datetime.datetime.fromtimestamp(end_time).isoformat(),
                    "total_time": end_time - start_time
                }
            }
            json.dump(log_data, f, indent=4, default=str)
        
        return ExplainabilityResponse(
            final_answer=final_answer,
            agent_steps=agent_steps,
            metadata={
                "query_timestamp": datetime.datetime.fromtimestamp(start_time).isoformat(),
                "completion_timestamp": datetime.datetime.fromtimestamp(end_time).isoformat(),
                "total_execution_time": end_time - start_time,
                "log_file": filename
            },
            explainability_results=explainability_output,
            safety_check=SafetyCheckInfo(
                is_safe=input_safety_result.is_safe and output_safety_result.is_safe,
                threat_level=max(
                    input_safety_result.threat_level.value,
                    output_safety_result.threat_level.value
                ),
                violation_type=input_safety_result.violation_type.value if not input_safety_result.is_safe else output_safety_result.violation_type.value,
                confidence_score=(input_safety_result.confidence_score + output_safety_result.confidence_score) / 2,
                explanation=f"Input: {input_safety_result.explanation}; Output: {output_safety_result.explanation}",
                recommendations=list(set(input_safety_result.recommendations + output_safety_result.recommendations))
            )
        )
        
    except Exception as e:
        print(f"Error in /chat-query-with-explainability endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        return ExplainabilityResponse(
            final_answer="I encountered an error while processing your question. Please try again.",
            agent_steps=[],
            metadata={
                "error": str(e),
                "query_timestamp": datetime.datetime.fromtimestamp(time.time()).isoformat(),
                "completion_timestamp": datetime.datetime.fromtimestamp(time.time()).isoformat()
            },
            explainability_results={}
        )
@rag_router.post("/solve", response_model=ReasoningResponse, summary="Solve Problem with Reasoning Agent")
def solve_with_reasoning(
    query: str = Query(..., description="The problem or question to solve"),
    subject: SubjectDomain = Query(SubjectDomain.GENERAL, description="Subject domain hint (optional - auto-detects if general)"),
    show_work: bool = Query(True, description="Include reasoning steps in response"),
    verify_answer: bool = Query(True, description="Verify solution if possible"),
    include_confidence: bool = Query(True, description="Include confidence scoring")
):
    """
    **Solve a problem using the Enhanced Reasoning Agent v2**
    
    The agent will:
    1. **Auto-detect problem type** (math, physics, qualitative) using LLM-guided parsing
    2. **Retrieve relevant knowledge** from subject-specific knowledge bases
    3. **Apply appropriate methods**: SymPy symbolic math, numerical solvers, 2D physics, optimization
    4. **Decompose complex problems** into sub-problems when needed
    5. **Verify solutions** by substitution and consistency checks
    6. **Estimate confidence** with detailed scoring and interpretation
    7. **Recover from errors** using fallback strategies
    
    **Subjects:**
    - `mathematics`: Calculus (derivatives, integrals, limits), algebra, optimization, systems of equations
    - `physics`: Kinematics, forces, energy, projectile motion, circular motion, waves, electromagnetism
    - `economics`: Supply/demand, game theory, fiscal/monetary policy, trade
    - `psychology`: Cognitive, social, developmental psychology
    - `statistics`: Descriptive stats, probability, distributions, hypothesis testing, regression
    - `social_science`: Political systems, international relations, anthropology
    - `general`: **Auto-detects domain** (recommended for mixed problems)
    
    **New Features:**
    - **Multi-step decomposition**: Handles "find X, then find Y" problems
    - **2D Physics**: Projectile motion, inclined planes with friction
    - **Numerical fallbacks**: When symbolic methods fail
    - **Unit tracking**: Automatic unit conversion and consistency checking
    - **Error recovery**: Retries with alternative strategies
    - **Verification**: Checks answers by substitution
    
    **Examples:**
    ```
    # Calculus
    "Find the derivative of sin(x²)"
    "Integrate x² + 3x from 0 to 2"
    "Maximize f(x) = -x² + 4x - 3"
    
    # Physics
    "Ball thrown at 20 m/s at 45°. Find max height."
    "10kg block on 30° incline, friction μ=0.2. Find acceleration."
    
    # Multi-step
    "If a(t) = 2t, find average velocity from t=0 to t=5"
    ```
    """
    try:
        start_time = datetime.datetime.now()
        
        # Get singleton reasoning agent
        agent = get_reasoning_agent()
        
        # Add subject hint only if not general (let agent auto-detect otherwise)
        if subject != SubjectDomain.GENERAL:
            enhanced_query = f"[Subject: {subject.value}] {query}"
        else:
            enhanced_query = query
        
        logger.info(f"[REASONING] Subject: {subject.value}, Query: {query[:100]}...")
        
        # Solve with the enhanced engine
        result = agent.solve(enhanced_query, show_work=show_work)
        
        # Check for success
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown error occurred')
            logger.error(f"[REASONING] Failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Extract problem type and subtype
        problem_type = result.get('problem_type', 'unknown')
        parsed_info = result.get('parsed', {})
        subtype = parsed_info.get('subtype')
        
        # Build reasoning steps
        steps = []
        for step_dict in result.get('steps', []):
            steps.append(ReasoningStep(
                step_num=step_dict['step_num'],
                thought=step_dict['thought'],
                action=step_dict['action'],
                observation=step_dict['observation']
            ))
        
        # Extract answer
        answer = result.get('answer', 'No answer generated')
        
        # Build comprehensive metadata
        metadata = {
            "timestamp": start_time.isoformat(),
            "processing_time_seconds": (datetime.datetime.now() - start_time).total_seconds(),
            "subject_hint": subject.value,
            "detected_domain": parsed_info.get('type', 'unknown'),
            "parsing_method": "LLM-guided" if parsed_info.get('parse_success') else "regex-fallback",
            "parsed_info": parsed_info,
            "computation_details": result.get('details', {}),
        }
        
        # Add decomposition info if problem was decomposed
        if result.get('decomposed', False):
            metadata["decomposed"] = True
            metadata["decomposition_method"] = result.get('method', 'unknown')
            metadata["sub_problems"] = result.get('sub_problems', [])
        
        # Add confidence scoring if requested
        if include_confidence and result.get('confidence'):
            confidence = result['confidence']
            metadata["confidence"] = {
                "score": confidence['score'],
                "level": confidence['level'],
                "interpretation": confidence['interpretation'],
                "factors": confidence.get('factors', [])
            }
        
        # Add verification results if available
        if verify_answer and result.get('details', {}).get('verification'):
            verification = result['details']['verification']
            metadata["verification"] = {
                "verified": verification.get('verified', False),
                "summary": verification.get('summary', 'N/A'),
                "details": verification.get('results', [])
            }
        
        # Add error recovery info if used
        if result.get('details', {}).get('recovery_attempted'):
            metadata["recovery_used"] = True
            metadata["recovery_strategy"] = result['details'].get('fallback', 'unknown')
        
        logger.info(f"[REASONING] ✓ Success - Type: {problem_type}/{subtype}, Steps: {len(steps)}, Time: {metadata['processing_time_seconds']:.2f}s")
        if result.get('confidence'):
            logger.info(f"[REASONING] Confidence: {result['confidence']['level']} ({result['confidence']['score']:.2f})")
        
        return ReasoningResponse(
            success=True,
            query=query,
            subject=subject.value,
            problem_type=problem_type,
            subtype=subtype,
            answer=answer,
            steps=steps,
            metadata=metadata
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"[REASONING] Error in /solve endpoint: {e}", exc_info=True)
        
        # Return user-friendly error response
        error_detail = str(e)
        if "SymPy" in error_detail:
            error_detail = "Mathematical computation error. Please check the expression format."
        elif "parsing" in error_detail.lower():
            error_detail = "Could not parse the problem. Please rephrase more clearly."
        
        raise HTTPException(
            status_code=500,
            detail=f"Reasoning agent error: {error_detail}"
        )
    
@rag_router.get("/run-metrics", summary="Run overall pipeline metrics evaluation")
def run_pipeline_metrics():
    """
    Runs the comprehensive pipeline metrics evaluation on all saved log files.
    """
    try:
        log_files = [str(p) for p in Path(RESPONSE_LOG_DIR).glob("*.json")]
        if not log_files:
            return JSONResponse(status_code=200, content={"message": "No log files found to evaluate."})
        
        evaluator = PipelineMetricsEvaluator()
        results = evaluator.evaluate_all_metrics(log_files)
        
        # Save the detailed evaluation results to a file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path("evaluation_results") / f"pipeline_metrics_report_{timestamp}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        return JSONResponse(status_code=200, content={
            "message": "Pipeline metrics evaluation completed successfully.",
            "results": results,
            "report_file": str(output_file)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during pipeline metrics evaluation: {str(e)}")

@rag_router.post("/doc_question", response_model=DocumentQueryResponseWithMindMap, summary="Enhanced RAG with Reranking and Optional Mind Map")
def document_question(
    query: str = Query(..., description="The question to ask the RAG pipeline."),
    k: int = Query(15, ge=5, le=30, description="Initial number of documents to retrieve"),
    top_n: int = Query(8, ge=3, le=15, description="Final number after reranking"),
    enable_reranking: bool = Query(True, description="Enable cross-encoder reranking"),
    generate_mindmap: bool = Query(False, description="Generate mind map from answer"),
    mindmap_max_nodes: int = Query(40, ge=10, le=100, description="Maximum nodes in mind map"),
    mindmap_depth: Literal["shallow", "balanced", "deep"] = Query("balanced", description="Mind map depth preference")
):
    """
    **Query active domain database with RAG and optional mind map generation**
    
    - Uses currently active domain's vector store
    - To switch domains: Use /select-domain endpoint
    - To see active domain: Use /domains endpoint
    
    Enhanced features:
    - Increased retrieval depth (k=15 default)
    - Cross-encoder reranking
    - Comprehensive answer generation
    - Better context utilization
    - Optional mind map visualization of the answer
    
    **Mind Map Feature:**
    Set `generate_mindmap=true` to get a hierarchical visualization of the answer.
    The mind map helps you understand the structure and relationships in the response.
    """
    ensure_active_domain()  # Check domain is set
    
    logger.info(f"[DOMAIN: {app_state.active_domain}] Received /doc_question: {query}")
    logger.info(f"Config: k={k}, top_n={top_n}, reranking={enable_reranking}, mindmap={generate_mindmap}")
    start_time = time.time()
    
    try:
        # Use active domain's retriever (already configured)
        retriever = app_state.retriever
        
        # Retrieve documents
        retrieved_docs = retriever.get_relevant_documents(query)
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents after processing")
        
        # Build comprehensive context
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'unknown'))
            page = doc.metadata.get('page', 0)
            
            context_parts.append(
                f"[Document {i+1}] (Source: {source}, Page: {page})\n{doc.page_content}\n"
            )
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt for comprehensive answers
        enhanced_prompt = f"""You are an expert assistant. Answer the following question using the provided context.

CRITICAL INSTRUCTIONS:
1. Provide a COMPREHENSIVE answer including ALL relevant details from the context
2. If the context contains tables, lists, or structured information, preserve that structure
3. Include specific technical terms, names, and categories mentioned
4. Organize your answer with clear sections and subsections
5. Do not summarize - provide full details

CONTEXT:
{context}

QUESTION: {query}

COMPREHENSIVE ANSWER:"""
        
        # Generate answer
        result = app_state.chain_obj.invoke({
            "input": query,
            "context": enhanced_prompt
        })
        
        answer = result.get("answer", "No answer found.")
        
        # Generate mind map if requested
        mindmap = None
        if generate_mindmap:
            logger.info(f"[DOMAIN: {app_state.active_domain}] Generating mind map from answer...")
            try:
                mindmap = generate_mindmap_from_content(
                    content=answer,
                    topic=query,
                    max_nodes=mindmap_max_nodes,
                    depth=mindmap_depth
                )
                
                if mindmap:
                    # Add domain information to mindmap metadata
                    mindmap.metadata["source_domain"] = app_state.active_domain
                    mindmap.metadata["domain_collection"] = app_state.dataingestion_obj.collection_name if app_state.dataingestion_obj else None
                    mindmap.metadata["query"] = query
                    mindmap.metadata["answer_length"] = len(answer)
                    
                    logger.info(f"Mind map generated: {mindmap.statistics.total_nodes} nodes, {mindmap.statistics.total_edges} edges")
            except Exception as e:
                logger.error(f"Failed to generate mind map: {e}", exc_info=True)
                # Continue without mind map rather than failing the entire request
                mindmap = None
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Format response with enhanced metadata
        formatted_context = []
        for doc in retrieved_docs:
            formatted_context.append(DocumentContextWithMindMap(
                source=doc.metadata.get('source_file', doc.metadata.get('source', 'unknown')),
                page=doc.metadata.get('page', 0),
                content_snippet=doc.page_content[:300] + "...",
                rerank_score=doc.metadata.get('relevance_score'),
                doc_type=doc.metadata.get('doc_type')
            ))
        
        logger.info(f"Successfully answered in {total_time:.2f}s with {len(retrieved_docs)} sources")
        logger.info(f"Answer length: {len(answer)} characters")
        if mindmap:
            logger.info(f"Mind map: {mindmap.statistics.total_nodes} nodes, depth={mindmap.statistics.max_depth}")
        
        return DocumentQueryResponseWithMindMap(
            answer=answer,
            retrieved_context=formatted_context,
            metadata={
                "query": query,
                "active_domain": app_state.active_domain,
                "collection_name": app_state.dataingestion_obj.collection_name if app_state.dataingestion_obj else None,
                "data_path": app_state.dataingestion_obj.data_path if app_state.dataingestion_obj else None,
                "query_timestamp": datetime.datetime.fromtimestamp(start_time).isoformat(),
                "total_time": total_time,
                "num_sources": len(formatted_context),
                "initial_k": k,
                "final_top_n": top_n,
                "reranking_enabled": enable_reranking,
                "answer_length": len(answer),
                "mindmap_generated": mindmap is not None,
                "mindmap_config": {
                    "max_nodes": mindmap_max_nodes,
                    "depth": mindmap_depth
                } if generate_mindmap else None
            },
            mindmap=mindmap
        )
    
    except Exception as e:
        logger.error(f"Error in /doc_question endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while querying the RAG pipeline: {str(e)}"
        )


@rag_router.post("/mindmap/full_vectorstore", response_model=MindMapResponse, summary="Generate Mind Map from Active Domain's Vector Store")
def generate_mindmap_from_vectorstore(
    topic: str = Query(..., description="Central topic for the mind map"),
    max_nodes: int = Query(50, ge=10, le=100, description="Maximum nodes in mind map"),
    depth_preference: Literal["shallow", "balanced", "deep"] = Query("balanced", description="Depth preference"),
    num_documents: int = Query(20, ge=5, le=50, description="Number of documents to retrieve from vector store"),
    output_file: Optional[str] = Query(None, description="Optional: Save to JSON file (e.g., 'healthcare_mindmap.json')"),
    export_mermaid: bool = Query(False, description="Also export as Mermaid diagram")
):
    """
    **Generate comprehensive mind map from active domain's entire vector store**
    
    **What this does:**
    1. Retrieves relevant documents from the **ACTIVE DOMAIN'S** vector store based on the topic
    2. Aggregates the content from multiple documents
    3. Generates a hierarchical mind map with clusters and relationships
    4. Optionally saves to JSON file and/or Mermaid diagram
    
    **Before using:**
    - Ensure you have an active domain set (use /domains to check)
    - Switch domains with /select-domain if needed
    - The mind map will be generated from: `{active_domain}` database
    
    **Use cases:**
    - Overview of a topic across multiple documents
    - Understanding relationships between concepts
    - Visual knowledge mapping of domain content
    - Educational material creation
    - Research topic exploration
    
    **Mind Map Depth Options:**
    - `shallow`: 3 levels, broad overview
    - `balanced`: 5 levels, good detail (recommended)
    - `deep`: 7 levels, maximum detail
    
    **Example:**
```
    topic: "Machine Learning Algorithms"
    num_documents: 20
    depth_preference: "balanced"
    output_file: "ml_algorithms_mindmap.json"
```
    """
    ensure_active_domain()  # Ensure domain is set
    
    try:
        start_time = time.time()
        logger.info(f"[DOMAIN: {app_state.active_domain}] Generating mind map from vector store for topic: {topic}")
        logger.info(f"Config: max_nodes={max_nodes}, depth={depth_preference}, num_docs={num_documents}")
        
        # Step 1: Use active domain's retriever
        retriever = app_state.retriever
        
        # Retrieve documents related to the topic
        retrieved_docs = retriever.get_relevant_documents(topic)
        logger.info(f"[DOMAIN: {app_state.active_domain}] Retrieved {len(retrieved_docs)} documents from vector store")
        
        if not retrieved_docs:
            raise HTTPException(
                status_code=404,
                detail=f"No documents found in domain '{app_state.active_domain}' for the topic: {topic}"
            )
        
        # Limit to requested number of documents
        retrieved_docs = retrieved_docs[:num_documents]
        
        # Step 2: Aggregate content from all documents
        aggregated_content = []
        sources = []
        source_details = []
        
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source_file', doc.metadata.get('source', f'doc_{i}'))
            page = doc.metadata.get('page', 0)
            sources.append(source)
            
            source_details.append({
                "source": source,
                "page": page,
                "content_length": len(doc.page_content)
            })
            
            # Add document content with source information
            aggregated_content.append(
                f"=== Source: {source} (Page {page}) ===\n{doc.page_content}\n"
            )
        
        full_content = "\n\n".join(aggregated_content)
        logger.info(f"Aggregated content length: {len(full_content)} characters from {len(retrieved_docs)} documents")
        
        # Step 3: Generate mind map using the app's mind map generator
        logger.info("Generating hierarchical mind map structure...")
        mindmap = app_state.mindmap_generator.generate_from_text(
            text_content=full_content,
            topic=topic,
            depth_preference=depth_preference
        )
        
        # Step 4: Add comprehensive metadata
        mindmap.metadata["source_domain"] = app_state.active_domain
        mindmap.metadata["domain_collection"] = app_state.dataingestion_obj.collection_name if app_state.dataingestion_obj else None
        mindmap.metadata["domain_data_path"] = app_state.dataingestion_obj.data_path if app_state.dataingestion_obj else None
        mindmap.metadata["sources"] = list(set(sources))
        mindmap.metadata["source_details"] = source_details
        mindmap.metadata["num_documents_used"] = len(retrieved_docs)
        mindmap.metadata["num_documents_requested"] = num_documents
        mindmap.metadata["content_length"] = len(full_content)
        mindmap.metadata["query_topic"] = topic
        
        # Step 5: Save to files if requested
        saved_files = {}
        
        if output_file:
            os.makedirs("mindmaps", exist_ok=True)
            
            # Ensure domain-specific subdirectory
            domain_dir = os.path.join("mindmaps", app_state.active_domain)
            os.makedirs(domain_dir, exist_ok=True)
            
            # Save JSON
            json_filename = output_file if output_file.endswith('.json') else f"{output_file}.json"
            json_path = os.path.join(domain_dir, json_filename)
            saved_json_path = app_state.mindmap_generator.export_to_json_file(mindmap, json_path)
            saved_files["json"] = saved_json_path
            logger.info(f"Mind map JSON saved to: {saved_json_path}")
        
        if export_mermaid:
            os.makedirs("mindmaps", exist_ok=True)
            domain_dir = os.path.join("mindmaps", app_state.active_domain)
            os.makedirs(domain_dir, exist_ok=True)
            
            # Generate Mermaid diagram
            mermaid_content = app_state.mindmap_generator.export_to_mermaid(mindmap)
            
            # Save Mermaid file
            mermaid_filename = output_file.replace('.json', '.mmd') if output_file else f"{topic.replace(' ', '_')}_mindmap.mmd"
            mermaid_path = os.path.join(domain_dir, mermaid_filename)
            
            with open(mermaid_path, 'w', encoding='utf-8') as f:
                f.write(mermaid_content)
            
            saved_files["mermaid"] = mermaid_path
            logger.info(f"Mermaid diagram saved to: {mermaid_path}")
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Step 6: Log success metrics
        logger.info("=" * 60)
        logger.info(f"Mind map generated successfully in {generation_time:.2f}s")
        logger.info(f"  Domain: {app_state.active_domain}")
        logger.info(f"  Topic: {topic}")
        logger.info(f"  Nodes: {mindmap.statistics.total_nodes}")
        logger.info(f"  Edges: {mindmap.statistics.total_edges}")
        logger.info(f"  Max Depth: {mindmap.statistics.max_depth}")
        logger.info(f"  Clusters: {mindmap.statistics.cluster_count}")
        logger.info(f"  Sources Used: {len(set(sources))}")
        logger.info(f"  Documents Processed: {len(retrieved_docs)}")
        if saved_files:
            logger.info(f"  Saved Files: {saved_files}")
        logger.info("=" * 60)
        
        # Add generation info to metadata
        mindmap.metadata["generation_time_seconds"] = generation_time
        mindmap.metadata["generation_timestamp"] = datetime.datetime.fromtimestamp(start_time).isoformat()
        if saved_files:
            mindmap.metadata["saved_files"] = saved_files
        
        # Add usage instructions
        mindmap.metadata["usage_instructions"] = {
            "visualization": "Use the Mermaid diagram with https://mermaid.live or any Mermaid-compatible viewer",
            "json_structure": "The JSON contains full node/edge data with metadata for programmatic use",
            "switching_domains": "Use /select-domain to analyze different domain databases",
            "available_endpoints": {
                "export": "GET /api/rag/mindmap/export/{generation_id} - Export previous mind maps",
                "visualize": "POST /api/rag/mindmap/visualize - Get Mermaid visualization",
                "query_with_mindmap": "POST /api/rag/doc_question?generate_mindmap=true - Get answer + mind map"
            }
        }
        
        return mindmap
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error generating mind map from vector store: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate mind map from domain '{app_state.active_domain}': {str(e)}"
        )
# ============================================================================
# DOCUMENT GENERATION ENDPOINTS
# ============================================================================

@rag_router.post("/generate/briefing", response_model=GeneratedDocument)
def generate_briefing_document(
    topic: str = Query(..., description="Topic for the briefing document"),
    tone: Optional[str] = Query("professional", description="Tone of the document"),
    target_audience: Optional[str] = Query("executives", description="Target audience"),
    num_sources: Optional[int] = Query(5, description="Number of sources to use", ge=1, le=20),
    include_citations: bool = Query(True, description="Include citations"),
    multi_stage: bool = Query(False, description="Use multi-stage generation"),
    additional_instructions: Optional[str] = Query(None, description="Additional instructions")
):
    """
    **Generate briefing from active domain database**
    
    - Uses: Current active domain's knowledge base
    - Perfect for business reports and strategic decision-making
    - Switch domain: Use /select-domain endpoint
    """
    ensure_active_domain()
    
    logger.info(f"[DOMAIN: {app_state.active_domain}] Generating briefing on: {topic}")
    
    config = GenerationConfig(
        topic=topic,
        document_type=DocumentType.BRIEFING,
        tone=tone,
        target_audience=target_audience,
        length="long",
        num_sources=num_sources,
        include_citations=include_citations,
        multi_stage_generation=multi_stage,
        max_iterations=3 if multi_stage else 1,
        additional_instructions=additional_instructions
    )
    
    try:
        result = app_state.doc_generator.generate_document(config)
        
        generation_id = str(uuid.uuid4())
        result.metadata["generation_id"] = generation_id
        result.metadata["source_domain"] = app_state.active_domain
        result.metadata["domain_collection"] = app_state.dataingestion_obj.collection_name if app_state.dataingestion_obj else None
        result.metadata["domain_data_path"] = app_state.dataingestion_obj.data_path if app_state.dataingestion_obj else None
        
        generation_history[generation_id] = GenerationHistoryItem(
            generation_id=generation_id,
            timestamp=datetime.datetime.now().isoformat(),
            request=DocumentGenerationRequest(
                topic=topic,
                document_type=DocumentType.BRIEFING,
                tone=tone,
                target_audience=target_audience,
                num_sources=num_sources,
                include_citations=include_citations,
                multi_stage_generation=multi_stage,
                additional_instructions=additional_instructions
            ),
            response=result
        )
        
        logger.info(f"Generated briefing (ID: {generation_id}, Domain: {app_state.active_domain}, Quality: {result.quality_metrics.overall_score:.2f})")
        return result
        
    except Exception as e:
        logger.error(f"Error generating briefing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@rag_router.post("/generate/study-guide", response_model=GeneratedDocument)
def generate_study_guide(
    topic: str = Query(..., description="Topic for the study guide"),
    language_level: Literal["beginner", "intermediate", "advanced"] = Query("intermediate"),
    target_audience: Optional[str] = Query("students", description="Target audience"),
    num_sources: Optional[int] = Query(5, description="Number of sources", ge=1, le=20),
    include_visual_suggestions: bool = Query(True, description="Suggest visual aids"),
    additional_instructions: Optional[str] = Query(None)
):
    """
    **Generate study guide from active domain database**
    
    - Uses: Current active domain's knowledge base
    - Comprehensive with quiz, answers, essay questions, and glossary
    - Optimized for learning and retention
    """
    ensure_active_domain()
    
    logger.info(f"[DOMAIN: {app_state.active_domain}] Generating study guide on: {topic}")
    
    config = GenerationConfig(
        topic=topic,
        document_type=DocumentType.STUDY_GUIDE,
        tone="educational",
        target_audience=target_audience,
        language_level=language_level,
        length="long",
        num_sources=num_sources,
        include_visual_suggestions=include_visual_suggestions,
        additional_instructions=additional_instructions
    )
    
    try:
        result = app_state.doc_generator.generate_document(config)
        
        generation_id = str(uuid.uuid4())
        result.metadata["generation_id"] = generation_id
        result.metadata["source_domain"] = app_state.active_domain
        result.metadata["domain_collection"] = app_state.dataingestion_obj.collection_name if app_state.dataingestion_obj else None
        
        generation_history[generation_id] = GenerationHistoryItem(
            generation_id=generation_id,
            timestamp=datetime.datetime.now().isoformat(),
            request=DocumentGenerationRequest(
                topic=topic,
                document_type=DocumentType.STUDY_GUIDE,
                target_audience=target_audience,
                language_level=language_level,
                num_sources=num_sources,
                include_visual_suggestions=include_visual_suggestions,
                additional_instructions=additional_instructions
            ),
            response=result
        )
        
        logger.info(f"Generated study guide (ID: {generation_id}, Domain: {app_state.active_domain})")
        return result
        
    except Exception as e:
        logger.error(f"Error generating study guide: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/generate/blog-post", response_model=GeneratedDocument)
def generate_blog_post(
    topic: str = Query(..., description="Topic for the blog post"),
    tone: Optional[str] = Query("conversational", description="Tone"),
    target_audience: Optional[str] = Query("general readers", description="Target audience"),
    num_sources: Optional[int] = Query(5, ge=1, le=20),
    include_visual_suggestions: bool = Query(True, description="Suggest visuals"),
    additional_instructions: Optional[str] = Query(None)
):
    """
    **Generate blog post from active domain database**
    
    - Uses: Current active domain's knowledge base
    - Engaging, scannable with compelling headline
    - Optimized for web readability
    """
    ensure_active_domain()
    
    logger.info(f"[DOMAIN: {app_state.active_domain}] Generating blog post on: {topic}")
    
    config = GenerationConfig(
        topic=topic,
        document_type=DocumentType.BLOG_POST,
        tone=tone,
        target_audience=target_audience,
        length="medium",
        num_sources=num_sources,
        include_visual_suggestions=include_visual_suggestions,
        additional_instructions=additional_instructions
    )
    
    try:
        result = app_state.doc_generator.generate_document(config)
        
        generation_id = str(uuid.uuid4())
        result.metadata["generation_id"] = generation_id
        result.metadata["source_domain"] = app_state.active_domain
        result.metadata["domain_collection"] = app_state.dataingestion_obj.collection_name if app_state.dataingestion_obj else None
        
        generation_history[generation_id] = GenerationHistoryItem(
            generation_id=generation_id,
            timestamp=datetime.datetime.now().isoformat(),
            request=DocumentGenerationRequest(
                topic=topic,
                document_type=DocumentType.BLOG_POST,
                tone=tone,
                target_audience=target_audience,
                num_sources=num_sources,
                include_visual_suggestions=include_visual_suggestions,
                additional_instructions=additional_instructions
            ),
            response=result
        )
        
        logger.info(f"Generated blog post (ID: {generation_id}, Domain: {app_state.active_domain})")
        return result
        
    except Exception as e:
        logger.error(f"Error generating blog post: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/generate/technical-report", response_model=GeneratedDocument)
def generate_technical_report(
    topic: str = Query(..., description="Topic for technical report"),
    target_audience: str = Query("technical professionals", description="Target audience"),
    language_level: Literal["intermediate", "advanced"] = Query("advanced"),
    num_sources: int = Query(10, ge=5, le=20, description="Number of sources"),
    include_citations: bool = Query(True),
    additional_instructions: Optional[str] = Query(None)
):
    """
    **Generate technical report from active domain database**
    
    - Uses: Current active domain's knowledge base
    - Rigorous with detailed analysis and methodology
    - Includes abstract, theoretical framework, references
    """
    ensure_active_domain()
    
    logger.info(f"[DOMAIN: {app_state.active_domain}] Generating technical report on: {topic}")
    
    config = GenerationConfig(
        topic=topic,
        document_type=DocumentType.TECHNICAL_REPORT,
        tone="formal",
        target_audience=target_audience,
        language_level=language_level,
        length="long",
        num_sources=num_sources,
        include_citations=include_citations,
        additional_instructions=additional_instructions
    )
    
    try:
        result = app_state.doc_generator.generate_document(config)
        
        generation_id = str(uuid.uuid4())
        result.metadata["generation_id"] = generation_id
        result.metadata["source_domain"] = app_state.active_domain
        result.metadata["domain_collection"] = app_state.dataingestion_obj.collection_name if app_state.dataingestion_obj else None
        
        logger.info(f"Generated technical report (ID: {generation_id}, Domain: {app_state.active_domain})")
        return result
        
    except Exception as e:
        logger.error(f"Error generating technical report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/generate/tutorial", response_model=GeneratedDocument)
def generate_tutorial(
    topic: str = Query(..., description="Topic for tutorial"),
    language_level: Literal["beginner", "intermediate", "advanced"] = Query("beginner"),
    target_audience: str = Query("learners", description="Target audience"),
    num_sources: int = Query(5, ge=1, le=15),
    include_visual_suggestions: bool = Query(True),
    additional_instructions: Optional[str] = Query(None)
):
    """
    **Generate tutorial from active domain database**
    
    - Uses: Current active domain's knowledge base
    - Step-by-step with clear instructions
    - Perfect for hands-on learning
    """
    ensure_active_domain()
    
    logger.info(f"[DOMAIN: {app_state.active_domain}] Generating tutorial on: {topic}")
    
    config = GenerationConfig(
        topic=topic,
        document_type=DocumentType.TUTORIAL,
        tone="instructional",
        target_audience=target_audience,
        language_level=language_level,
        length="long",
        num_sources=num_sources,
        include_visual_suggestions=include_visual_suggestions,
        additional_instructions=additional_instructions
    )
    
    try:
        result = app_state.doc_generator.generate_document(config)
        generation_id = str(uuid.uuid4())
        result.metadata["generation_id"] = generation_id
        result.metadata["source_domain"] = app_state.active_domain
        result.metadata["domain_collection"] = app_state.dataingestion_obj.collection_name if app_state.dataingestion_obj else None
        
        logger.info(f"Generated tutorial (ID: {generation_id}, Domain: {app_state.active_domain})")
        return result
        
    except Exception as e:
        logger.error(f"Error generating tutorial: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/generate/comparative-analysis", response_model=GeneratedDocument)
def generate_comparative_analysis(
    topic: str = Query(..., description="Items/approaches to compare"),
    target_audience: str = Query("decision makers"),
    num_sources: int = Query(8, ge=5, le=20),
    additional_instructions: Optional[str] = Query(None)
):
    """
    **Generate comparative analysis from active domain database**
    
    - Uses: Current active domain's knowledge base
    - Balanced with evaluation criteria
    - Ideal for decision support
    """
    ensure_active_domain()
    
    logger.info(f"[DOMAIN: {app_state.active_domain}] Generating comparative analysis on: {topic}")
    
    config = GenerationConfig(
        topic=topic,
        document_type=DocumentType.COMPARATIVE_ANALYSIS,
        tone="analytical",
        target_audience=target_audience,
        length="long",
        num_sources=num_sources,
        include_citations=True,
        additional_instructions=additional_instructions
    )
    
    try:
        result = app_state.doc_generator.generate_document(config)
        generation_id = str(uuid.uuid4())
        result.metadata["generation_id"] = generation_id
        result.metadata["source_domain"] = app_state.active_domain
        result.metadata["domain_collection"] = app_state.dataingestion_obj.collection_name if app_state.dataingestion_obj else None
        
        logger.info(f"Generated comparative analysis (ID: {generation_id}, Domain: {app_state.active_domain})")
        return result
        
    except Exception as e:
        logger.error(f"Error generating comparative analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/generate/faq", response_model=GeneratedDocument)
def generate_faq(
    topic: str = Query(..., description="Topic for FAQ"),
    target_audience: str = Query("general users"),
    num_sources: int = Query(5, ge=1, le=15),
    additional_instructions: Optional[str] = Query(None)
):
    """
    **Generate FAQ from active domain database**
    
    - Uses: Current active domain's knowledge base
    - Comprehensive, organized by category
    - Anticipates common questions
    """
    ensure_active_domain()
    
    logger.info(f"[DOMAIN: {app_state.active_domain}] Generating FAQ on: {topic}")
    
    config = GenerationConfig(
        topic=topic,
        document_type=DocumentType.FAQ,
        tone="helpful",
        target_audience=target_audience,
        language_level="beginner",
        length="medium",
        num_sources=num_sources,
        additional_instructions=additional_instructions
    )
    
    try:
        result = app_state.doc_generator.generate_document(config)
        generation_id = str(uuid.uuid4())
        result.metadata["generation_id"] = generation_id
        result.metadata["source_domain"] = app_state.active_domain
        result.metadata["domain_collection"] = app_state.dataingestion_obj.collection_name if app_state.dataingestion_obj else None
        
        logger.info(f"Generated FAQ (ID: {generation_id}, Domain: {app_state.active_domain})")
        return result
        
    except Exception as e:
        logger.error(f"Error generating FAQ: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/generate/chat", response_model=GeneratedDocument)
def generate_chat_response(
    topic: str = Query(..., description="Question or topic"),
    length: Literal["short", "long"] = Query("short"),
    persona: str = Query("helpful assistant"),
    num_sources: int = Query(3, ge=1, le=10)
):
    """
    **Generate chat response from active domain database**
    
    - Uses: Current active domain's knowledge base
    - Conversational, natural flowing
    - Short or comprehensive based on length parameter
    """
    ensure_active_domain()
    
    logger.info(f"[DOMAIN: {app_state.active_domain}] Generating {length} chat response on: {topic}")
    
    doc_type = DocumentType.CHAT_SHORT if length == "short" else DocumentType.CHAT_LONG
    
    config = GenerationConfig(
        topic=topic,
        document_type=doc_type,
        persona=persona,
        tone="conversational",
        length=length,
        num_sources=num_sources,
        include_citations=False
    )
    
    try:
        result = app_state.doc_generator.generate_document(config)
        generation_id = str(uuid.uuid4())
        result.metadata["generation_id"] = generation_id
        result.metadata["source_domain"] = app_state.active_domain
        result.metadata["domain_collection"] = app_state.dataingestion_obj.collection_name if app_state.dataingestion_obj else None
        
        logger.info(f"Generated {length} chat response (ID: {generation_id}, Domain: {app_state.active_domain})")
        return result
        
    except Exception as e:
        logger.error(f"Error generating chat response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/generate/custom", response_model=GeneratedDocument)
def generate_custom_document(request: DocumentGenerationRequest):
    """
    **Generate custom document from active domain database**
    
    - Uses: Current active domain's knowledge base
    - Full customization of all generation parameters
    - Maximum control over generation process
    """
    ensure_active_domain()
    
    logger.info(f"[DOMAIN: {app_state.active_domain}] Generating custom {request.document_type} on: {request.topic}")
    
    config = GenerationConfig(
        topic=request.topic,
        document_type=request.document_type,
        persona=request.persona,
        tone=request.tone,
        length=request.length,
        target_audience=request.target_audience,
        language_level=request.language_level,
        include_citations=request.include_citations,
        include_visual_suggestions=request.include_visual_suggestions,
        adaptive_retrieval=request.adaptive_retrieval,
        multi_stage_generation=request.multi_stage_generation,
        max_iterations=request.max_iterations,
        additional_instructions=request.additional_instructions
    )
    
    try:
        strategy = RetrievalStrategy(
            initial_k=request.num_sources,
            max_k=min(request.num_sources * 2, 20),
            relevance_threshold=0.6
        )
        
        result = app_state.doc_generator.generate_document(config, strategy=strategy)
        
        generation_id = str(uuid.uuid4())
        result.metadata["generation_id"] = generation_id
        result.metadata["source_domain"] = app_state.active_domain
        result.metadata["domain_collection"] = app_state.dataingestion_obj.collection_name if app_state.dataingestion_obj else None
        
        generation_history[generation_id] = GenerationHistoryItem(
            generation_id=generation_id,
            timestamp=datetime.datetime.now().isoformat(),
            request=request,
            response=result
        )
        
        logger.info(f"Generated custom document (ID: {generation_id}, Domain: {app_state.active_domain}, Type: {request.document_type})")
        return result
        
    except Exception as e:
        logger.error(f"Error generating custom document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/generate/refine", response_model=GeneratedDocument)
def refine_generated_document(refinement: DocumentRefinementRequest):
    """
    **Refine previously generated document**
    
    - Works with document from any domain
    - Intelligently improves content based on feedback
    - Preserves good sections while addressing concerns
    """
    ensure_active_domain()
    
    if refinement.generation_id not in generation_history:
        raise HTTPException(
            status_code=404,
            detail=f"Generation ID {refinement.generation_id} not found"
        )
    
    previous = generation_history[refinement.generation_id]
    
    logger.info(f"[DOMAIN: {app_state.active_domain}] Refining document {refinement.generation_id}")
    
    try:
        config = GenerationConfig(
            topic=previous.request.topic,
            document_type=previous.request.document_type,
            persona=previous.request.persona,
            tone=previous.request.tone,
            length=previous.request.length,
            target_audience=previous.request.target_audience,
            language_level=previous.request.language_level,
            include_citations=previous.request.include_citations,
            include_visual_suggestions=previous.request.include_visual_suggestions,
            additional_instructions=previous.request.additional_instructions
        )
        
        result = app_state.doc_generator.generate_with_feedback(
            config=config,
            feedback=refinement.feedback,
            previous_generation=previous.response.content
        )
        
        new_generation_id = str(uuid.uuid4())
        result.metadata["generation_id"] = new_generation_id
        result.metadata["refined_from"] = refinement.generation_id
        result.metadata["source_domain"] = app_state.active_domain

        result.metadata["domain_collection"] = app_state.dataingestion_obj.collection_name if app_state.dataingestion_obj else None
        
        logger.info(f"Refined document (New: {new_generation_id}, Previous: {refinement.generation_id}, Domain: {app_state.active_domain})")
        return result
        
    except Exception as e:
        logger.error(f"Error refining document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Hybrid Graph-Vector Search Endpoint ---

from src.hybrid_retriever_pipeline import HybridRetrieverPipeline
from src.graph_knowledge_manager import GraphKnowledgeManager

class HybridSearchRequest(BaseModel):
    query: str
    limit: int = 10

@rag_router.post("/hybrid-search", summary="Hybrid Graph-Vector Search")
def hybrid_search(request: HybridSearchRequest):
    """
    **Performs a Hybrid Search combining Vector Similarity and Graph Traversal.**
    
    1.  **RAG Fusion**: Expands the query into multiple variations.
    2.  **Vector Search**: Finds semantically similar documents.
    3.  **Graph Search**: Identifies entities and traverses the knowledge graph for relationships.
    4.  **Fusion**: Merges results for a comprehensive context.
    
    Required: `networkx` must be installed.
    """
    ensure_initialized()
    ensure_active_domain()
    
    try:
        # Initialize Graph Components (lazy loading for efficiency)
        if not hasattr(app_state, 'graph_manager'):
             app_state.graph_manager = GraphKnowledgeManager(app_state.llm_model)

        # Initialize Retriever Components
        config_manager = ConfigurationManager()
        domain_config = config_manager.get_domain_config(app_state.active_domain)
        retriever_manager = RetrieverManager(app_state.embedding, domain_config)
        
        # Build Pipeline
        hybrid_pipeline = HybridRetrieverPipeline(
            llm=app_state.llm_model,
            retriever_manager=retriever_manager,
            graph_manager=app_state.graph_manager
        )
        
        # Execute
        results = hybrid_pipeline.run_full_search(request.query)
        
        return results
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENHANCED RAG ENDPOINTS - WITH CITATIONS & QUALITY ASSESSMENT
# ============================================================================

class EnhancedRAGQueryRequest(BaseModel):
    """Request model for enhanced RAG queries."""
    query: str = Field(..., description="The question to answer")
    document_type: Literal["chat_short", "chat_long", "detailed"] = Field(
        "chat_short", description="Response format: short, long, or detailed"
    )
    enable_citations: bool = Field(True, description="Include citation markers [DOC1], [DOC2]")
    enable_quality_assessment: bool = Field(False, description="Assess response quality (adds latency)")
    enable_refinement: bool = Field(False, description="Enable multi-stage refinement for complex queries")
    max_refinement_iterations: int = Field(2, ge=1, le=5, description="Max refinement iterations")


class EnhancedRAGQueryResponse(BaseModel):
    """Response model for enhanced RAG queries."""
    answer: str
    document_type: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    context_documents: List[Dict[str, Any]] = Field(default_factory=list)
    quality_metrics: Optional[Dict[str, Any]] = None
    generation_trace: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@rag_router.post(
    "/enhanced-query",
    response_model=EnhancedRAGQueryResponse,
    summary="🚀 Enhanced RAG Query with Citations & Quality"
)
def enhanced_rag_query(request: EnhancedRAGQueryRequest):
    """
    **Enhanced RAG Query with Document Generator Features**
    
    This endpoint provides advanced RAG capabilities:
    
    - **Citation Tracking**: Responses include [DOC1], [DOC2] markers linking to sources
    - **Quality Assessment**: Optional scoring of coherence, relevance, completeness
    - **Multi-Stage Refinement**: Iteratively improves low-quality responses
    - **Document Types**: Choose between short, long, or detailed responses
    
    **Example Request:**
    ```json
    {
        "query": "What are the security mechanisms for autonomous agents?",
        "document_type": "detailed",
        "enable_citations": true,
        "enable_quality_assessment": true
    }
    ```
    """
    ensure_initialized()
    ensure_active_domain()
    
    try:
        start_time = time.time()
        logger.info(f"[ENHANCED RAG] Processing: {request.query[:100]}...")
        
        # Build configuration
        rag_config = RAGConfig(
            enable_citations=request.enable_citations,
            enable_quality_assessment=request.enable_quality_assessment,
            enable_multi_stage=request.enable_refinement,
            max_iterations=request.max_refinement_iterations,
            document_type=request.document_type
        )
        
        # Create enhanced RAG instance
        enhanced_rag = EnhancedRAG(
            model=app_state.llm_model,
            retriever=app_state.retriever,
            config=rag_config
        )
        
        # Invoke
        result = enhanced_rag.invoke({"input": request.query})
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Convert citations to dict format
        citations_list = [
            {
                "doc_id": cit.doc_id,
                "source": cit.source,
                "page": cit.page,
                "position": cit.position_in_output
            }
            for cit in result.citations
        ]
        
        # Convert quality metrics to dict if present
        quality_dict = None
        if result.quality_metrics:
            quality_dict = {
                "coherence": result.quality_metrics.coherence_score,
                "relevance": result.quality_metrics.relevance_score,
                "completeness": result.quality_metrics.completeness_score,
                "citation_coverage": result.quality_metrics.citation_coverage,
                "readability": result.quality_metrics.readability_score,
                "overall": result.quality_metrics.overall_score,
                "feedback": result.quality_metrics.feedback
            }
        
        logger.info(f"[ENHANCED RAG] Completed in {processing_time:.2f}s, "
                   f"citations={len(citations_list)}, quality={quality_dict.get('overall') if quality_dict else 'N/A'}")
        
        return EnhancedRAGQueryResponse(
            answer=result.answer,
            document_type=request.document_type,
            citations=citations_list,
            context_documents=result.context_documents,
            quality_metrics=quality_dict,
            generation_trace=result.generation_trace,
            metadata={
                **result.metadata,
                "processing_time_seconds": round(processing_time, 2),
                "active_domain": app_state.active_domain,
                "query_timestamp": datetime.datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"[ENHANCED RAG] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Enhanced RAG query failed: {str(e)}")


# ============================================================================
# RAG-INTEGRATED MINDMAP ENDPOINTS
# ============================================================================

class RAGMindmapRequest(BaseModel):
    """Request model for RAG-integrated mindmap generation."""
    query: str = Field(..., description="Search query for document retrieval")
    topic: Optional[str] = Field(None, description="Central topic (defaults to query)")
    max_nodes: int = Field(50, ge=10, le=100, description="Maximum nodes in mindmap")
    depth_preference: Literal["shallow", "balanced", "deep"] = Field(
        "balanced", description="Hierarchy depth preference"
    )
    num_documents: int = Field(10, ge=3, le=20, description="Number of documents to retrieve")
    focus_areas: Optional[List[str]] = Field(None, description="Optional areas to focus on")
    export_format: Optional[Literal["json", "mermaid", "html"]] = Field(
        None, description="Optional export format"
    )


class RAGMindmapQueryResponse(BaseModel):
    """Response model for RAG-integrated mindmap."""
    mindmap: Dict[str, Any]
    source_documents: List[Dict[str, Any]]
    quality_metrics: Dict[str, Any]
    generation_metadata: Dict[str, Any]
    export_content: Optional[str] = None


@rag_router.post(
    "/mindmap-from-rag",
    response_model=RAGMindmapQueryResponse,
    summary="🧠 Generate Mindmap from RAG Documents"
)
def generate_rag_mindmap(request: RAGMindmapRequest):
    """
    **Generate Mind Map from RAG-Retrieved Documents**
    
    This endpoint combines RAG retrieval with mindmap generation:
    
    - **Source Attribution**: Each node links to its source document
    - **Quality Metrics**: Coverage, coherence, and balance scores
    - **Cross-References**: Detects relationships across documents
    - **Multiple Exports**: JSON, Mermaid, or interactive HTML
    
    **Example Request:**
    ```json
    {
        "query": "machine learning security vulnerabilities",
        "topic": "ML Security Threats",
        "max_nodes": 40,
        "depth_preference": "balanced",
        "export_format": "html"
    }
    ```
    """
    ensure_initialized()
    ensure_active_domain()
    
    try:
        start_time = time.time()
        logger.info(f"[RAG MINDMAP] Generating for: {request.query[:100]}...")
        
        # Create RAG mindmap generator
        generator = RAGMindMapGenerator(
            llm=app_state.llm_model,
            retriever=app_state.retriever,
            max_nodes=request.max_nodes
        )
        
        # Generate mindmap from RAG
        result = generator.generate_from_query(
            query=request.query,
            topic=request.topic,
            focus_areas=request.focus_areas,
            depth_preference=request.depth_preference,
            k=request.num_documents
        )
        
        # Handle export format
        export_content = None
        if request.export_format == "mermaid":
            export_content = generator.export_to_mermaid_with_sources(result)
        elif request.export_format == "html":
            export_content = generator.export_to_html_with_tooltips(result)
        elif request.export_format == "json":
            export_content = result.mindmap.model_dump_json(indent=2)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Convert mindmap to dict
        mindmap_dict = result.mindmap.model_dump() if hasattr(result.mindmap, 'model_dump') else result.mindmap.dict()
        
        # Convert quality metrics to dict
        quality_dict = {
            "coverage_score": result.quality_metrics.coverage_score,
            "coherence_score": result.quality_metrics.coherence_score,
            "balance_score": result.quality_metrics.balance_score,
            "cross_reference_count": result.quality_metrics.cross_reference_count,
            "overall_score": result.quality_metrics.overall_score
        } if result.quality_metrics else {}
        
        logger.info(f"[RAG MINDMAP] Generated: {mindmap_dict.get('statistics', {}).get('total_nodes', 'N/A')} nodes, "
                   f"quality={quality_dict.get('overall_score', 'N/A')}, time={processing_time:.2f}s")
        
        return RAGMindmapQueryResponse(
            mindmap=mindmap_dict,
            source_documents=result.source_documents,
            quality_metrics=quality_dict,
            generation_metadata={
                **result.generation_metadata,
                "processing_time_seconds": round(processing_time, 2),
                "active_domain": app_state.active_domain,
                "export_format": request.export_format
            },
            export_content=export_content
        )
        
    except Exception as e:
        logger.error(f"[RAG MINDMAP] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG mindmap generation failed: {str(e)}")


@rag_router.get(
    "/enhanced-query/config",
    summary="Get Enhanced RAG Configuration Options"
)
def get_enhanced_rag_config():
    """
    **Get configuration options for enhanced RAG queries.**
    
    Returns available document types, default settings, and capability info.
    """
    return {
        "document_types": ["chat_short", "chat_long", "detailed"],
        "default_settings": {
            "enable_citations": True,
            "enable_quality_assessment": False,
            "enable_refinement": False,
            "max_refinement_iterations": 2
        },
        "capabilities": {
            "citation_tracking": "Marks sources with [DOC1], [DOC2] etc.",
            "quality_assessment": "Scores coherence, relevance, completeness, citation coverage, readability",
            "multi_stage_refinement": "Iteratively improves low-quality responses",
            "document_types": {
                "chat_short": "Concise 2-4 paragraph response",
                "chat_long": "Comprehensive multi-section response",
                "detailed": "Full technical detail with citations"
            }
        },
        "active_domain": app_state.active_domain if app_state.is_initialized() else None
    }

