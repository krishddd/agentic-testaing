import os
from src.app_state import app_state
from src.configuration import ConfigurationManager
from src.model import Model
from src.data_ingestion import DataIngestion
from src.retriever import RetrieverManager
from src.agent import create_enhanced_rag_agent_executor, create_graph_agent
from src.rag import RAG
from src.crag_pipeline import CRAGPipeline
from src.document_generator import EnhancedDocumentGenerator
from src.chain_of_thought_tracer import ChainOfThoughtTracer
from src.tool_attribution import ToolAttributionTracker
from src.advanced_mindmap import AdvancedMindMapGenerator
from src.prompt_safety_guard import create_safety_guard
from src.logging import logger
from src.reasoning_agent.core import ReasoningAgentSystem
from src.configuration import AgentConfig

async def initialize_application():
    """
    Initialize all application components with multi-domain support.
    """
    try:
        # 1. Load configuration
        logger.info("Loading configuration...")
        config_manager = ConfigurationManager()
        app_state.config_dict = config_manager.configurations()
        
        if not app_state.config_dict:
            raise ValueError("Configuration dictionary is empty. Check your config file.")
        
        logger.info("Configuration loaded successfully")
        
        # 2. Discover available domains
        logger.info("Scanning for domain-specific folders...")
        app_state.available_domains = config_manager.get_available_domains()
        logger.info(f"Found {len(app_state.available_domains)} domains: {app_state.available_domains}")
        
        # 3. Load models ONCE (shared across all domains)
        logger.info("Loading models (embedding + LLM)...")
        model_obj = Model(app_state.config_dict)
        app_state.embedding = model_obj.load_ollama_embedding()
        
        try:
            app_state.llm_model = model_obj.load_ollama_model_enhanced()
            logger.info("Using enhanced LLM with extended context window")
        except:
            logger.warning("Enhanced model loading failed, using standard model")
            app_state.llm_model = model_obj.load_ollama_model()
        
        logger.info("Models loaded successfully")
        
        # 4. Initialize retrievers for each domain
        if app_state.available_domains:
            logger.info("Initializing domain-specific retrievers...")
            for domain in app_state.available_domains:
                try:
                    domain_config = config_manager.get_domain_config(domain)
                    
                    # Create data ingestion object for this domain
                    app_state.dataingestion_objs[domain] = DataIngestion(
                        domain_config,
                        embedding_model=app_state.embedding
                    )
                    
                    # Create retriever for this domain
                    retriever_manager = RetrieverManager(
                        embeddings=app_state.embedding,
                        config_dict=domain_config
                    )
                    
                    app_state.retrievers[domain] = retriever_manager.get_retriever(
                        k=15,
                        enable_reranking=True,
                        top_n_rerank=8
                    )
                    
                    logger.info(f"[OK] Initialized retriever for domain: {domain}")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize domain '{domain}': {e}")
                    continue
            
            # Set first domain as active by default
            if app_state.available_domains:
                first_domain = app_state.available_domains[0]
                app_state.set_active_domain(first_domain)
                app_state.retriever = app_state.retrievers[first_domain]
                app_state.dataingestion_obj = app_state.dataingestion_objs[first_domain]
                logger.info(f"Set default active domain: {first_domain}")
        else:
            # Fallback to single domain mode
            logger.warning("No domain folders found, initializing single default retriever")
            app_state.dataingestion_obj = DataIngestion(
                app_state.config_dict,
                embedding_model=app_state.embedding
            )
            
            retriever_manager = RetrieverManager(
                embeddings=app_state.embedding,
                config_dict=app_state.config_dict
            )
            
            app_state.retriever = retriever_manager.get_retriever(
                k=15,
                enable_reranking=True,
                top_n_rerank=8
            )
        
        # 5. Initialize explainability trackers
        logger.info("Initializing explainability trackers...")
        app_state.cot_tracer = ChainOfThoughtTracer(trace_dir="traces/chain_of_thought")
        app_state.attribution_tracker = ToolAttributionTracker(attribution_dir="traces/tool_attribution")
        logger.info("Explainability trackers initialized")
        
        # 6. Initialize safety guard
        logger.info("Initializing safety guard...")
        app_state.safety_guard = create_safety_guard(app_state.config_dict)
        logger.info(f"Safety guard initialized")
        
        # 7. Create directories
        os.makedirs("api_responses", exist_ok=True)
        os.makedirs("security_logs", exist_ok=True)
        
        # 8. Initialize agents (use active retriever)
        logger.info("Initializing ReAct agent executor...")
        app_state.agent_executor = create_enhanced_rag_agent_executor(
            memory_type="summary_buffer",
            max_iterations=25
        )
        logger.info("ReAct agent executor initialized")
        
        logger.info("Initializing Plan-and-Execute graph agent...")
        app_state.graph_agent = create_graph_agent(memory_type="summary_buffer")
        logger.info("Plan-and-Execute graph agent initialized")
        
        # 9. Initialize simple RAG chain
        logger.info("Initializing simple RAG chain...")
        rag_pipeline = RAG(model=app_state.llm_model, retriever=app_state.retriever)
        app_state.chain_obj = rag_pipeline.chain
        logger.info("Simple RAG chain initialized")
        
        # 10. Initialize CRAG pipeline
        logger.info("Initializing CRAG pipeline...")
        app_state.crag_pipeline = CRAGPipeline(
            llm=app_state.llm_model,
            retriever=app_state.retriever,
            config=app_state.config_dict
        )
        logger.info("CRAG pipeline initialized")
        
        # 11. Initialize document generator
        logger.info("Initializing Enhanced Document Generator...")
        app_state.doc_generator = EnhancedDocumentGenerator(
            llm=app_state.llm_model,
            retriever=app_state.retriever,
            embedding_model=app_state.embedding
        )
        logger.info("Document generator initialized")
        # 11. Initialize Mind Map Generator - ADD THIS SECTION
        logger.info("Initializing Advanced Mind Map Generator...")
        app_state.mindmap_generator = AdvancedMindMapGenerator(
            llm=app_state.llm_model,
            max_nodes=50,
            max_depth=5
        )
        logger.info("Initializing Reasoning Agent...")
        app_state.reasoning_agent = ReasoningAgentSystem(AgentConfig())
        logger.info("Reasoning Agent initialized")
        logger.info("Mind Map Generator initialized")
        logger.info("=" * 50)
        logger.info("ALL APPLICATION COMPONENTS INITIALIZED SUCCESSFULLY")
        logger.info(f"Available domains: {app_state.available_domains}")
        logger.info(f"Active domain: {app_state.active_domain}")
        logger.info("Epistemic Agent: available (lazy-initialized on first request)")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"FATAL ERROR during initialization: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize application: {str(e)}")

async def shutdown_application():
    """Cleanup on application shutdown"""
    logger.info("Shutting down application...")
    logger.info("Application shutdown complete")