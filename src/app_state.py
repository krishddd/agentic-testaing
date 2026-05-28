from typing import Optional, Any, Dict

class AppState:
    """Singleton to hold application-wide initialized objects"""
    def __init__(self):
        self.config_dict: Optional[dict] = None
        self.embedding: Optional[Any] = None
        self.llm_model: Optional[Any] = None
        
        # Multi-database support
        self.retrievers: Dict[str, Any] = {}  # domain_name -> retriever
        self.vectorstores: Dict[str, Any] = {}  # domain_name -> vectorstore
        self.dataingestion_objs: Dict[str, Any] = {}  # domain_name -> DataIngestion
        
        # Default/active retriever
        self.retriever: Optional[Any] = None
        self.dataingestion_obj: Optional[Any] = None
        
        self.agent_executor: Optional[Any] = None
        self.graph_agent: Optional[Any] = None
        self.chain_obj: Optional[Any] = None
        self.crag_pipeline: Optional[Any] = None
        self.doc_generator: Optional[Any] = None
        self.safety_guard: Optional[Any] = None
        self.cot_tracer: Optional[Any] = None
        self.attribution_tracker: Optional[Any] = None
        
        # Track available domains
        self.available_domains: list = []
        self.active_domain: Optional[str] = None
        self.mindmap_generator: Optional[Any] = None
        
        # Epistemic Agent (lazy-initialized when endpoints are called)
        self.epistemic_agent: Optional[Any] = None
        
    def is_initialized(self) -> bool:
        """Check if core components are initialized"""
        return all([
            self.config_dict is not None,
            self.embedding is not None,
            self.llm_model is not None,
            self.retriever is not None
        ])
    
    def set_active_domain(self, domain_name: str) -> bool:
        """Set the active domain for queries"""
        if domain_name in self.retrievers:
            self.retriever = self.retrievers[domain_name]
            self.active_domain = domain_name
            print(f"[DOMAIN] Switched to domain: {domain_name}")
            return True
        return False
    
    def get_domain_retriever(self, domain_name: str) -> Optional[Any]:
        """Get retriever for a specific domain"""
        return self.retrievers.get(domain_name)

# Global singleton instance
app_state = AppState()