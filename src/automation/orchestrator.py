
import os
import time
import logging
from typing import List, Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from src.data_ingestion import DataIngestion
from src.evaluation import RetrievalEvaluator
from src.configuration import ConfigurationManager
from src.logging import logger
from src.deep_research_agent import build_optimized_research_graph, ResearchConfig
from src.retriever import RetrieverManager
from src.validators import save_documents
from langchain_core.documents import Document

class PipelineOrchestrator:
    """
    Autonomous agent that orchestrates the RAG pipeline:
    1. Monitors data source for new files.
    2. Triggers incremental ingestion.
    3. Runs automatic health checks (evaluation).
    4. Triggers self-healing (Deep Research) if quality drops.
    """

    def __init__(self, llm: BaseChatModel, config: Optional[dict] = None):
        self.llm = llm
        self.config_manager = ConfigurationManager()
        self.config = config or self.config_manager.configurations()
        
        self.data_path = self.config['data_path']
        self.collection_name = self.config['collection_name']
        
        # known_files_cache could be persisted to disk for robustness
        self.known_files_cache = set()
        self._initialize_file_cache()

    def _initialize_file_cache(self):
        """Populate cache with existing files to avoid re-ingesting on startup."""
        if os.path.exists(self.data_path):
            if os.path.isdir(self.data_path):
                for f in os.listdir(self.data_path):
                    if f.endswith('.pdf') or f.endswith('.txt'):
                        self.known_files_cache.add(f)
            else:
                 self.known_files_cache.add(os.path.basename(self.data_path))
        logger.info(f"PipelineOrchestrator initialized. Watching {len(self.known_files_cache)} files.")

    def monitor_and_update(self) -> Dict[str, Any]:
        """
        One-stop method to check for changes and trigger updates.
        Returns a status dict.
        """
        new_files = self._detect_new_files()
        
        if not new_files:
            return {"status": "idle", "message": "No new files detected."}
        
        logger.info(f"Detected {len(new_files)} new files: {new_files}")
        
        # 1. Trigger Ingestion
        ingestion_result = self.trigger_ingestion(new_files)
        
        # 2. Run Health Check
        health_report = self.run_health_check()
        
        return {
            "status": "updated",
            "ingested_files": new_files,
            "ingestion_info": ingestion_result,
            "health_report": health_report
        }

    def _detect_new_files(self) -> List[str]:
        """Scans the data directory for new files not in cache."""
        current_files = set()
        if os.path.exists(self.data_path) and os.path.isdir(self.data_path):
             for f in os.listdir(self.data_path):
                    if f.endswith('.pdf') or f.endswith('.txt'):
                        current_files.add(f)
        
        new_files = list(current_files - self.known_files_cache)
        return new_files

    def trigger_ingestion(self, new_files: List[str]) -> str:
        """
        Triggers data ingestion. 
        Note: Currently DataIngestion rebuilds the vector store. 
        For true incremental updates, we'd need to modify DataIngestion.
        For now, we re-run the main ingestion pipeline if changes are detected.
        """
        logger.info("Triggering Data Ingestion...")
        
        # Initialize DataIngestion
        # We need an embedding model. The main app usually sets this up. 
        # For simplicity, we assume the DataIngestion class handles its own embedding model 
        # or we pass None and it loads defaults/config.
        
        # In a real scenario, passing the embedding model object is best.
        # Here we re-instantiate everything based on config.
        from src.model import Model
        model_setup = Model(self.config)
        embeddings = model_setup.load_embedding_model()
        
        ingestion = DataIngestion(self.config, embedding_model=embeddings)
        
        # TODO: Ideally call a method to ingest ONLY new files.
        # But `load_and_split_data` loads everything in the path.
        # Since we detected new files, re-running `ingestion_main` is the safest current approach
        # to ensure the vector store is in sync with the folder.
        
        try:
            ingestion.ingestion_main()
            # Update cache
            for f in new_files:
                self.known_files_cache.add(f)
            return f"Successfully ingested {len(new_files)} new files (Rebuilt vector store)."
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return f"Ingestion failed: {str(e)}"

    def run_health_check(self) -> Dict[str, Any]:
        """
        Runs a quick evaluation to ensure retrieval is working.
        """
        logger.info("Running Pipeline Health Check...")
        
        # Setup Retriever
        from src.model import Model
        model_setup = Model(self.config)
        embeddings = model_setup.load_embedding_model()
        retriever_manager = RetrieverManager(self.config, embeddings)
        retriever = retriever_manager.get_retriever()
        
        evaluator = RetrievalEvaluator(retriever, self.config)
        
        # Use a small set of 'golden' questions for health check
        # ideally loaded from a persistent file
        chk_dataset = [
            {"question": "What is the main topic?", "expected_retrieved_phrases": ["topic", "introduction"]},
             # Add more generic checks or load from validation set
        ]
        
        # If we have a validation dataset file:
        val_file = "evaluation_dataset.json"
        if os.path.exists(val_file):
            chk_dataset = evaluator.load_evaluation_dataset(val_file)[:5] # Check first 5 items
        
        if not chk_dataset:
             return {"status": "skipped", "reason": "No evaluation dataset found."}

        results_df = evaluator.evaluate_retriever_comprehensive(chk_dataset)
        
        # Basic aggregation
        avg_precision = results_df['precision'].mean() if 'precision' in results_df else 0.0
        
        logger.info(f"Health Check Complete. Avg Precision: {avg_precision}")
        
        if avg_precision < 0.5:
            logger.warning("Pipeline Health Detection: Low Precision!")
            self._trigger_self_healing(avg_precision)
            return {"status": "warning", "avg_precision": avg_precision, "action": "Self-healing triggered"}
            
        return {"status": "healthy", "avg_precision": avg_precision}

    def _trigger_self_healing(self, score: float):
        """
        Triggers Deep Research to find missing info if retrieval quality is low.
        This is a 'simulated' healing for now, printing the intent.
        """
        logger.info(f"Triggering Self-Healing protocols due to low score: {score}")
        # Logic: 
        # 1. Identify which queries failed.
        # 2. Spin up DeepResearchAgent for those queries.
        # 3. Save findings as new knowledge files.
        # 4. Re-ingest.
        pass

    def run_continuously(self, interval_seconds: int = 60):
        """
        Blocking method to run the orchestrator loop.
        """
        print(f"Starting Pipeline Orchestrator... Watching {self.data_path}")
        try:
            while True:
                status = self.monitor_and_update()
                if status['status'] != 'idle':
                    print(f"[Orchestrator] Update: {status}")
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("Stopping Orchestrator.")
